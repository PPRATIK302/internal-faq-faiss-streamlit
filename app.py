import hashlib
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import streamlit as st

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_SCHEMA_VERSION = "v2"
NO_ANSWER_TEXT = "I don't know based on the available documents."

CATEGORIES = [
    "technical",
    "safety",
    "lbp_regulation",
    "business_tips",
    "insurance",
    "legal",
]

INDEX_ROOT = Path("faiss_indexes")
INDEX_ROOT.mkdir(exist_ok=True)

CORPUS_PATH = Path("placemakers_learn_corpus.txt")

HEADER_RE = re.compile(r"^(TITLE|CATEGORY|DATE|URL)\s*:\s*(.*)$")
DOC_MARKER_RE = re.compile(r"^DOC\s+\d+$", re.IGNORECASE)
MONTH_YEAR_RE = re.compile(r"^[A-Za-z]+\s+\d{4}$")
DAY_MONTH_YEAR_RE = re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$")
METADATA_LINE_RE = re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4},")
COMMENT_LINE_RE = re.compile(r"\bsays:\b", re.IGNORECASE)
CITATION_RE = re.compile(r"\[DOC\s+\d+\]")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9/_-]*")

FOOTER_BREAK_MARKERS = {
    "register to earn lbp points sign in",
    "you must be logged in to post a comment.",
}

SIMPLE_SKIP_LINES = {
    "facebook",
    "twitter",
    "share...",
    "share…",
}

MOJIBAKE_MARKERS = ("â", "Ã", "€", "™", "œ")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "the",
    "their",
    "there",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

CATEGORY_KEYWORDS = {
    "safety": {
        "asbestos",
        "demolition",
        "fall",
        "hazard",
        "hazards",
        "health",
        "ppe",
        "safety",
        "worksafe",
    },
    "lbp_regulation": {
        "acceptable",
        "building",
        "code",
        "codewords",
        "compliance",
        "consent",
        "e1",
        "e2",
        "h1",
        "lbp",
        "licence",
        "licensed",
        "nzs",
        "regulation",
    },
    "business_tips": {
        "business",
        "customer",
        "customers",
        "marketing",
        "pricing",
        "profit",
        "sales",
        "service",
    },
    "insurance": {
        "claim",
        "cover",
        "insurance",
        "insurer",
        "policy",
    },
    "legal": {
        "contract",
        "dispute",
        "legal",
        "liability",
        "warranty",
    },
    "technical": {
        "cladding",
        "concrete",
        "foundation",
        "gutter",
        "insulation",
        "moisture",
        "roof",
        "slab",
        "timber",
        "ventilation",
        "weatherboard",
        "window",
    },
}

BROAD_QUERY_HINTS = (
    "best practice",
    "best practices",
    "compare",
    "difference",
    "differences",
    "explain",
    "guide",
    "overview",
    "requirements",
    "summary",
    "summarize",
    "what are",
)

SAMPLE_QUESTIONS = [
    "What should I check when dealing with ground clearances around a slab?",
    "What does the corpus say about LBP responsibilities and compliance?",
    "How should I think about safety risks before starting demolition work?",
    "What are the main considerations when comparing cladding options?",
]

CATEGORY_LABELS = {
    "any": "Auto detect",
    "business_tips": "Business tips",
    "insurance": "Insurance",
    "lbp_regulation": "LBP regulation",
    "legal": "Legal",
    "safety": "Safety",
    "technical": "Technical",
}


def ensure_openai_key() -> None:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets.")
        st.stop()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        cleaned = normalize_whitespace(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def repair_mojibake(text: str) -> str:
    if not text or not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

    before = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    after = sum(repaired.count(marker) for marker in MOJIBAKE_MARKERS)
    return repaired if after < before else text


def normalize_category(value: str) -> str:
    cleaned = normalize_whitespace(value).lower().replace("&", "and").replace(" ", "_")
    aliases = {
        "business_tips": "business_tips",
        "insurance": "insurance",
        "lbp_regulation": "lbp_regulation",
        "legal": "legal",
        "safety": "safety",
        "technical": "technical",
    }
    return aliases.get(cleaned, "any")


def derive_title_from_url(url: str) -> str:
    if not url:
        return ""
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    if not slug:
        return ""

    words = []
    uppercase_terms = {"h1", "h3.2", "lbp", "nzs", "e1", "e2"}
    for part in slug.split("-"):
        if not part:
            continue
        lowered = part.lower()
        if lowered in uppercase_terms:
            words.append(lowered.upper())
        elif re.search(r"\d", part):
            words.append(part.upper())
        else:
            words.append(part.capitalize())
    return " ".join(words)


def extract_published_at(lines: List[str], header_date: str) -> Tuple[str, List[str]]:
    published_at = normalize_whitespace(repair_mojibake(header_date))
    working = list(lines)

    while working:
        first = normalize_whitespace(working[0])
        if not first:
            working.pop(0)
            continue
        if MONTH_YEAR_RE.match(first) or DAY_MONTH_YEAR_RE.match(first):
            published_at = published_at or first
            working.pop(0)
            continue
        if METADATA_LINE_RE.match(first):
            if not published_at:
                published_at = first.split(",", 1)[0].strip()
            working.pop(0)
            continue
        break

    return published_at, working


def clean_article_lines(lines: Sequence[str], title: str) -> List[str]:
    cleaned: List[str] = []
    seen_long_lines = set()
    title_normalized = normalize_whitespace(title).lower()

    for raw_line in lines:
        line = normalize_whitespace(repair_mojibake(raw_line))
        if not line:
            continue
        if DOC_MARKER_RE.match(line):
            continue

        lowered = line.lower()
        if lowered in FOOTER_BREAK_MARKERS:
            break
        if lowered in SIMPLE_SKIP_LINES:
            continue
        if lowered.startswith("share the post"):
            continue
        if COMMENT_LINE_RE.search(line):
            continue
        if title_normalized and lowered == title_normalized:
            continue

        if len(line) > 35:
            if line in seen_long_lines:
                continue
            seen_long_lines.add(line)

        cleaned.append(line)

    return cleaned


def compose_article_body(lines: Sequence[str]) -> str:
    paragraphs: List[str] = []
    buffer: List[str] = []

    for line in lines:
        buffer.append(line)
        joined = " ".join(buffer).strip()
        if (
            line.endswith((".", "?", "!", ":"))
            or len(buffer) >= 3
            or len(joined) >= 450
        ):
            paragraphs.append(joined)
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def parse_corpus(raw_text: str) -> List[Document]:
    raw_text = repair_mojibake(raw_text)
    blocks = [block.strip() for block in raw_text.split("=" * 80) if block.strip()]

    articles: List[Document] = []

    for article_number, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        metadata: Dict[str, str] = {}
        body_lines: List[str] = []
        body_started = False

        for line in lines:
            if DOC_MARKER_RE.match(line.strip()):
                continue
            header_match = HEADER_RE.match(line.strip())
            if header_match:
                metadata[header_match.group(1)] = header_match.group(2).strip()
                if header_match.group(1) == "URL":
                    body_started = True
                continue
            if body_started:
                body_lines.append(line)

        if not body_started:
            body_lines = lines

        title = normalize_whitespace(metadata.get("TITLE", "")) or derive_title_from_url(metadata.get("URL", "")) or f"Article {article_number}"
        category = normalize_category(metadata.get("CATEGORY", ""))
        published_at, body_lines = extract_published_at(body_lines, metadata.get("DATE", ""))
        cleaned_lines = clean_article_lines(body_lines, title)
        body = compose_article_body(cleaned_lines)

        if not body:
            continue

        article_key = f"{article_number}|{title}|{metadata.get('URL', '')}"
        article_id = hashlib.sha256(article_key.encode("utf-8")).hexdigest()[:12]
        article_metadata = {
            "article_id": article_id,
            "article_number": article_number,
            "title": title,
            "url": metadata.get("URL", "").strip(),
            "category": category,
            "published_at": published_at or "unknown",
        }
        articles.append(Document(page_content=body, metadata=article_metadata))

    return articles


def chunk_articles(articles: Sequence[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )

    chunks: List[Document] = []

    for article in articles:
        chunk_texts = splitter.split_text(article.page_content)
        total_chunks = len(chunk_texts)

        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            metadata = dict(article.metadata)
            metadata.update(
                {
                    "chunk_id": chunk_index,
                    "chunk_count": total_chunks,
                    "chunk_preview": normalize_whitespace(chunk_text)[:180],
                }
            )
            page_content = (
                f"Title: {metadata.get('title', 'Unknown title')}\n"
                f"Category: {metadata.get('category', 'any')}\n"
                f"Published: {metadata.get('published_at', 'unknown')}\n"
                f"URL: {metadata.get('url', '')}\n\n"
                f"{chunk_text.strip()}"
            )
            chunks.append(Document(page_content=page_content, metadata=metadata))

    return chunks


def build_vectorstore(documents: Sequence[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(list(documents), embeddings)


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _index_dir(file_bytes: bytes) -> Path:
    return INDEX_ROOT / f"{INDEX_SCHEMA_VERSION}_{_sha16(file_bytes)}"


def _has_saved_index(dir_path: Path) -> bool:
    return (dir_path / "index.faiss").exists() and (dir_path / "index.pkl").exists()


@st.cache_resource(show_spinner=False)
def load_or_build_vectordb(file_bytes: bytes) -> Tuple[FAISS, Dict[str, Any]]:
    ensure_openai_key()

    idx_dir = _index_dir(file_bytes)
    idx_dir.mkdir(parents=True, exist_ok=True)
    stats_path = idx_dir / "stats.json"

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    if _has_saved_index(idx_dir):
        vectordb = FAISS.load_local(
            folder_path=str(idx_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        stats: Dict[str, Any] = {}
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except Exception:
                stats = {}
        stats.setdefault("mode", "loaded")
        stats.setdefault("index_dir", str(idx_dir))
        stats.setdefault("sha16", _sha16(file_bytes))
        stats.setdefault("schema_version", INDEX_SCHEMA_VERSION)
        return vectordb, stats

    raw_text = file_bytes.decode("utf-8", errors="replace")
    articles = parse_corpus(raw_text)
    chunks = chunk_articles(articles)
    vectordb = build_vectorstore(chunks)
    vectordb.save_local(str(idx_dir))

    stats = {
        "mode": "built",
        "articles_indexed": len(articles),
        "chunks_indexed": len(chunks),
        "index_dir": str(idx_dir),
        "schema_version": INDEX_SCHEMA_VERSION,
        "sha16": _sha16(file_bytes),
    }
    try:
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except Exception:
        pass
    return vectordb, stats


@st.cache_resource(show_spinner=True)
def get_vectordb_from_local_corpus() -> Tuple[FAISS, Dict[str, Any]]:
    if not CORPUS_PATH.exists():
        st.error(f"Corpus file not found: {CORPUS_PATH.resolve()}")
        st.stop()

    return load_or_build_vectordb(CORPUS_PATH.read_bytes())


def get_llm() -> ChatOpenAI:
    ensure_openai_key()
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def extract_json_payload(text: str) -> Any:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
            break

    return json.loads(cleaned)


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    return [token for token in tokens if token not in STOPWORDS]


def guess_category_from_question(question: str) -> str:
    question_tokens = set(tokenize(question))
    category_scores = {
        category: len(question_tokens & keywords)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(category_scores.items(), key=lambda item: item[1], default=("any", 0))
    return best_category if best_score > 0 else "any"


def heuristic_query_analysis(question: str) -> Dict[str, Any]:
    lowered = question.lower()
    tokens = tokenize(question)
    is_broad = len(tokens) >= 9 or any(hint in lowered for hint in BROAD_QUERY_HINTS)
    return {
        "intent": "broad" if is_broad else "specific",
        "category": guess_category_from_question(question),
        "multi_query": is_broad or len(tokens) >= 5,
    }


def analyze_query(llm: ChatOpenAI, question: str) -> Dict[str, Any]:
    fallback = heuristic_query_analysis(question)
    system_msg = (
        "You are a query analyzer for a corpus-grounded FAQ assistant.\n"
        "Classify the user question.\n"
        f"Allowed categories: {', '.join(CATEGORIES)}, any.\n"
        "Return only valid JSON with keys: intent, category, multi_query.\n"
        "intent must be 'specific' or 'broad'."
    )

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Question: {question}"},
            ]
        )
        data = extract_json_payload(str(response.content))
    except Exception:
        return fallback

    if not isinstance(data, dict):
        return fallback

    intent = data.get("intent", fallback["intent"])
    category = data.get("category", fallback["category"])
    multi_query = bool(data.get("multi_query", fallback["multi_query"]))

    if intent not in {"specific", "broad"}:
        intent = fallback["intent"]
    if category not in set(CATEGORIES) | {"any"}:
        category = fallback["category"]
    if category == "any" and fallback["category"] != "any":
        category = fallback["category"]

    return {
        "intent": intent,
        "category": category,
        "multi_query": multi_query or fallback["multi_query"],
    }


def generate_alternative_queries(llm: ChatOpenAI, question: str) -> List[str]:
    system_msg = (
        "Generate up to 2 concise search rewrites for retrieving corpus passages.\n"
        "Keep the meaning the same. Return only a JSON list of strings."
    )
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Question: {question}"},
            ]
        )
        data = extract_json_payload(str(response.content))
    except Exception:
        return []

    if not isinstance(data, list):
        return []
    return [item for item in unique_preserve_order(str(value) for value in data) if item != normalize_whitespace(question)]


def keyword_query(question: str) -> str:
    tokens = tokenize(question)
    keywords = [token for token in tokens if len(token) > 2][:8]
    return " ".join(keywords)


def query_phrases(question: str) -> List[str]:
    quoted = re.findall(r'"([^"]+)"', question)
    tokens = tokenize(question)
    bigrams = [" ".join(tokens[index:index + 2]) for index in range(max(0, len(tokens) - 1))]
    return unique_preserve_order(quoted + bigrams[:4])


def doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return f"{metadata.get('article_id', '')}:{metadata.get('chunk_id', '')}:{metadata.get('url', '')}"


def lexical_overlap_score(question: str, doc: Document) -> Tuple[float, float, float]:
    question_tokens = set(tokenize(question))
    if not question_tokens:
        return 0.0, 0.0, 0.0

    metadata = doc.metadata or {}
    doc_text = f"{metadata.get('title', '')} {doc.page_content}".lower()
    doc_tokens = set(tokenize(doc_text))
    title_tokens = set(tokenize(metadata.get("title", "")))

    overlap = len(question_tokens & doc_tokens) / len(question_tokens)
    title_overlap = len(question_tokens & title_tokens) / len(question_tokens)

    phrase_hits = 0
    lowered_text = doc_text
    for phrase in query_phrases(question):
        if phrase and phrase.lower() in lowered_text:
            phrase_hits += 1
    phrase_score = min(0.3, phrase_hits * 0.08)

    return overlap, title_overlap, phrase_score


def clone_with_metadata(doc: Document, extra_metadata: Dict[str, Any]) -> Document:
    metadata = dict(doc.metadata or {})
    metadata.update(extra_metadata)
    return Document(page_content=doc.page_content, metadata=metadata)


def retrieve_docs(
    vectordb: FAISS,
    question: str,
    llm: ChatOpenAI,
    intent: str,
    category: str,
    multi_query: bool,
) -> Tuple[List[Document], Dict[str, Any]]:
    final_k = 8 if intent == "broad" else 6
    dense_k = 14 if intent == "broad" else 10
    mmr_k = 8 if intent == "broad" else 6

    queries = [question]
    keyword_only = keyword_query(question)
    if keyword_only and keyword_only.lower() != normalize_whitespace(question).lower():
        queries.append(keyword_only)
    if multi_query:
        queries.extend(generate_alternative_queries(llm, question))
    queries = unique_preserve_order(queries)[:4]

    candidates: Dict[str, Dict[str, Any]] = {}

    for query in queries:
        dense_results = vectordb.similarity_search_with_score(query, k=dense_k)
        mmr_results = vectordb.max_marginal_relevance_search(query, k=mmr_k, fetch_k=max(18, dense_k * 2))

        for rank, (doc, distance) in enumerate(dense_results, start=1):
            key = doc_key(doc)
            candidate = candidates.setdefault(
                key,
                {
                    "doc": doc,
                    "rrf": 0.0,
                    "dense_hits": 0,
                    "mmr_hits": 0,
                    "best_distance": float("inf"),
                },
            )
            candidate["rrf"] += 1.0 / (50 + rank)
            candidate["dense_hits"] += 1
            candidate["best_distance"] = min(candidate["best_distance"], float(distance))

        for rank, doc in enumerate(mmr_results, start=1):
            key = doc_key(doc)
            candidate = candidates.setdefault(
                key,
                {
                    "doc": doc,
                    "rrf": 0.0,
                    "dense_hits": 0,
                    "mmr_hits": 0,
                    "best_distance": float("inf"),
                },
            )
            candidate["rrf"] += 1.0 / (60 + rank)
            candidate["mmr_hits"] += 1

    ranked_candidates = []
    for candidate in candidates.values():
        doc = candidate["doc"]
        metadata = doc.metadata or {}
        lexical, title_overlap, phrase_score = lexical_overlap_score(question, doc)
        category_bonus = 0.14 if category != "any" and metadata.get("category") == category else 0.0
        multi_hit_bonus = min(0.1, 0.03 * (candidate["dense_hits"] + candidate["mmr_hits"]))
        distance_bonus = 0.0
        if candidate["best_distance"] != float("inf"):
            distance_bonus = min(0.12, 0.12 / (1.0 + candidate["best_distance"]))

        hybrid_score = (
            candidate["rrf"]
            + 0.55 * lexical
            + 0.22 * title_overlap
            + phrase_score
            + category_bonus
            + multi_hit_bonus
            + distance_bonus
        )
        ranked_candidates.append(
            {
                **candidate,
                "hybrid_score": hybrid_score,
                "lexical_overlap": lexical,
                "title_overlap": title_overlap,
            }
        )

    ranked_candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)

    selected_docs: List[Document] = []
    chunks_per_article = defaultdict(int)

    for item in ranked_candidates:
        doc = item["doc"]
        metadata = doc.metadata or {}
        article_id = metadata.get("article_id", "")
        if chunks_per_article[article_id] >= 2:
            continue

        selected_docs.append(
            clone_with_metadata(
                doc,
                {
                    "_hybrid_score": round(item["hybrid_score"], 4),
                    "_lexical_overlap": round(item["lexical_overlap"], 4),
                    "_title_overlap": round(item["title_overlap"], 4),
                },
            )
        )
        chunks_per_article[article_id] += 1
        if len(selected_docs) >= final_k:
            break

    retrieval_meta = {
        "queries": queries,
        "top_matches": [
            {
                "title": item["doc"].metadata.get("title", "Unknown title"),
                "category": item["doc"].metadata.get("category", "any"),
                "chunk": f"{item['doc'].metadata.get('chunk_id', '?')}/{item['doc'].metadata.get('chunk_count', '?')}",
                "score": round(item["hybrid_score"], 4),
                "url": item["doc"].metadata.get("url", ""),
            }
            for item in ranked_candidates[: min(6, len(ranked_candidates))]
        ],
    }

    return selected_docs, retrieval_meta


def build_context_for_llm(docs: Sequence[Document]) -> str:
    chunks = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        chunks.append(
            f"[DOC {index}]\n"
            f"Title: {metadata.get('title', 'Unknown title')}\n"
            f"Category: {metadata.get('category', 'any')}\n"
            f"Published: {metadata.get('published_at', 'unknown')}\n"
            f"URL: {metadata.get('url', '')}\n"
            f"Chunk: {metadata.get('chunk_id', '?')}/{metadata.get('chunk_count', '?')}\n\n"
            f"{doc.page_content}\n"
        )
    return "\n\n".join(chunks)


def _is_idk(answer: str) -> bool:
    lowered = normalize_whitespace(answer).lower()
    if not lowered:
        return True
    markers = [
        "i don't know based on the available documents",
        "i dont know based on the available documents",
        "i don't know based on the context",
        "i dont know based on the context",
        "not enough information",
        "no relevant info",
    ]
    return any(marker in lowered for marker in markers)


def _has_citations(answer: str) -> bool:
    return bool(CITATION_RE.search(answer or ""))


def answer_question(llm: ChatOpenAI, docs: Sequence[Document], question: str, intent: str) -> str:
    context = build_context_for_llm(docs)
    mode_instruction = (
        "Synthesize the main themes, differences, and practical implications across the excerpts."
        if intent == "broad"
        else "Answer the user's specific question directly and practically."
    )

    def run_prompt(force_citations: bool) -> str:
        citation_rule = (
            "Every paragraph or bullet must include at least one citation like [DOC 2]."
            if force_citations
            else "Cite all substantive claims with [DOC n]."
        )
        system_msg = (
            "You are an internal FAQ assistant for PlaceMakers.\n"
            "Use only the provided corpus excerpts.\n"
            "Turn the excerpts into the most practical real-world answer possible without adding outside facts.\n"
            "If the excerpts contain considerations but not a full procedure, provide a practical checklist based only on the excerpts and clearly say what is missing.\n"
            f"{citation_rule}\n"
            f"If the excerpts do not contain enough relevant information, reply exactly: {NO_ANSWER_TEXT}"
        )
        user_msg = (
            f"Question: {question}\n"
            f"Guidance: {mode_instruction}\n\n"
            "Output requirements:\n"
            "- Start with a direct answer.\n"
            "- Use short bullets if they improve clarity.\n"
            "- Mention any important limitation in one sentence starting with 'Corpus gap:' if needed.\n"
            "- Do not mention information that is not supported by the excerpts.\n\n"
            f"Context:\n{context}"
        )
        response = llm.invoke(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )
        return str(response.content).strip()

    answer = run_prompt(force_citations=False)
    if _is_idk(answer):
        return answer
    if not _has_citations(answer):
        answer = run_prompt(force_citations=True)
    return answer.strip() or NO_ANSWER_TEXT


def list_sources_markdown(docs: Sequence[Document]) -> str:
    seen = set()
    lines = []
    for doc in docs:
        metadata = doc.metadata or {}
        title = metadata.get("title", "Unknown title")
        url = metadata.get("url", "")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        if url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def render_retrieval_debug(info: Dict[str, Any], intent: str, category: str) -> None:
    with st.expander("Retrieval diagnostics", expanded=False):
        st.write(f"Intent: {intent}")
        st.write(f"Category hint: {category}")
        st.write("Query variants:")
        for query in info.get("queries", []):
            st.write(f"- {query}")
        st.write("Top retrieved chunks:")
        for match in info.get("top_matches", []):
            st.write(
                f"- {match['title']} | {match['category']} | chunk {match['chunk']} | score {match['score']}"
            )


def init_app_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_question", "")


def queue_question(question: str) -> None:
    st.session_state["pending_question"] = question.strip()


def clear_conversation() -> None:
    st.session_state["chat_history"] = []
    st.session_state["pending_question"] = ""


def format_category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def format_intent_label(intent: str) -> str:
    return "Broader overview" if intent == "broad" else "Specific answer"


def collect_source_details(docs: Sequence[Document]) -> List[Dict[str, str]]:
    seen = set()
    sources: List[Dict[str, str]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        title = metadata.get("title", "Unknown title")
        url = metadata.get("url", "")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": title,
                "url": url,
                "category": format_category_label(metadata.get("category", "any")),
                "published_at": metadata.get("published_at", "unknown"),
            }
        )
    return sources


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-text: #0f172a;
            --app-muted: #475569;
            --app-border: rgba(148, 163, 184, 0.24);
            --app-accent: #ea580c;
            --app-accent-deep: #9a3412;
            --app-accent-soft: #ffedd5;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 24%),
                linear-gradient(180deg, #fffaf5 0%, #f8fafc 42%, #f1f5f9 100%);
            color: var(--app-text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
            border-right: 1px solid rgba(234, 88, 12, 0.12);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
        }

        .hero-shell {
            background: linear-gradient(
                135deg,
                rgba(255, 247, 237, 0.92) 0%,
                rgba(255, 255, 255, 0.96) 62%,
                rgba(241, 245, 249, 0.92) 100%
            );
            border: 1px solid rgba(234, 88, 12, 0.14);
            border-radius: 24px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .hero-eyebrow {
            margin: 0 0 0.45rem 0;
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--app-accent);
        }

        .hero-title {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
            color: var(--app-text);
        }

        .hero-copy {
            margin: 0.85rem 0 0 0;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 760px;
            color: var(--app-muted);
        }

        .hero-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.2rem;
        }

        .stat-chip {
            min-width: 155px;
            padding: 0.8rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(234, 88, 12, 0.14);
            background: rgba(255, 255, 255, 0.82);
        }

        .stat-chip span {
            display: block;
            font-size: 0.82rem;
            color: var(--app-muted);
        }

        .stat-chip strong {
            display: block;
            margin-top: 0.2rem;
            font-size: 1.15rem;
            color: var(--app-text);
        }

        .helper-note,
        .empty-state {
            border: 1px solid var(--app-border);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.04);
        }

        .helper-note {
            padding: 0.95rem 1rem;
            margin-bottom: 1rem;
            color: var(--app-muted);
            line-height: 1.55;
        }

        .empty-state {
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }

        .empty-state h3 {
            margin: 0 0 0.4rem 0;
            color: var(--app-text);
        }

        .empty-state p {
            margin: 0;
            color: var(--app-muted);
            line-height: 1.55;
        }

        .source-card {
            border: 1px solid var(--app-border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: rgba(255, 255, 255, 0.86);
            min-height: 150px;
        }

        .source-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            font-size: 0.8rem;
            color: var(--app-muted);
            margin-bottom: 0.65rem;
        }

        .source-title {
            margin: 0;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.45;
            color: var(--app-text);
        }

        .source-link {
            display: inline-flex;
            align-items: center;
            margin-top: 0.85rem;
            color: var(--app-accent-deep);
            font-weight: 600;
            text-decoration: none;
        }

        .source-link:hover {
            color: var(--app-accent);
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(234, 88, 12, 0.2);
            background: rgba(255, 255, 255, 0.92);
            color: var(--app-accent-deep);
            font-weight: 600;
        }

        div.stButton > button:hover {
            border-color: rgba(234, 88, 12, 0.42);
            color: #7c2d12;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--app-border);
            border-radius: 22px;
            padding: 0.15rem 0.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.04);
            backdrop-filter: blur(10px);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid var(--app-border);
            border-radius: 18px;
            padding: 0.8rem 0.95rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--app-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--app-text);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--app-border);
            border-radius: 18px;
            background: rgba(248, 250, 252, 0.76);
        }

        [data-testid="stChatInput"] {
            background: rgba(255, 255, 255, 0.92);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(stats: Dict[str, Any]) -> None:
    article_count = stats.get("articles_indexed", "n/a")
    chunk_count = stats.get("chunks_indexed", "n/a")
    mode_label = str(stats.get("mode", "ready")).title()
    st.markdown(
        f"""
        <section class="hero-shell">
            <p class="hero-eyebrow">Internal knowledge assistant</p>
            <h1 class="hero-title">Find grounded answers from the PlaceMakers LEARN corpus</h1>
            <p class="hero-copy">
                Ask in plain language, review the supporting articles, and keep useful questions close at hand.
                The app stays anchored to the indexed LEARN material instead of guessing.
            </p>
            <div class="hero-stats">
                <div class="stat-chip">
                    <span>Articles indexed</span>
                    <strong>{html.escape(str(article_count))}</strong>
                </div>
                <div class="stat-chip">
                    <span>Search chunks</span>
                    <strong>{html.escape(str(chunk_count))}</strong>
                </div>
                <div class="stat-chip">
                    <span>Index mode</span>
                    <strong>{html.escape(mode_label)}</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(stats: Dict[str, Any]) -> bool:
    with st.sidebar:
        st.markdown("### Workspace")
        st.button(
            "Start new conversation",
            use_container_width=True,
            on_click=clear_conversation,
        )
        show_debug = st.checkbox("Show retrieval diagnostics", value=False)

        st.markdown(
            """
            <div class="helper-note">
                Questions work best when you mention the building element, regulation, product, or scenario you are working through.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Knowledge base")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Articles", stats.get("articles_indexed", "n/a"))
        metric_col2.metric("Chunks", stats.get("chunks_indexed", "n/a"))
        st.caption(
            f"Schema {stats.get('schema_version', INDEX_SCHEMA_VERSION)} | Mode {stats.get('mode', 'unknown')}"
        )

        st.markdown("### Suggested prompts")
        for index, sample in enumerate(SAMPLE_QUESTIONS):
            st.button(
                sample,
                key=f"sidebar_sample_{index}",
                use_container_width=True,
                on_click=queue_question,
                args=(sample,),
            )

        if st.session_state["chat_history"]:
            st.markdown("### Recent questions")
            for index, item in enumerate(reversed(st.session_state["chat_history"][-5:]), start=1):
                preview = item["question"]
                if len(preview) > 70:
                    preview = f"{preview[:67]}..."
                st.button(
                    preview,
                    key=f"recent_question_{index}",
                    use_container_width=True,
                    on_click=queue_question,
                    args=(item["question"],),
                )

    return show_debug


def render_empty_state() -> None:
    st.markdown(
        """
        <section class="empty-state">
            <h3>Start with a practical question</h3>
            <p>
                You can ask for a direct answer, a comparison, or a quick overview. Answers will cite the
                retrieved evidence when the corpus supports the response.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    prompt_col1, prompt_col2 = st.columns(2)
    for index, sample in enumerate(SAMPLE_QUESTIONS):
        column = prompt_col1 if index % 2 == 0 else prompt_col2
        column.button(
            sample,
            key=f"welcome_sample_{index}",
            use_container_width=True,
            on_click=queue_question,
            args=(sample,),
        )


def run_assistant_query(vectordb: FAISS, question: str) -> Dict[str, Any]:
    llm = get_llm()

    with st.spinner("Analyzing the question and retrieving the strongest evidence..."):
        analysis = analyze_query(llm, question)
        intent = analysis["intent"]
        category = analysis["category"]
        multi_query = analysis["multi_query"]
        docs, retrieval_info = retrieve_docs(vectordb, question, llm, intent, category, multi_query)

    if not docs:
        return {
            "question": question,
            "answer": (
                "I could not find a strong match in the LEARN corpus for that question yet. "
                "Try adding the product, code reference, building element, or situation you care about."
            ),
            "intent": intent,
            "category": category,
            "retrieval_info": retrieval_info,
            "sources": [],
            "status": "no_results",
        }

    with st.spinner("Writing a grounded answer..."):
        answer = answer_question(llm, docs, question, intent)

    sources = [] if _is_idk(answer) else collect_source_details(docs)
    return {
        "question": question,
        "answer": answer,
        "intent": intent,
        "category": category,
        "retrieval_info": retrieval_info,
        "sources": sources,
        "status": "needs_more_evidence" if _is_idk(answer) else "answered",
    }


def render_source_cards(sources: Sequence[Dict[str, str]]) -> None:
    columns = st.columns(2)
    for index, source in enumerate(sources):
        column = columns[index % 2]
        title = html.escape(source.get("title", "Unknown title"))
        category = html.escape(source.get("category", "Auto detect"))
        published_at = html.escape(source.get("published_at", "unknown"))
        url = source.get("url", "")
        link_markup = (
            f'<a class="source-link" href="{html.escape(url)}" target="_blank">Open article</a>'
            if url
            else '<span class="source-link">URL not available</span>'
        )
        column.markdown(
            f"""
            <article class="source-card">
                <div class="source-meta">
                    <span>{category}</span>
                    <span>{published_at}</span>
                </div>
                <p class="source-title">{title}</p>
                {link_markup}
            </article>
            """,
            unsafe_allow_html=True,
        )


def render_chat_history(entries: Sequence[Dict[str, Any]], show_debug: bool) -> None:
    total_entries = len(entries)
    for index, entry in enumerate(entries, start=1):
        with st.chat_message("user"):
            st.markdown(entry["question"])

        with st.chat_message("assistant"):
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Question type", format_intent_label(entry["intent"]))
            metric_col2.metric("Topic", format_category_label(entry["category"]))
            metric_col3.metric("Source articles", len(entry.get("sources", [])))

            answer = entry.get("answer", "")
            status = entry.get("status", "answered")
            if status == "no_results":
                st.warning(answer)
            elif _is_idk(answer):
                st.info(answer)
            else:
                st.markdown(answer)

            if entry.get("sources"):
                with st.expander("Source articles", expanded=index == total_entries):
                    render_source_cards(entry["sources"])
            elif status == "no_results":
                st.caption(
                    "Try adding more detail, such as a product name, code clause, material, or construction scenario."
                )
            else:
                st.caption(
                    "No source links are shown when the corpus cannot support a grounded answer with enough confidence."
                )

            if show_debug:
                render_retrieval_debug(
                    entry.get("retrieval_info", {}),
                    entry.get("intent", "specific"),
                    entry.get("category", "any"),
                )


def main() -> None:
    st.set_page_config(
        page_title="PlaceMakers LEARN RAG Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_app_state()
    apply_custom_styles()
    ensure_openai_key()
    vectordb, stats = get_vectordb_from_local_corpus()
    show_debug = render_sidebar(stats)

    render_hero(stats)
    st.caption(
        "Answers stay grounded in the indexed LEARN material and surface the supporting articles when confidence is strong enough."
    )

    queued_question = st.session_state.pop("pending_question", "").strip()
    typed_question = st.chat_input(
        "Ask about compliance, safety, technical guidance, insurance, legal issues, or business tips..."
    )
    question = queued_question or (typed_question or "").strip()

    if question:
        st.session_state["chat_history"].append(run_assistant_query(vectordb, question))

    if not st.session_state["chat_history"]:
        render_empty_state()

    render_chat_history(st.session_state["chat_history"], show_debug)


if __name__ == "__main__":
    main()

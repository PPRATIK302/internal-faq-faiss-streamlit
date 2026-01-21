import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

import streamlit as st

# ✅ LangChain splitter import compatibility
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # or "gpt-4o"

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

# ============================================================
# KEY HANDLING
# ============================================================

def ensure_openai_key():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets.")
        st.stop()

# ============================================================
# STEP 1 – LOAD + PARSE TXT CORPUS
# ============================================================

def load_and_parse_txt(raw_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    blocks = [b.strip() for b in raw_text.split("=" * 80) if b.strip()]

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]

        title, url, category = "", "", ""
        body_start = 0

        for i, ln in enumerate(lines):
            if ln.startswith("TITLE"):
                title = ln.split(":", 1)[1].strip()
            elif ln.startswith("URL"):
                url = ln.split(":", 1)[1].strip()
                body_start = i + 1
            elif ln.startswith("CATEGORY"):
                category = ln.split(":", 1)[1].strip()

        if body_start == 0:
            body_start = 3 if len(lines) > 3 else 0

        body = "\n".join(lines[body_start:]).strip()
        if not body:
            continue

        full_text = f"Title: {title}\nCategory: {category}\nURL: {url}\n\n{body}"
        texts.append(full_text)
        metadatas.append({"title": title, "url": url, "category": category})

    return texts, metadatas

# ============================================================
# STEP 2 – BUILD VECTOR STORE
# ============================================================

def build_vectorstore(texts: List[str], metadatas: List[Dict[str, Any]]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

# ============================================================
# FAISS PERSISTENCE HELPERS
# ============================================================

def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]

def _index_dir(file_bytes: bytes) -> Path:
    return INDEX_ROOT / _sha16(file_bytes)

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
        stats = {}
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except Exception:
                stats = {}
        stats.setdefault("mode", "loaded")
        stats.setdefault("index_dir", str(idx_dir))
        stats.setdefault("sha16", _sha16(file_bytes))
        return vectordb, stats

    raw_text = file_bytes.decode("utf-8", errors="replace")
    texts, metadatas = load_and_parse_txt(raw_text)
    vectordb = build_vectorstore(texts, metadatas)
    vectordb.save_local(str(idx_dir))

    stats = {
        "mode": "built",
        "articles_indexed": len(texts),
        "index_dir": str(idx_dir),
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

    file_bytes = CORPUS_PATH.read_bytes()
    return load_or_build_vectordb(file_bytes)

# ============================================================
# STEP 3 – QUERY ANALYSIS & MULTI-QUERY EXPANSION
# ============================================================

def get_llm() -> ChatOpenAI:
    ensure_openai_key()
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)

def analyze_query(llm: ChatOpenAI, question: str) -> Dict[str, Any]:
    system_msg = (
        "You are a query analyzer for an internal FAQ assistant.\n"
        "Decide if the user's question is broad or specific.\n"
        "Also select a best-fit category from: "
        f"{', '.join(CATEGORIES)}. If not sure, use 'any'.\n"
        "Decide if multi-query expansion is helpful (true/false).\n"
        "Output ONLY valid JSON with keys: intent, category, multi_query."
    )

    resp = llm.invoke(
        [{"role": "system", "content": system_msg},
         {"role": "user", "content": f"Question: {question}"}]
    )

    text = (resp.content or "").strip()
    try:
        data = json.loads(text)
        intent = data.get("intent", "specific")
        category = data.get("category", "any")
        multi_query = bool(data.get("multi_query", False))
    except Exception:
        intent, category, multi_query = "specific", "any", False

    if intent not in {"specific", "broad"}:
        intent = "specific"
    if category not in CATEGORIES:
        category = "any"

    return {"intent": intent, "category": category, "multi_query": multi_query}

def generate_alternative_queries(llm: ChatOpenAI, question: str) -> List[str]:
    system_msg = (
        "Generate 2 alternate phrasings of the question with the same meaning. "
        "Return ONLY a JSON list of strings."
    )
    resp = llm.invoke(
        [{"role": "system", "content": system_msg},
         {"role": "user", "content": f"Original question: {question}"}]
    )

    text = (resp.content or "").strip()
    try:
        alts = json.loads(text)
        if isinstance(alts, list):
            return [str(q) for q in alts if isinstance(q, str) and q.strip()]
    except Exception:
        pass
    return []

# ============================================================
# STEP 4 – RETRIEVAL WITH CATEGORY-AWARE RE-RANKING
# ============================================================

def retrieve_docs(
    vectordb: FAISS,
    question: str,
    llm: ChatOpenAI,
    intent: str,
    category: str,
    multi_query: bool,
) -> List[Document]:
    base_k_specific = 6
    base_k_broad = 10
    base_k = base_k_broad if intent == "broad" else base_k_specific

    queries = [question]
    if multi_query:
        queries.extend(generate_alternative_queries(llm, question))

    all_docs: List[Document] = []
    seen_ids = set()

    for q in queries:
        docs = vectordb.similarity_search(q, k=base_k)
        for d in docs:
            key = (
                d.metadata.get("title", ""),
                d.metadata.get("url", ""),
                (d.page_content or "")[:200],
            )
            if key not in seen_ids:
                seen_ids.add(key)
                all_docs.append(d)

    if category != "any":
        cat_docs = [d for d in all_docs if d.metadata.get("category") == category]
        if len(cat_docs) >= 3:
            others = [d for d in all_docs if d not in cat_docs]
            all_docs = cat_docs + others

    max_docs = base_k_broad if intent == "broad" else base_k_specific
    return all_docs[:max_docs]

# ============================================================
# STEP 5 – ANSWERING (FIX: GROUNDED FALLBACK IF IDK)
# ============================================================

def build_context_for_llm(docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        chunks.append(
            f"[DOC {i}]\n"
            f"Title: {meta.get('title', 'Unknown title')}\n"
            f"Category: {meta.get('category', 'unknown')}\n"
            f"URL: {meta.get('url', 'no-url')}\n\n"
            f"{d.page_content}\n"
        )
    return "\n\n".join(chunks)

def _is_idk(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    markers = [
        "i don't know based on the available documents",
        "i dont know based on the available documents",
        "i don't know based on the context",
        "i dont know based on the context",
        "i don't know",
        "i dont know",
        "no relevant info",
        "not enough information",
    ]
    return any(m in a for m in markers)

def answer_specific(llm: ChatOpenAI, docs: List[Document], question: str) -> str:
    context = build_context_for_llm(docs)

    # Pass 1 (strict)
    system_msg = (
        "You are an internal FAQ assistant for a construction company.\n"
        "Answer using ONLY the provided context.\n\n"
        "If the answer is not clearly contained in the context, say exactly:\n"
        "\"I don't know based on the available documents.\"\n"
        "Do NOT invent details."
    )
    resp = llm.invoke(
        [{"role": "system", "content": system_msg},
         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    answer = (resp.content or "").strip()

    # ✅ FIX: If docs exist but model still says IDK, force grounded "best possible" answer
    if _is_idk(answer) and docs:
        fallback_system = (
            "You are an internal FAQ assistant for a construction company.\n"
            "You MUST use ONLY the provided context and MUST NOT invent.\n\n"
            "Important: If the documents contain related guidance but not a full step-by-step procedure, "
            "you should still answer with what IS available and clearly say what is missing.\n\n"
            "Only reply with:\n"
            "\"I don't know based on the available documents.\"\n"
            "if the context contains NOTHING relevant."
        )
        fallback_user = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Write a helpful answer grounded strictly in the context.\n"
            "- If 'how to install' steps are not explicitly given, provide the relevant considerations/checklist from the text.\n"
            "- Clearly state limitations (e.g., 'these documents discuss considerations, not full installation steps')."
        )
        resp2 = llm.invoke(
            [{"role": "system", "content": fallback_system},
             {"role": "user", "content": fallback_user}]
        )
        answer2 = (resp2.content or "").strip()
        if answer2 and not _is_idk(answer2):
            return answer2

    return answer

def answer_broad(llm: ChatOpenAI, docs: List[Document], question: str) -> str:
    partials = []
    for d in docs:
        meta = d.metadata or {}
        system_msg = (
            "Summarize ONE document for answering the question.\n"
            "Use ONLY the given document.\n"
            "If it doesn't help, say: \"No relevant info in this document.\""
        )
        user_msg = (
            f"Document title: {meta.get('title', 'Unknown title')}\n"
            f"Category: {meta.get('category', 'unknown')}\n\n"
            f"Document content:\n{d.page_content}\n\n"
            f"Question: {question}\n\n"
            "Write a brief partial answer (3–6 sentences) grounded ONLY in this document."
        )
        resp = llm.invoke(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}]
        )
        partials.append((resp.content or "").strip())

    combined = "\n\n---\n\n".join(partials)
    system_msg = (
        "Combine partial answers into one coherent answer.\n"
        "Remove duplicates.\n"
        "If there is effectively no relevant info, reply exactly:\n"
        "\"I don't know based on the available documents.\""
    )
    resp = llm.invoke(
        [{"role": "system", "content": system_msg},
         {"role": "user", "content": f"Question: {question}\n\nPartial answers:\n{combined}"}]
    )
    answer = (resp.content or "").strip()

    # ✅ Same grounded fallback for broad mode
    if _is_idk(answer) and docs:
        context = build_context_for_llm(docs)
        fallback_system = (
            "You MUST use ONLY the provided context and MUST NOT invent.\n"
            "If docs contain partial/related guidance, summarize it and state limitations.\n"
            "Only reply with \"I don't know based on the available documents.\" if nothing relevant exists."
        )
        resp2 = llm.invoke(
            [{"role": "system", "content": fallback_system},
             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        )
        answer2 = (resp2.content or "").strip()
        if answer2 and not _is_idk(answer2):
            return answer2

    return answer

# ============================================================
# UI – LINKS LIST
# ============================================================

def list_sources_markdown(docs: List[Document]) -> str:
    seen = set()
    lines = []
    for d in docs:
        meta = d.metadata or {}
        title = meta.get("title", "Unknown title")
        url = meta.get("url", "no-url")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        if url and url != "no-url":
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)

# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="PlaceMakers Under-Construction Regulations & Compliance Assistant",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 PlaceMakers Under-Construction Regulations & Compliance Assistant")
    st.caption("Internal FAQ Assistant – PlaceMakers LEARN (FAISS, Auto Corpus)")

    st.sidebar.empty()

    # ✅ Added: brief user guideline block
    with st.expander("📘 Guidelines: How to use this application", expanded=True):
        st.markdown(
            """
**How to use**
1. **Type your question** in the box (keep it short and specific).
2. Click **Get Answer**.
3. Read the **Answer** generated strictly from the LEARN corpus.
4. If source links appear, open them to verify the original article.

**Tips for better results**
- Use **keywords**: e.g., *unheated slab, insulation, consent, LBP, H1, moisture, cladding, PPE*.
- If the answer is weak, **rephrase** the question using simpler words.

**Important**
- If content is not present in documents, the assistant responds:  
  **“I don't know based on the available documents.”**
- ✅ If the answer is **IDK**, the app will **NOT** show source links.
            """
        )

    ensure_openai_key()

    with st.spinner("Loading corpus and building/loading FAISS index..."):
        vectordb, _ = get_vectordb_from_local_corpus()

    st.subheader("Ask a question")
    question = st.text_input(
        "Ask anything based on the LEARN content:",
        placeholder="e.g., How to install an unheated slab?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("Get Answer")
    with col2:
        st.caption("Tip: Keep questions specific for best results.")

    if ask:
        if not question.strip():
            st.warning("Please type a question.")
            return

        llm = get_llm()

        with st.spinner("Analyzing query and retrieving context..."):
            analysis = analyze_query(llm, question)
            intent = analysis["intent"]
            category = analysis["category"]
            multi_query = analysis["multi_query"]
            docs = retrieve_docs(vectordb, question, llm, intent, category, multi_query)

        if not docs:
            st.warning("No relevant documents found. Try rephrasing your question.")
            return

        with st.spinner("Generating answer..."):
            answer = answer_broad(llm, docs, question) if intent == "broad" else answer_specific(llm, docs, question)

        st.markdown("### ✅ Answer")
        st.write(answer)

        # ✅ Requirement: if IDK, do not show links
        if not _is_idk(answer):
            st.markdown("### 🔗 Source Articles (Links Only)")
            st.caption("These LEARN articles were used to answer your question:")
            st.markdown(list_sources_markdown(docs))

if __name__ == "__main__":
    main()


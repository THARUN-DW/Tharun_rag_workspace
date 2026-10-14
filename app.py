"""
app.py
======
Streamlit front-end for the generic RAG chatbot.

The UI is completely document-agnostic: whatever file the user uploads
(PDF / DOCX / TXT / CSV / MD) flows through the SAME pipeline
(document_loader -> chunker -> vector_store -> rag_pipeline).

Nothing here knows anything about the CONTENT of the documents. No document
text is hard-coded anywhere in this project.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

# Make sibling modules importable no matter where streamlit is launched from.
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from document_loader import DocumentLoadError, is_supported, load_document  # noqa: E402
from rag_pipeline import (  # noqa: E402
    MissingAPIKeyError,
    RAGPipeline,
    RAGPipelineError,
    get_gemini_client,
)
from vector_store import DEFAULT_COLLECTION_NAME, VectorStore, VectorStoreError  # noqa: E402

# ---------------------------------------------------------------------------
# Page + shared state
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Generic RAG Chatbot", page_icon="📚", layout="wide")

st.title("📚 Generic RAG Chatbot")
st.caption(
    "Upload any document (PDF, DOCX, TXT, CSV, Markdown) and ask questions "
    "about it. Answers come only from your uploaded documents."
)


def get_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# Chat history lives in session state: [{"role": "user"/"assistant",
# "content": str, "sources": [...] } ]
get_state("chat_history", [])

# ---------------------------------------------------------------------------
# Heavy resources are cached so Streamlit re-runs don't rebuild them
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the vector database and embedding model ...")
def get_vector_store() -> VectorStore:
    """One VectorStore (ChromaDB + MiniLM embeddings) per app process."""
    return VectorStore(collection_name=DEFAULT_COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """One RAG pipeline bound to that store; the LLM client is created lazily."""
    return RAGPipeline(vector_store=get_vector_store())


try:
    store = get_vector_store()
    pipeline = get_pipeline()
except VectorStoreError as exc:
    st.error(f"Failed to start the vector database: {exc}")
    st.stop()
except Exception as exc:  # never crash the app on startup problems
    st.error(f"Unexpected startup error: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: upload section, uploaded document list, indexing status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Documents")

    st.subheader("Upload a document")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT, CSV, Markdown",
        type=["pdf", "docx", "txt", "csv", "md", "markdown"],
        accept_multiple_files=True,
    )

    if st.button("🗑️ Clear all documents", use_container_width=True):
        try:
            for doc in store.list_documents():
                store.delete_document(doc["filename"])
            st.success("All documents removed from the index.")
            st.rerun()
        except VectorStoreError as exc:
            st.error(f"Could not clear the vector store: {exc}")

    # ---------------- indexing status ----------------
    try:
        total_chunks = store.count()
    except VectorStoreError as exc:
        st.error(f"Vector database error: {exc}")
        total_chunks = 0

    if total_chunks == 0:
        st.info("Index is empty — upload a document to begin.")
    else:
        st.subheader("Uploaded documents (indexed)")
        try:
            docs = store.list_documents()
        except VectorStoreError as exc:
            st.error(f"Could not list documents: {exc}")
            docs = []
        for doc in docs:
            pages = f" · {doc['pages']} pages" if doc.get("pages") else ""
            st.markdown(
                f"- **{doc['filename']}** \n"
                f"  `{doc['doc_type']}` · {doc['chunks']} chunks{pages}"
            )
        st.caption(f"Total chunks indexed: **{total_chunks}**")

# ---------------------------------------------------------------------------
# Upload handling: detect type -> extract -> chunk -> embed -> store
# ---------------------------------------------------------------------------


def index_file(uploaded_file) -> None:
    """Full ingestion of one uploaded file, with UI feedback and no crashes."""
    name = uploaded_file.name
    with st.sidebar:
        box = st.container()
    with box:
        if not is_supported(name):
            st.error(
                f"❌ {name}: unsupported file format. Use PDF, DOCX, TXT, CSV or MD."
            )
            return

        placeholder = st.empty()
        placeholder.info(f"⏳ Indexing **{name}** ...")

        try:
            # 1) detect type + extract text (raises DocumentLoadError subclasses)
            sections = load_document(name, uploaded_file.getvalue())

            # 2) split into meaningful chunks (deterministic ids)
            from chunker import chunk_document  # local import keeps startup fast

            chunks = chunk_document(sections)

            # 3) generate embeddings + 4) persist text/vectors/metadata in ChromaDB
            stored = store.add_chunks(chunks)

            placeholder.success(
                f"✅ **{name}** indexed: {stored} chunks "
                f"({len(sections)} sections extracted)."
            )
        except DocumentLoadError as exc:
            placeholder.error(f"❌ {name}: {exc}")
        except VectorStoreError as exc:
            placeholder.error(f"❌ {name}: vector database error — {exc}")
        except Exception as exc:
            placeholder.error(f"❌ {name}: unexpected error — {exc}")
            st.sidebar.expander("Technical details", expanded=False).code(
                traceback.format_exc()
            )


if uploaded_files:
    for uploaded_file in uploaded_files:
        index_file(uploaded_file)

# ---------------------------------------------------------------------------
# Chat UI: history, input, assistant answer, retrieved sources
# ---------------------------------------------------------------------------

st.subheader("💬 Chat")

# Render the existing conversation (including per-answer sources).
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 Retrieved sources"):
                for i, src in enumerate(message["sources"], start=1):
                    page = src.get("page")
                    page_s = f"Page {page}" if page is not None else "Page n/a"
                    st.markdown(
                        f"**{i}. `{src.get('filename')}`** — {page_s} · "
                        f"Chunk {src.get('chunk_index')} · "
                        f"Similarity **{src.get('similarity')}%**"
                        f" ({src.get('doc_type')})"
                    )
                    st.caption(
                        (src.get("text") or "")[:400]
                        + ("..." if len(src.get("text") or "") > 400 else "")
                    )

user_input = st.chat_input("Ask a question about your uploaded documents ...")

if user_input:
    # Append + render the user's message.
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate the answer with retrieval for EVERY question.
    with st.chat_message("assistant"):
        if store.count() == 0:
            st.warning(
                "No documents are indexed yet — upload one in the sidebar first."
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "No documents indexed yet."}
            )
        else:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]
            ]
            answer_text = ""
            sources: list = []

            with st.spinner("Searching your documents and thinking ..."):
                try:
                    result = pipeline.run(user_input, history=history)
                    answer_text = result["answer"]
                    sources = result["sources"]

                except MissingAPIKeyError as exc:
                    answer_text = (
                        "⚠️ No Gemini API key configured. Add GEMINI_API_KEY to "
                        "your .env file and restart the app."
                    )
                    st.error(str(exc))

                except RAGPipelineError as exc:
                    answer_text = f"⚠️ Something went wrong while answering: {exc}"
                    st.error(str(exc))

                except VectorStoreError as exc:
                    answer_text = "⚠️ The vector database is unavailable right now."
                    st.error(f"Vector database error: {exc}")

                except Exception as exc:
                    answer_text = "⚠️ Unexpected error while answering."
                    st.error(f"Unexpected error: {exc}")
                    st.expander("Technical details", expanded=False).code(
                        traceback.format_exc()
                    )

            st.markdown(answer_text)

            if sources:
                with st.expander("🔍 Retrieved sources"):
                    for i, src in enumerate(sources, start=1):
                        page = src.get("page")
                        page_s = f"Page {page}" if page is not None else "Page n/a"
                        st.markdown(
                            f"**{i}. `{src.get('filename')}`** — {page_s} · "
                            f"Chunk {src.get('chunk_index')} · "
                            f"Similarity **{src.get('similarity')}%**"
                            f" ({src.get('doc_type')})"
                        )
                        st.caption(
                            (src.get("text") or "")[:400]
                            + ("..." if len(src.get("text") or "") > 400 else "")
                        )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                }
            )

# ---------------------------------------------------------------------------
# Footer: show (never the key itself) whether the LLM is configured.
# ---------------------------------------------------------------------------

try:
    configured = get_gemini_client() is not None
except Exception:
    configured = False

if not configured:
    st.sidebar.warning(
        "No GEMINI_API_KEY found. Upload + retrieval still work, but answers "
        "need the key in .env."
    )

import hashlib

import streamlit as st

from document_loader import load_document
from chunker import chunk_document
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tharun's RAG Workspace",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f7f8fa;
    }

    .main .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .workspace-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .workspace-subtitle {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }


    /* ========================================================
       PANEL TITLES
       ======================================================== */

    .panel-title {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .panel-subtitle {
        font-size: 0.76rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }


    /* ========================================================
       STREAMLIT CONTAINERS
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }


    /* ========================================================
       DOCUMENT INFORMATION
       ======================================================== */

    .document-type {
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 700;
        color: #6b7280;
        background-color: #f3f4f6;
        border-radius: 5px;
        padding: 0.18rem 0.4rem;
        margin-bottom: 0.35rem;
    }

    .document-name {
        font-size: 0.8rem;
        font-weight: 650;
        color: #111827;
        line-height: 1.35;
        word-break: break-word;
    }

    .document-meta {
        font-size: 0.7rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }


    /* ========================================================
       SOURCE INFORMATION
       ======================================================== */

    .source-label {
        font-size: 0.63rem;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.04em;
    }

    .source-name {
        font-size: 0.78rem;
        font-weight: 650;
        color: #111827;
        line-height: 1.35;
        word-break: break-word;
    }

    .source-meta {
        font-size: 0.69rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    .source-similarity {
        font-size: 0.7rem;
        font-weight: 650;
        color: #374151;
        margin-top: 0.35rem;
    }

    .source-excerpt {
        font-size: 0.72rem;
        line-height: 1.45;
        color: #4b5563;
        margin-top: 0.55rem;
    }


    /* ========================================================
       QUERY
       ======================================================== */

    .query-label {
        font-size: 0.63rem;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.04em;
    }

    .query-text {
        font-size: 0.78rem;
        color: #111827;
        line-height: 1.4;
        word-break: break-word;
        margin-top: 0.3rem;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        color: #374151;
        font-weight: 600;
        min-height: 38px;
    }

    .stButton > button:hover {
        border-color: #9ca3af;
        color: #111827;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 12px;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    [data-testid="stChatInput"] {
        margin-top: 1rem;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .workspace-title {
            font-size: 1.3rem;
        }

        .workspace-subtitle {
            font-size: 0.78rem;
        }
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# SERVICES
# ============================================================

@st.cache_resource
def get_vector_store():
    return VectorStore()


@st.cache_resource
def get_pipeline():
    return RAGPipeline(get_vector_store())


store = get_vector_store()
pipeline = get_pipeline()


# ============================================================
# HELPERS
# ============================================================

def file_signature(uploaded_file):
    """
    Generate a stable signature for an uploaded file.
    """

    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


def format_page_count(pages):
    """
    Display page/pages correctly.
    """

    try:
        pages = int(pages)
    except (TypeError, ValueError):
        return f"{pages} pages"

    if pages == 1:
        return "1 page"

    return f"{pages} pages"


def render_document_card(doc):
    """
    Render an indexed document using native Streamlit
    components rather than HTML content.
    """

    filename = str(
        doc.get("filename", "")
    )

    doc_type = str(
        doc.get("doc_type", "")
    ).upper()

    chunks = doc.get(
        "chunks",
        0,
    )

    pages = doc.get(
        "pages",
        0,
    )

    with st.container(border=True):

        st.caption(doc_type)

        st.markdown(
            f"**{filename}**"
        )

        st.caption(
            f"{chunks} chunks · "
            f"{format_page_count(pages)}"
        )


def render_source_card(index, source):
    """
    Render one retrieved source using native Streamlit
    components.
    """

    filename = str(
        source.get("filename", "")
    )

    page = source.get(
        "page",
        "—",
    )

    chunk = source.get(
        "chunk_index",
        "—",
    )

    similarity = source.get(
        "similarity",
        0,
    )

    excerpt = str(
        source.get("text", "")
    ).strip()

    if len(excerpt) > 600:
        excerpt = excerpt[:600] + "..."

    with st.container(border=True):

        st.caption(
            f"SOURCE {index:02d}"
        )

        st.markdown(
            f"**{filename}**"
        )

        st.caption(
            f"Page {page} · Chunk {chunk}"
        )

        st.caption(
            f"{similarity}% similarity"
        )

        if excerpt:
            st.write(excerpt)


def index_file(uploaded_file):
    """
    Extract, chunk and store a document.
    """

    signature = file_signature(
        uploaded_file
    )

    if signature in st.session_state.processed_uploads:

        return (
            False,
            "This file has already been "
            "processed in this session.",
        )

    try:

        document = load_document(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )

        chunks = chunk_document(
            document
        )

        if not chunks:

            return (
                False,
                "No text chunks were extracted "
                "from this document.",
            )

        store.add_chunks(
            chunks
        )

        st.session_state.processed_uploads.add(
            signature
        )

        section_count = len(
            document.get(
                "sections",
                [],
            )
        )

        return (
            True,
            f"Indexed {uploaded_file.name} — "
            f"{len(chunks)} chunks from "
            f"{section_count} extracted sections.",
        )

    except Exception as exc:

        return (
            False,
            f"Could not index "
            f"{uploaded_file.name}: {exc}",
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Tharun's RAG Workspace"
)

st.caption(
    "Ask questions using only your uploaded documents"
)


# ============================================================
# THREE-COLUMN WORKSPACE
# ============================================================

documents_col, conversation_col, context_col = st.columns(
    [1.05, 2.25, 1.15],
    gap="large",
)


# ============================================================
# DOCUMENTS
# ============================================================

with documents_col:

    st.markdown(
        '<div class="panel-title">'
        "Documents"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-subtitle">'
        "Your indexed knowledge"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "md",
            "markdown",
        ],
        accept_multiple_files=True,
        help="Maximum 200MB per file.",
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            success, message = index_file(
                uploaded_file
            )

            if success:

                st.success(
                    message
                )

            elif message.startswith(
                "This file has already"
            ):

                st.info(
                    message
                )

            else:

                st.error(
                    message
                )

    st.markdown(
        '<div class="panel-title" '
        'style="margin-top:1.25rem;">'
        "Indexed documents"
        "</div>",
        unsafe_allow_html=True,
    )

    documents = store.list_documents()

    if documents:

        for doc in documents:

            render_document_card(
                doc
            )

            filename = doc.get(
                "filename",
                "",
            )

            if st.button(
                f"Remove {filename}",
                key=f"remove_{filename}",
                use_container_width=True,
            ):

                try:

                    store.delete_document(
                        filename
                    )

                    st.session_state.last_sources = []

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Could not remove "
                        f"{filename}: {exc}"
                    )

        total_chunks = store.count()

        st.caption(
            f"{total_chunks} chunks indexed · "
            f"{len(documents)} documents"
        )

        if st.button(
            "Clear all documents",
            use_container_width=True,
        ):

            try:

                for doc in documents:

                    store.delete_document(
                        doc.get(
                            "filename",
                            "",
                        )
                    )

                st.session_state.last_sources = []

                st.session_state.last_search_query = ""

                st.session_state.processed_uploads = set()

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not clear documents: "
                    f"{exc}"
                )

    else:

        st.info(
            "No documents indexed yet. "
            "Upload a document to begin."
        )


# ============================================================
# CONVERSATION
# ============================================================

with conversation_col:

    st.markdown(
        '<div class="panel-title">'
        "Conversation"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-subtitle">'
        "Grounded answers from your documents"
        "</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:

        st.subheader(
            "Ask your documents anything"
        )

        st.info(
            "Upload one or more documents and ask "
            "a question. Answers are generated "
            "using retrieved evidence from your "
            "indexed knowledge base."
        )

        st.caption(
            "Try a question"
        )

        suggestion_cols = st.columns(2)

        suggestions = [
            "What are the main topics in these documents?",
            "Summarize the most important information.",
            "What products are discussed?",
            "What information is available about this topic?",
        ]

        for index, suggestion in enumerate(
            suggestions
        ):

            with suggestion_cols[
                index % 2
            ]:

                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True,
                ):

                    st.session_state.pending_question = (
                        suggestion
                    )

                    st.rerun()

    else:

        for message in (
            st.session_state.chat_history
        ):

            role = message.get(
                "role",
                "assistant",
            )

            content = message.get(
                "content",
                "",
            )

            with st.chat_message(
                role
            ):

                st.markdown(
                    content
                )


# ============================================================
# CONTEXT
# ============================================================

with context_col:

    st.markdown(
        '<div class="panel-title">'
        "Context"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-subtitle">'
        "Retrieved evidence used for the answer"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.last_search_query:

        with st.container(border=True):

            st.caption(
                "SEARCH QUERY"
            )

            st.write(
                st.session_state.last_search_query
            )

    if st.session_state.last_sources:

        st.markdown(
            '<div class="panel-title" '
            'style="font-size:0.85rem; '
            'margin-top:0.5rem;">'
            "Retrieved sources"
            "</div>",
            unsafe_allow_html=True,
        )

        for index, source in enumerate(
            st.session_state.last_sources,
            start=1,
        ):

            render_source_card(
                index,
                source,
            )

    else:

        st.info(
            "Sources retrieved for your latest "
            "question will appear here."
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# HANDLE SUGGESTION QUESTIONS
# ============================================================

if (
    question is None
    and st.session_state.pending_question
):

    question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        # ----------------------------------------------------
        # Add user question
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # ----------------------------------------------------
        # Check for documents
        # ----------------------------------------------------

        if store.count() == 0:

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "I don't have any indexed "
                        "documents yet. Please "
                        "upload a document first."
                    ),
                }
            )

            st.rerun()

        # ----------------------------------------------------
        # Run RAG pipeline
        # ----------------------------------------------------

        try:

            history = (
                st.session_state.chat_history[:-1]
            )

            result = pipeline.run(
                question,
                history=history,
            )

            answer = result.get(
                "answer",
                "I could not generate an answer.",
            )

            sources = result.get(
                "sources",
                [],
            )

            search_query = result.get(
                "search_query",
                question,
            )

            # ------------------------------------------------
            # Save retrieved context
            # ------------------------------------------------

            st.session_state.last_sources = (
                sources
            )

            st.session_state.last_search_query = (
                search_query
            )

            # ------------------------------------------------
            # Save answer
            # ------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "Something went wrong while "
                        "processing your question.\n\n"
                        f"`{exc}`"
                    ),
                }
            )

        st.rerun()

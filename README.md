# 📚 Generic RAG Chatbot (Streamlit + ChromaDB + OpenAI)

A production-style, **document-agnostic** Retrieval-Augmented Generation
chatbot. Upload **any** PDF / DOCX / TXT / CSV / Markdown file through the UI
and ask questions about it — **no source-code changes, ever**. Nothing about
document content is hard-coded; the pipeline reads whatever you upload.

```
Upload → detect type → extract text (+ page numbers) → chunk → embed
       → store in ChromaDB → user question → embed → semantic search (top-k)
       → build context → LLM (OpenAI API) → answer + sources
```

## 1. Folder structure

```
rag-chatbot/
├── app.py               # Streamlit chatbot UI (upload, chat, sources)
├── document_loader.py   # format detection + text extraction (+ metadata)
├── chunker.py           # recursive splitting into overlapping chunks
├── vector_store.py      # MiniLM embeddings + persistent ChromaDB store
├── rag_pipeline.py      # question → embedding → retrieval → context → LLM
├── requirements.txt
├── .env.example         # copy to .env and add your key
└── chroma_db/           # created automatically (persistent vector store)
```

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

## 3. Run

```bash
streamlit run app.py
```

Then open http://localhost:8501, upload a document in the sidebar, wait for
the green "indexed" banner, and ask questions in the chat box.

## 4. How document ingestion works

`document_loader.detect_document_type()` picks the parser from the extension:

| Format | Parser | Page numbers |
|--------|--------|--------------|
| PDF    | `pypdf` — one section per real page | real page number |
| DOCX   | `python-docx` — paragraphs + tables grouped (~60 per logical page) | logical page |
| TXT/MD | decoded (UTF-8, fallback latin-1), split on form-feed | real/1 |
| CSV    | stdlib `csv` — rows rendered as `column: value`, ~50 rows per logical page | logical page |

Every section carries `{filename, doc_type, page}`. Empty files, corrupted
PDFs, password-protected PDFs and unsupported formats raise descriptive
errors that the UI shows as friendly messages.

## 5. How chunking works

`chunker.split_text()` recursively splits on `\n\n` → `\n` → `. ` → space →
characters until pieces fit `chunk_size` (1000 chars), then merges small
pieces and carries a 150-char overlap tail between consecutive chunks so
sentences cut at a boundary stay retrievable.

Each chunk gets a **deterministic id**:

```
employee_policy.pdf_a1b2c3d4__p3__c0007
└── safe name ──┘ └md5-8┘   └page┘ └chunk no┘
```

Re-uploading the same file **upserts** (overwrites) instead of duplicating.

## 6. How embeddings are generated

`vector_store.MiniLMEmbeddingFunction` wraps the open-source model
`sentence-transformers/all-MiniLM-L6-v2`. The model converts text into
384-dimension vectors where similar meanings sit close together — that is
what makes semantic (not keyword) search possible. The model is loaded **once
per process** and cached; vectors are L2-normalised so cosine similarity is a
dot product.

## 7. How ChromaDB stores vectors

`chromadb.PersistentClient` writes to `./chroma_db` (override with
`CHROMA_PERSIST_DIR`). Per chunk the collection (`cosine` space) stores:

- `documents=` the raw chunk text
- `embeddings=` the MiniLM vector
- `metadatas=` filename, doc_type, page, chunk_index
- `ids=` the deterministic chunk id

Text + metadata are always retrievable — never just vectors.

## 8. How retrieval works

For **every** user question: embed the question with the same model →
`collection.query(query_embeddings=..., n_results=k)` → top-k chunks across
**all** indexed documents (optionally filtered to one file). Each hit reports
cosine distance and similarity % (`(1 - distance) × 100`).

## 9. How context reaches the LLM

`rag_pipeline.build_context()` renders numbered blocks:

```
[1] Source: employee_policy.pdf | page 3 | chunk 7
<chunk text>
```

The current OpenAI SDK is used (`from openai import OpenAI`;
`client.chat.completions.create`). Model is configurable via `OPENAI_MODEL`
(default `gpt-4o-mini`). The system prompt instructs the LLM to answer **only**
from the retrieved context and to say when the answer is not in the documents.

## 10. Conversation memory

Recent history is kept in session state. Before retrieval, a cheap LLM call
rewrites follow-ups into standalone queries:

> "What is the leave policy?" → answer → "How many can I take consecutively?"
> is rewritten to "How many casual leave days can be taken consecutively?"

Retrieval still runs for every question; rewriting can fail safely (falls
back to the original question).

## 11. Security

- API key only in `.env` (never committed; `.env.example` is a template).
- Key is read via `os.getenv`, never hard-coded, never rendered in the UI.
- A missing key degrades gracefully: the UI warns instead of crashing.

## 12. Architecture diagram

```
                       ┌────────────────────────────────────────┐
                       │              Streamlit UI              │
                       │  upload · doc list · status · chat UI  │
                       └───────┬────────────────────┬───────────┘
                               │ files              │ question / answer
                    ┌──────────▼─────────┐  ┌───────▼───────────────────┐
                    │  document_loader   │  │       rag_pipeline        │
                    │ pdf/docx/txt/csv/md│  │ rewrite Q (history)       │
                    │  sections + pages  │  │   ↓                       │
                    └──────────┬─────────┘  │ embed Q ──► top-k search  │
                               │            │   ↓                       │
                    ┌──────────▼─────────┐  │ context + Q ──► OpenAI    │
                    │      chunker       │  └───────┬──────────┬────────┘
                    │ deterministic ids  │          │          │
                    └──────────┬─────────┘   ┌──────▼───┐  ┌───▼────────┐
                               │             │ ChromaDB │  │ OpenAI API │
                    ┌──────────▼─────────┐   │ persist  │  │ (key in    │
                    │  MiniLM embeddings │──►│ ./chroma_db │ .env)   │
                    └────────────────────┘   └──────────┘  └────────────┘
```

## 13. Example questions for testing

After uploading an HR-policy style document:

1. "What is the maximum number of casual leaves I can take?"
2. "What does the document say about work from home?"
3. "How many leave days can I take consecutively?" *(follow-up test)*
4. "Summarise the probation period rules."

Cross-document (upload several files):

5. "According to the documents, what is the difference between supervised and
   unsupervised learning?"
6. "Which document mentions the refund policy, and on which page?"

Grounding test (ask something absent):

7. "What is the company's policy on time travel?" → the bot must say the
   information is not available in the uploaded documents.

## 14. Common errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `Unsupported file format` | extension not in pdf/docx/txt/csv/md | convert the file |
| `No extractable text found` | scanned/image-only PDF | run OCR first (e.g. ocrmypdf) |
| `password-protected` | encrypted PDF | remove the password, re-upload |
| `No OPENAI_API_KEY configured` | `.env` missing/empty | copy `.env.example` → `.env`, add key, restart |
| `openai` import error / 401 | bad key or SDK version | `pip install -U openai`; check the key |
| first answer slow | embedding model downloading | one-time ~90 MB download, cached afterwards |
| `chromadb` telemetry errors | network blocked | harmless; set `anonymized_telemetry=False` (already done) |
| answers ignore documents | index empty / question unrelated | check the sidebar doc list; re-upload |
| duplicate chunks after re-upload | older version of ids | ids are deterministic → re-upload upserts |

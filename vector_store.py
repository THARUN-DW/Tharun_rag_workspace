"""
vector_store.py
===============
Embeddings + persistent vector storage (ChromaDB).

Embedding model
---------------
`sentence-transformers/all-MiniLM-L6-v2` converts text into 384-dimension
numerical vectors.  Texts that mean similar things land CLOSE TOGETHER in that
vector space, so a user question can find answers by comparing vector
distances instead of exact keywords -- that is semantic similarity search.

The model is loaded ONCE per process and cached in a class-level dict, so
embedding thousands of chunks does not reload the neural network each time.

Vector database
---------------
ChromaDB's PersistentClient writes everything to disk (./chroma_db by
default), so the index survives app restarts.  For every chunk we store:

    - the raw chunk text      -> documents=
    - its embedding vector    -> embeddings=  (computed by us, passed explicitly)
    - metadata: filename, doc_type, page, chunk_index -> metadatas=
    - a deterministic chunk id                        -> ids=

Because the ids are deterministic, re-uploading a document UPSERTS rather
than duplicates. The collection uses cosine space; reported similarity is
(1 - cosine_distance) as a percentage.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "rag_documents"
# Stored next to the project files by default; override via CHROMA_PERSIST_DIR.
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


class VectorStoreError(Exception):
    """Vector-store level failure with a user-friendly message."""


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------


class MiniLMEmbeddingFunction:
    """
    Wraps the open-source sentence-transformers model
    `sentence-transformers/all-MiniLM-L6-v2`.

    The embedding model converts text into numerical vectors that allow
    semantic similarity search: the meaning of the text is captured by the
    DIRECTION of the vector, so "maximum casual leaves" and "how many days
    off can I take" end up near each other even with zero keyword overlap.
    """

    _model_cache: Dict[str, Any] = {}  # class-level cache: load the NN once

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name

    def _model(self):
        if self.model_name not in MiniLMEmbeddingFunction._model_cache:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise VectorStoreError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt"
                ) from exc
            MiniLMEmbeddingFunction._model_cache[self.model_name] = SentenceTransformer(
                self.model_name
            )
        return MiniLMEmbeddingFunction._model_cache[self.model_name]

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into normalised (cosine-ready) vectors."""
        if not texts:
            return []
        model = self._model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    # ChromaDB (>= 0.4.16) embedding-function interface. We mostly pass
    # embeddings explicitly, but keeping the interface lets Chroma use this
    # function directly if configured to.
    def __call__(self, input=None, documents=None, **_):  # noqa: A002
        docs = input if input is not None else documents
        return self.encode(list(docs or []))

    def name(self) -> str:
        return f"minilm:{self.model_name}"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """Thin, error-safe wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR") or DEFAULT_PERSIST_DIR
        self.embedding = MiniLMEmbeddingFunction(model_name)

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError(
                "chromadb is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            try:
                from chromadb.config import Settings

                settings = Settings(anonymized_telemetry=False)
                self.client = chromadb.PersistentClient(path=self.persist_dir, settings=settings)
            except Exception:
                self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not initialise the ChromaDB database at '{self.persist_dir}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict]) -> int:
        """Embed + upsert chunks. Returns the number of chunks stored."""
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [dict(c["metadata"]) for c in chunks]
        # Chroma metadata values must be str / int / float / bool (no None).
        for m in metadatas:
            if m.get("page") is None:
                m["page"] = -1

        try:
            vectors = self.embedding.encode(texts)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to compute embeddings: {exc}") from exc

        try:
            self.collection.upsert(
                ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB write failed: {exc}") from exc
        return len(ids)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search(
        self, query: str, k: int = 4, filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and return the top-k most semantically similar chunks
        across ALL indexed documents (or only one document when `filename` is
        given). Each hit carries text, metadata, distance and similarity %.
        """
        total = self.count()
        if total == 0:
            return []

        try:
            query_vector = self.embedding.encode([query])[0]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed the question: {exc}") from exc

        try:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=max(1, min(k, total)),
                where={"filename": filename} if filename else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB search failed: {exc}") from exc

        hits: List[Dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            meta = dict(meta or {})
            page = meta.get("page")
            if page == -1:
                page = None
            hits.append(
                {
                    "id": chunk_id,
                    "text": text or "",
                    "metadata": meta,
                    "filename": meta.get("filename", "unknown"),
                    "doc_type": meta.get("doc_type", ""),
                    "page": page,
                    "chunk_index": meta.get("chunk_index"),
                    "distance": float(dist),
                    # cosine distance 0.0 -> 100 % similarity, 0.3 -> 70 %, ...
                    "similarity": round(max(0.0, 1.0 - float(dist)) * 100, 1),
                }
            )
        return hits

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete_document(self, filename: str) -> int:
        """Remove every chunk belonging to one document. Returns deleted count."""
        try:
            found = self.collection.get(where={"filename": filename}, include=["metadatas"])
            ids = found.get("ids", []) or []
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Could not delete '{filename}': {exc}") from exc

    def list_documents(self) -> List[Dict[str, Any]]:
        """Aggregate stored chunks per document: filename, type, chunk count, pages."""
        if self.count() == 0:
            return []
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Could not read the vector store: {exc}") from exc

        agg: Dict[str, Dict[str, Any]] = {}
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            fname = meta.get("filename", "unknown")
            entry = agg.setdefault(
                fname,
                {"filename": fname, "doc_type": meta.get("doc_type", ""), "chunks": 0, "pages": set()},
            )
            entry["chunks"] += 1
            if meta.get("page") not in (None, -1):
                entry["pages"].add(meta.get("page"))

        documents: List[Dict[str, Any]] = []
        for entry in agg.values():
            pages = entry.pop("pages")
            entry["pages"] = len(pages) if pages else None
            documents.append(entry)
        documents.sort(key=lambda d: d["filename"])
        return documents

    def count(self) -> int:
        """Total number of stored chunks."""
        try:
            return self.collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Could not count chunks in the vector store: {exc}") from exc

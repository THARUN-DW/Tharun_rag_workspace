"""
rag_pipeline.py
===============
Document-agnostic RAG pipeline using Google Gemini.

Flow:
    user question
      -> optional follow-up rewriting
      -> semantic search over indexed documents
      -> retrieved context
      -> Gemini Chat API
      -> grounded answer + retrieved sources
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TOP_K = 4
MAX_HISTORY_MESSAGES = 6
MAX_CHUNK_CHARS_IN_PROMPT = 1200
MAX_HISTORY_CHARS_PER_MESSAGE = 1500

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

RAG_SYSTEM_PROMPT = """You are a document-based AI assistant.

Answer the user's question using only the retrieved context.

Do not invent information that is not present in the retrieved context.

If the answer cannot be found in the retrieved context, clearly state that the
information is not available in the uploaded documents.

When possible, give a concise answer and mention the source document (the
file name the information came from)."""

CONDENSE_SYSTEM_PROMPT = """Rewrite the user's latest message as ONE standalone
search query that can be understood without seeing the chat history.

Rules:
- Resolve pronouns and shorthand using the recent conversation.
- Keep the original language and domain terms.
- Output ONLY the rewritten query, nothing else.
- If the message is already standalone, output it unchanged."""


class RAGPipelineError(Exception):
    """Any failure inside the RAG flow."""


class MissingAPIKeyError(RAGPipelineError):
    """GEMINI_API_KEY is not configured."""


def get_gemini_client():
    """
    Build a Google Gemini client from GEMINI_API_KEY.

    The API key is read from .env/environment variables and is never
    displayed in the UI.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    if not api_key:
        return None

    try:
        from google import genai
    except ImportError as exc:
        raise RAGPipelineError(
            "The 'google-genai' package is not installed. "
            "Run: pip install -U google-genai"
        ) from exc

    return genai.Client(api_key=api_key)


class RAGPipeline:
    """question -> retrieval -> context -> Gemini -> answer."""

    def __init__(
        self,
        vector_store,
        client=None,
        model: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store
        self._client = client
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            DEFAULT_GEMINI_MODEL,
        )
        self.top_k = top_k

    # ---------------------------------------------------------
    # Gemini client
    # ---------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            self._client = get_gemini_client()
        return self._client

    @property
    def llm_available(self) -> bool:
        try:
            return self._ensure_client() is not None
        except RAGPipelineError:
            return False

    # ---------------------------------------------------------
    # Step 1: conversation-memory question rewriting
    # ---------------------------------------------------------

    def rewrite_question(
        self,
        question: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Turn a follow-up question into a standalone search query.

        Gemini Chat API is used here instead of the OpenAI API.
        If rewriting fails, the original question is returned.
        """

        client = self._ensure_client()

        if client is None or not history:
            return question

        recent = history[-MAX_HISTORY_MESSAGES:]

        transcript = "\n".join(
            f"{m.get('role', 'user').upper()}: "
            f"{(m.get('content') or '')[:500]}"
            for m in recent
        )

        prompt = (
            f"Recent conversation:\n\n"
            f"{transcript}\n\n"
            f"Latest user message:\n{question}\n\n"
            f"Rewrite the latest user message as one standalone search query."
        )

        try:
            from google.genai import types

            chat = client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=CONDENSE_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=200,
                ),
            )

            response = chat.send_message(prompt)

            rewritten = (
                (response.text or "")
                .strip()
                .strip('"')
            )

            return rewritten or question

        except Exception:
            return question

    # ---------------------------------------------------------
    # Step 2: retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Embed the query and fetch the top-k chunks."""
        return self.vector_store.search(
            query,
            k=k or self.top_k,
            filename=filename,
        )

    # ---------------------------------------------------------
    # Step 3: context construction
    # ---------------------------------------------------------

    @staticmethod
    def build_context(hits: List[Dict[str, Any]]) -> str:
        """Render retrieved chunks as a numbered context block."""

        blocks = []

        for i, hit in enumerate(hits, start=1):
            page = hit.get("page")
            page_s = (
                f"page {page}"
                if page is not None
                else "page n/a"
            )

            header = (
                f"[{i}] Source: "
                f"{hit.get('filename', 'unknown')} | "
                f"{page_s} | "
                f"chunk {hit.get('chunk_index')}"
            )

            body = (
                hit.get("text") or ""
            )[:MAX_CHUNK_CHARS_IN_PROMPT]

            blocks.append(
                f"{header}\n{body}"
            )

        return "\n\n".join(blocks)

    # ---------------------------------------------------------
    # Step 4: Gemini answer generation
    # ---------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
    ) -> str:

        client = self._ensure_client()

        if client is None:
            raise MissingAPIKeyError(
                "No GEMINI_API_KEY configured. "
                "Add GEMINI_API_KEY to your .env file "
                "and restart the app."
            )

        history_text = ""

        for m in history[-MAX_HISTORY_MESSAGES:]:
            role = m.get("role", "user")
            content = (
                m.get("content") or ""
            )[:MAX_HISTORY_CHARS_PER_MESSAGE]

            if content:
                history_text += (
                    f"{role.upper()}: {content}\n"
                )

        prompt = (
            "Retrieved context from the uploaded documents:\n\n"
            f"{context}\n\n"
            "Recent conversation:\n\n"
            f"{history_text}\n"
            "User question:\n"
            f"{question}\n\n"
            "Answer using ONLY the retrieved context. "
            "Do not use outside knowledge."
        )

        try:
            from google.genai import types

            chat = client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )

            response = chat.send_message(prompt)

            answer = (response.text or "").strip()

            if not answer:
                raise RAGPipelineError(
                    "Gemini returned an empty response."
                )

            return answer

        except RAGPipelineError:
            raise

        except Exception as exc:
            raise RAGPipelineError(
                f"Gemini API call failed: {exc}"
            ) from exc

    # ---------------------------------------------------------
    # Complete RAG flow
    # ---------------------------------------------------------

    def run(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        k: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:

        history = history or []

        if self.vector_store.count() == 0:
            return {
                "answer": (
                    "No documents are indexed yet. "
                    "Upload a document in the sidebar and "
                    "ask again once indexing finishes."
                ),
                "sources": [],
                "search_query": question,
            }

        search_query = (
            self.rewrite_question(question, history)
            if history
            else question
        )

        hits = self.retrieve(
            search_query,
            k=k,
            filename=filename,
        )

        if not hits:
            return {
                "answer": (
                    "I could not find any relevant information "
                    "in the uploaded documents for that question."
                ),
                "sources": [],
                "search_query": search_query,
            }

        context = self.build_context(hits)

        answer = self.generate_answer(
            question,
            context,
            history,
        )

        return {
            "answer": answer,
            "sources": hits,
            "search_query": search_query,
        }

"""
chunker.py
==========
Splits extracted document text into meaningful, overlapping chunks.

Why chunk?  Embedding models and LLMs work best on passages of a few hundred
to ~1500 characters.  Small, coherent chunks make vector search precise,
because a single embedding vector can only "describe" so much content.

This module implements a recursive character splitter (the same idea as
LangChain's RecursiveCharacterTextSplitter, but dependency-free):

    1. Try to split on paragraph breaks ("\\n\\n"), then line breaks ("\\n"),
       then sentence boundaries (". "), then words, then characters -- the
       first separator that produces pieces <= chunk_size wins.
    2. Merge the small pieces back up to chunk_size.
    3. Carry a short tail of the previous chunk forward (chunk_overlap) so a
       sentence split across two chunks is still retrievable.

Every chunk receives a DETERMINISTIC id:

    {safe_filename}_{md5_8}__p{page}__c{chunk_index}

Re-uploading the same document therefore UPSERTS (overwrites) the same ids in
ChromaDB instead of creating duplicate vectors. The md5 suffix prevents two
different files with similar names from ever colliding.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List

DEFAULT_CHUNK_SIZE = 1000    # characters per chunk (≈ 200-250 tokens)
DEFAULT_CHUNK_OVERLAP = 150  # characters carried over between consecutive chunks

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Core recursive splitting
# ---------------------------------------------------------------------------


def _merge_pieces(pieces: List[str], separator: str, chunk_size: int) -> List[str]:
    """Greedily merge small pieces into blocks of at most chunk_size chars."""
    merged: List[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= chunk_size:
            current = current + separator + piece
        else:
            merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Return pieces of text, each (best-effort) <= chunk_size characters."""
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    # Pick the first separator that actually appears in the text.
    separator = separators[-1]  # "" (character level) as the last resort
    next_separators: List[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            next_separators = []
            break
        if sep in text:
            separator = sep
            next_separators = separators[i + 1:]
            break

    if separator == "":
        # Last resort: hard character-level split.
        return [text[i: i + chunk_size].strip() for i in range(0, len(text), chunk_size)]

    pieces: List[str] = []
    small: List[str] = []
    for part in text.split(separator):
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            small.append(part)
        else:
            # An oversized piece: flush what we have, then recurse into it
            # with the finer-grained separators.
            if small:
                pieces.extend(_merge_pieces(small, separator, chunk_size))
                small = []
            pieces.extend(
                _recursive_split(part, chunk_size, next_separators if next_separators else [""])
            )
    if small:
        pieces.extend(_merge_pieces(small, separator, chunk_size))
    return pieces


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split one block of text into overlapping chunks of readable passages."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _recursive_split(text, chunk_size, _SEPARATORS)
    if not pieces:
        return []

    chunks = [pieces[0]]
    for piece in pieces[1:]:
        if chunk_overlap > 0:
            # Glue the tail of the previous chunk onto this one. The tail may
            # start mid-word; that is an accepted trade-off for continuity.
            tail = chunks[-1][-chunk_overlap:]
            glued = f"{tail} {piece}".strip()
            if len(glued) <= chunk_size + chunk_overlap:
                chunks.append(glued)
                continue
        chunks.append(piece)
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Chunk ids
# ---------------------------------------------------------------------------


def _safe_filename(filename: str) -> str:
    base = os.path.basename(filename or "document")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return safe[:80] or "document"


def _name_digest(filename: str) -> str:
    """Short deterministic hash so different files never produce the same ids."""
    return hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    sections: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Turn loader sections into a flat list of chunks:

        {
            "id": "employee_policy.pdf_a1b2c3d4__p2__c0003",
            "text": "...",
            "metadata": {
                "filename": "employee_policy.pdf",
                "doc_type": "pdf",
                "page": 2,            # -1 sentinel when the format has no pages
                "chunk_index": 3,     # 1-based, per page
            },
        }
    """
    chunks: List[Dict] = []
    for section in sections:
        meta = section.get("metadata", {})
        filename = meta.get("filename", "unknown")
        page = meta.get("page")
        page_label = f"p{page}" if page is not None else "np"

        text_chunks = split_text(section.get("text", ""), chunk_size, chunk_overlap)
        for local_index, chunk_text in enumerate(text_chunks, start=1):
            chunk_id = (
                f"{_safe_filename(filename)}_{_name_digest(filename)}"
                f"__{page_label}__c{local_index:04d}"
            )
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "doc_type": meta.get("doc_type", "unknown"),
                        "page": page if page is not None else -1,
                        "chunk_index": local_index,
                    },
                }
            )
    return chunks

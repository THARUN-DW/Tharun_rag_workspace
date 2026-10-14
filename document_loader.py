"""
document_loader.py
==================
Generic document-loading layer for the RAG chatbot.

Responsibilities
----------------
1. Detect the document type from the file extension (PDF / DOCX / TXT / CSV / MD).
2. Extract raw text from the uploaded bytes. NOTHING about document content is
   hard-coded -- every file is read at runtime, so the same pipeline works for
   company policies, research papers, notes, manuals, books, etc.
3. Preserve useful metadata per section:
       filename  - original upload name
       doc_type  - pdf | docx | txt | csv | md
       page      - real page number (PDF) or logical page number (DOCX paragraph
                   groups / CSV row groups); -1 sentinel is never used here,
                   pages are real integers or None.

Public entry point:  load_document(filename, data)  ->  list of sections:

    [
        {
            "text": "raw text of this page/section ...",
            "metadata": {"filename": "employee_policy.pdf",
                         "doc_type": "pdf",
                         "page": 3}
        },
        ...
    ]

Every loader is defensive: corrupted files, empty files and unsupported
formats raise descriptive exceptions that the UI turns into friendly error
messages instead of crashes.
"""

from __future__ import annotations

import csv
import io
import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Exceptions (the UI catches these and shows friendly messages)
# ---------------------------------------------------------------------------


class DocumentLoadError(Exception):
    """Base class for all document-loading errors."""


class UnsupportedFormatError(DocumentLoadError):
    """Raised for file types this app cannot parse."""


class EmptyDocumentError(DocumentLoadError):
    """Raised when a document contains no extractable text."""


class CorruptedDocumentError(DocumentLoadError):
    """Raised when a file cannot be parsed (e.g. damaged or password-protected PDF)."""


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".csv": "csv",
    ".md": "md",
    ".markdown": "md",
}


def detect_document_type(filename: str) -> str:
    """Return the canonical document type for a filename, or raise."""
    ext = os.path.splitext(filename or "")[1].lower()
    doc_type = SUPPORTED_EXTENSIONS.get(ext)
    if doc_type is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file format: '{filename}'. Supported formats: {supported}"
        )
    return doc_type


def is_supported(filename: str) -> bool:
    """True when the file extension is one we can parse."""
    return os.path.splitext(filename or "")[1].lower() in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Individual loaders (each returns List[section_dict])
# ---------------------------------------------------------------------------


def _load_pdf(filename: str, data: bytes) -> List[Dict]:
    """PDF -> one section per real page (page numbers preserved)."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError(
            "pypdf is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Try an empty password; if that fails, the file is unreadable for us.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise CorruptedDocumentError(
                    f"'{filename}' is password-protected and cannot be read."
                ) from exc
        pages = list(reader.pages)
    except PdfReadError as exc:
        raise CorruptedDocumentError(
            f"'{filename}' looks like a corrupted PDF ({exc})."
        ) from exc
    except CorruptedDocumentError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(
            f"'{filename}' could not be parsed as a PDF ({exc})."
        ) from exc

    sections: List[Dict] = []
    for page_no, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""  # a single broken page must not kill the whole document
        if text.strip():
            sections.append(
                {
                    "text": text,
                    "metadata": {"filename": filename, "doc_type": "pdf", "page": page_no},
                }
            )
    return sections


# .docx is flowing text without real pages: group N paragraphs/rows into one
# logical "page" so source references stay reasonably granular.
DOCX_PARAGRAPHS_PER_LOGICAL_PAGE = 60


def _load_docx(filename: str, data: bytes) -> List[Dict]:
    """DOCX -> paragraphs + table rows grouped into logical pages."""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError(
            "python-docx is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise CorruptedDocumentError(
            f"'{filename}' could not be parsed as a DOCX file ({exc})."
        ) from exc

    blocks: List[str] = []
    for para in document.paragraphs:
        text = (para.text or "").strip()
        if text:
            blocks.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [(cell.text or "").strip() for cell in row.cells]
            cells = [c for c in cells if c]
            if cells:
                blocks.append(" | ".join(cells))

    sections: List[Dict] = []
    for idx in range(0, len(blocks), DOCX_PARAGRAPHS_PER_LOGICAL_PAGE):
        group = blocks[idx: idx + DOCX_PARAGRAPHS_PER_LOGICAL_PAGE]
        sections.append(
            {
                "text": "\n\n".join(group),
                "metadata": {
                    "filename": filename,
                    "doc_type": "docx",
                    "page": idx // DOCX_PARAGRAPHS_PER_LOGICAL_PAGE + 1,
                },
            }
        )
    return sections


def _load_txt(filename: str, data: bytes) -> List[Dict]:
    """TXT / Markdown -> one section per form-feed page break (or a single page)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")

    pages = [p for p in text.split("\f") if p.strip()]
    if not pages:
        return []

    doc_type = "md" if filename.lower().endswith((".md", ".markdown")) else "txt"
    sections: List[Dict] = []
    for page_no, page_text in enumerate(pages, start=1):
        sections.append(
            {
                "text": page_text,
                "metadata": {"filename": filename, "doc_type": doc_type, "page": page_no},
            }
        )
    return sections


CSV_ROWS_PER_LOGICAL_PAGE = 50


def _load_csv(filename: str, data: bytes) -> List[Dict]:
    """CSV -> human-readable 'column: value' rows, grouped into logical pages."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")

    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise CorruptedDocumentError(f"'{filename}' is not a valid CSV file ({exc}).") from exc

    if not rows:
        return []

    header = [h.strip() for h in rows[0]]
    lines: List[str] = []
    for row in rows[1:]:
        if not any((cell or "").strip() for cell in row):
            continue
        fields = []
        for i, cell in enumerate(row):
            col = header[i] if i < len(header) and header[i] else f"column_{i + 1}"
            if (cell or "").strip():
                fields.append(f"{col}: {cell.strip()}")
        lines.append("; ".join(fields))

    sections: List[Dict] = []
    for idx in range(0, len(lines), CSV_ROWS_PER_LOGICAL_PAGE):
        group = lines[idx: idx + CSV_ROWS_PER_LOGICAL_PAGE]
        sections.append(
            {
                "text": "\n".join(group),
                "metadata": {
                    "filename": filename,
                    "doc_type": "csv",
                    "page": idx // CSV_ROWS_PER_LOGICAL_PAGE + 1,
                },
            }
        )
    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "txt": _load_txt,
    "md": _load_txt,
    "csv": _load_csv,
}


def load_document(filename: str, data: bytes) -> List[Dict]:
    """
    Detect the document type, extract its text and return sections with metadata.

    Raises UnsupportedFormatError / EmptyDocumentError / CorruptedDocumentError
    (all subclasses of DocumentLoadError) with user-presentable messages.
    """
    if not data:
        raise EmptyDocumentError(f"'{filename}' is empty (0 bytes).")

    doc_type = detect_document_type(filename)
    sections = _LOADERS[doc_type](filename, data)

    if not sections or all(not s["text"].strip() for s in sections):
        raise EmptyDocumentError(
            f"No extractable text found in '{filename}'. It may be empty, "
            "image-only (a scanned PDF needs OCR), or corrupted."
        )
    return sections

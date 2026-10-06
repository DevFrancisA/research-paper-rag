"""PDF loading and page-aware chunking."""

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SentenceSplitter

from .models import Chunk

_CHUNK_NS = uuid.UUID("6f1d3c52-6a8e-4b8f-9d2f-3e0b6c1a7f10")


def file_doc_id(path: str | Path) -> str:
    """Content hash, so the same file always maps to the same document."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]


def load_pdf_pages(path: str | Path) -> list[tuple[int, str]]:
    """Returns (1-based page number, text) for each page that has text."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[tuple[int, str]], doc_id: str, source: str, chunk_size: int, chunk_overlap: int,
                metadata: dict[str, Any] | None = None) -> list[Chunk]:
    """Splits each page on sentence boundaries; chunks never span pages, so citations stay exact."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 20:  # page numbers, stray headers
                continue
            index = len(chunks)
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid5(_CHUNK_NS, f"{doc_id}:{index}")),
                doc_id=doc_id, source=source, page=page, chunk_index=index, text=piece,
                metadata=dict(metadata or {}),
            ))
    return chunks

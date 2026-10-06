"""Data models shared by ingestion, retrieval, generation and the API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalMode(StrEnum):
    DENSE = "dense"  # vector search only (baseline)
    SPARSE = "sparse"  # BM25 only
    HYBRID = "hybrid"  # dense + BM25 fused with reciprocal rank fusion
    HYBRID_RERANK = "hybrid_rerank"  # hybrid candidates reordered by a cross-encoder


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int | None = None
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def embedding_text(self) -> str:
        """Text sent to the embedders: a short header gives each chunk its document context."""
        location = f"{self.source}, page {self.page}" if self.page is not None else self.source
        return f"[{location}]\n{self.text}"


class RetrievedChunk(Chunk):
    score: float  # fused / dense / sparse score from Qdrant
    rerank_score: float | None = None  # cross-encoder relevance probability (0..1)


class MetadataFilter(BaseModel):
    sources: list[str] | None = None
    doc_ids: list[str] | None = None
    page_gte: int | None = None
    page_lte: int | None = None
    # Exact-match filters on metadata supplied at ingest time, e.g. {"category": "finance"}.
    metadata: dict[str, str | int | bool] | None = None

    def is_empty(self) -> bool:
        return not any(v not in (None, [], {}) for v in self.model_dump().values())


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    cache_hit: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    number: int
    chunk_id: str
    source: str
    page: int | None
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    sources: list[RetrievedChunk]
    refused: bool = False
    refusal_reason: str | None = None
    cache_hit: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


class IngestResult(BaseModel):
    doc_id: str
    source: str
    num_pages: int
    num_chunks: int
    embeddings_cached: int
    embeddings_computed: int
    skipped: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    num_pages: int
    num_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)

"""Runtime settings, read from environment variables (and .env)."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_optional_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    return None if raw.strip().lower() in ("", "none") else float(raw)


@dataclass
class Settings:
    # Models
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-large"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 3072))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gpt-4.1-mini"))
    # Set CHAT_TEMPERATURE=none for models that reject the parameter.
    chat_temperature: float | None = field(default_factory=lambda: _env_optional_float("CHAT_TEMPERATURE", 0.0))
    sparse_model: str = field(default_factory=lambda: _env("SPARSE_MODEL", "Qdrant/bm25"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    # Infrastructure. Empty QDRANT_URL -> embedded on-disk Qdrant; empty DATABASE_URL -> in-memory store.
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", "qdrant_local"))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "docs"))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))
    upload_dir: str = field(default_factory=lambda: _env("UPLOAD_DIR", "uploads"))

    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 96))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 40))
    rerank_k: int = field(default_factory=lambda: _env_int("RERANK_K", 30))
    # Minimum reranker probability for a chunk to count as relevant context.
    relevance_threshold: float = field(default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.05))
    # Replace answers that cite no valid source with a refusal.
    require_citations: bool = field(default_factory=lambda: _env("REQUIRE_CITATIONS", "true").lower() == "true")

    # Batching / caching
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 128))
    retrieval_cache_ttl_s: int = field(default_factory=lambda: _env_int("RETRIEVAL_CACHE_TTL_S", 3600))
    memory_cache_size: int = field(default_factory=lambda: _env_int("MEMORY_CACHE_SIZE", 10_000))


settings = Settings()

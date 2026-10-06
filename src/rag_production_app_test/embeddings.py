"""Dense (OpenAI) and sparse (BM25) encoders.

Dense embeddings are batched and cached by content hash, first in-process (LRU) and then in the
Database cache table, so re-ingesting a document or repeating a query never re-embeds the same text.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Protocol

import numpy as np
from qdrant_client import models as qm

from .db import Database

EMBEDDING_NS = "embedding"


class DenseEmbedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, dim: int, client=None):
        from openai import OpenAI

        self.model = model
        self.dim = dim
        self._client = client or OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Only text-embedding-3-* models accept `dimensions`; older models reject it.
        kwargs = {"dimensions": self.dim} if self.model.startswith("text-embedding-3") else {}
        response = self._client.embeddings.create(model=self.model, input=texts, **kwargs)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


class CachedEmbedder:
    """Wraps a DenseEmbedder with batching and a memory + database cache."""

    def __init__(self, inner: DenseEmbedder, db: Database, batch_size: int = 128, memory_size: int = 10_000):
        self.inner = inner
        self.model = inner.model
        self.dim = inner.dim
        self._db = db
        self._batch_size = batch_size
        self._memory: OrderedDict[str, list[float]] = OrderedDict()
        self._memory_size = memory_size
        self._lock = threading.Lock()
        self.stats = {"memory_hits": 0, "db_hits": 0, "computed": 0, "api_calls": 0}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{self.dim}:{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors, _ = self.embed_with_stats(texts)
        return vectors

    def embed_with_stats(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Returns (vectors, number served from cache)."""
        keys = [self._key(t) for t in texts]
        found: dict[str, list[float]] = {}

        with self._lock:
            for k in keys:
                if k in self._memory:
                    self._memory.move_to_end(k)
                    found[k] = self._memory[k]
        self.stats["memory_hits"] += len(found)

        missing = [k for k in dict.fromkeys(keys) if k not in found]
        if missing:
            from_db = self._db.cache_get_many(EMBEDDING_NS, missing)
            for k, blob in from_db.items():
                found[k] = np.frombuffer(blob, dtype=np.float32).tolist()
            self.stats["db_hits"] += len(from_db)

        # Deduplicate, then embed what's left in batches.
        to_compute = {k: t for k, t in zip(keys, texts) if k not in found}
        items = list(to_compute.items())
        for start in range(0, len(items), self._batch_size):
            batch = items[start:start + self._batch_size]
            vectors = self.inner.embed([t for _, t in batch])
            self.stats["api_calls"] += 1
            self.stats["computed"] += len(batch)
            self._db.cache_set_many(
                EMBEDDING_NS, {k: np.asarray(v, dtype=np.float32).tobytes() for (k, _), v in zip(batch, vectors)}
            )
            for (k, _), v in zip(batch, vectors):
                found[k] = v

        with self._lock:
            for k in keys:
                self._memory[k] = found[k]
                self._memory.move_to_end(k)
            while len(self._memory) > self._memory_size:
                self._memory.popitem(last=False)

        return [found[k] for k in keys], len(keys) - len(to_compute)

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()


class SparseEncoder:
    """BM25 term weights via fastembed; Qdrant applies IDF at query time (Modifier.IDF)."""

    def __init__(self, model: str = "Qdrant/bm25"):
        self._model_name = model
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed import SparseTextEmbedding

                self._model = SparseTextEmbedding(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[qm.SparseVector]:
        return [qm.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in self._get().embed(texts)]

    def embed_query(self, text: str) -> qm.SparseVector:
        e = next(iter(self._get().query_embed(text)))
        return qm.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

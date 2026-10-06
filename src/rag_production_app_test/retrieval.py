"""Retrieval: dense, BM25, hybrid (RRF) and hybrid + cross-encoder rerank, with result caching."""

import hashlib
import json
import time

from .db import Database
from .embeddings import DenseEmbedder, SparseEncoder
from .models import MetadataFilter, RetrievalMode, RetrievalResult, RetrievedChunk
from .reranker import Reranker, rerank
from .vector_store import VectorStore

RETRIEVAL_NS = "retrieval"


class Retriever:
    def __init__(self, store: VectorStore, embedder: DenseEmbedder, sparse: SparseEncoder, reranker: Reranker | None,
                 db: Database, candidate_k: int = 40, rerank_k: int = 30, cache_ttl_s: int = 3600,
                 reranker_name: str = ""):
        self.store = store
        self.embedder = embedder
        self.sparse = sparse
        self.reranker = reranker
        self.db = db
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.cache_ttl_s = cache_ttl_s
        self.reranker_name = reranker_name
        self.cache_enabled = True

    def retrieve(self, query: str, top_k: int = 5, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK,
                 flt: MetadataFilter | None = None) -> RetrievalResult:
        """Returns up to top_k chunks, best first. Relevance thresholding is left to the caller."""
        start = time.perf_counter()
        query = " ".join(query.split())
        if mode == RetrievalMode.HYBRID_RERANK and self.reranker is None:
            mode = RetrievalMode.HYBRID

        key = self._cache_key(query, top_k, mode, flt)
        if self.cache_enabled:
            cached = self.db.cache_get_many(RETRIEVAL_NS, [key]).get(key)
            if cached is not None:
                chunks = [RetrievedChunk.model_validate(c) for c in json.loads(cached)]
                return RetrievalResult(chunks=chunks, cache_hit=True,
                                       timings_ms={"retrieval_total": _ms(start)})

        timings: dict[str, float] = {}
        chunks = self._search(query, top_k, mode, flt, timings)
        timings["retrieval_total"] = _ms(start)

        if self.cache_enabled:
            payload = json.dumps([c.model_dump() for c in chunks]).encode()
            self.db.cache_set_many(RETRIEVAL_NS, {key: payload}, ttl_s=self.cache_ttl_s)
        return RetrievalResult(chunks=chunks, cache_hit=False, timings_ms=timings)

    def _search(self, query: str, top_k: int, mode: RetrievalMode, flt: MetadataFilter | None,
                timings: dict[str, float]) -> list[RetrievedChunk]:
        dense = sparse = None
        if mode != RetrievalMode.SPARSE:
            t = time.perf_counter()
            dense = self.embedder.embed([query])[0]
            timings["embed_query"] = _ms(t)
        if mode != RetrievalMode.DENSE:
            t = time.perf_counter()
            sparse = self.sparse.embed_query(query)
            timings["sparse_query"] = _ms(t)

        t = time.perf_counter()
        if mode == RetrievalMode.DENSE:
            chunks = self.store.search_dense(dense, top_k, flt)
        elif mode == RetrievalMode.SPARSE:
            chunks = self.store.search_sparse(sparse, top_k, flt)
        else:
            limit = self.rerank_k if mode == RetrievalMode.HYBRID_RERANK else top_k
            chunks = self.store.search_hybrid(dense, sparse, limit=max(limit, top_k),
                                              prefetch_limit=self.candidate_k, flt=flt)
        timings["vector_search"] = _ms(t)

        if mode == RetrievalMode.HYBRID_RERANK and chunks:
            t = time.perf_counter()
            chunks = rerank(self.reranker, query, chunks)
            timings["rerank"] = _ms(t)
        return chunks[:top_k]

    def _cache_key(self, query: str, top_k: int, mode: RetrievalMode, flt: MetadataFilter | None) -> str:
        # The corpus version changes on every ingest/delete, which invalidates all cached results.
        parts = {
            "q": query.lower(), "k": top_k, "mode": mode.value,
            "filter": flt.model_dump(exclude_none=True) if flt else None,
            "corpus": self.db.corpus_version(), "emb": self.embedder.model,
            "rerank": self.reranker_name, "cand": self.candidate_k, "rk": self.rerank_k,
        }
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def apply_relevance_threshold(chunks: list[RetrievedChunk], threshold: float) -> list[RetrievedChunk]:
    """Drops chunks the cross-encoder judged irrelevant. Chunks without a rerank score are kept."""
    return [c for c in chunks if c.rerank_score is None or c.rerank_score >= threshold]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)

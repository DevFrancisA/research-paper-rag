"""End-to-end latency with and without caching, and embedding throughput with and without batching."""

import random
import time

from ..pipeline import RAGService
from ..retrieval import RETRIEVAL_NS
from .benchmark import BenchmarkItem
from .metrics import latency_summary, relative_change


def evaluate_latency(service: RAGService, items: list[BenchmarkItem], n_requests: int = 300,
                     seed: int = 11, n_embed_texts: int = 64) -> dict:
    questions = [i.question for i in items]
    # Requests are drawn with replacement, so popular questions repeat as in real traffic.
    workload = random.Random(seed).choices(questions, k=n_requests)
    retriever = service.retriever

    service.query(questions[0], log=False)  # load BM25 + reranker models before timing

    # Baseline: no retrieval cache and no query-embedding cache.
    retriever.cache_enabled = False
    retriever.embedder = service.embedder.inner
    try:
        baseline = _run(service, workload)
    finally:
        retriever.cache_enabled = True
        retriever.embedder = service.embedder

    # Cached: start cold (empty caches) and let the workload warm them.
    service.db.cache_clear(RETRIEVAL_NS)
    service.embedder.clear_memory()
    cached = _run(service, workload)

    result = {
        "n_requests": n_requests, "unique_questions": len(set(workload)),
        "baseline": baseline["summary"], "cached": cached["summary"],
        "stage_p50_ms": {"baseline": baseline["stages"], "cached": cached["stages"]},
        "cache_hit_rate": cached["cache_hit_rate"],
        "p50_change": round(relative_change(baseline["summary"]["p50_ms"], cached["summary"]["p50_ms"]), 4),
    }
    result["embedding_batching"] = _embedding_batching(service, n_embed_texts)
    return result


def _run(service: RAGService, workload: list[str]) -> dict:
    totals, hits = [], 0
    stages: dict[str, list[float]] = {}
    for n, question in enumerate(workload, start=1):
        start = time.perf_counter()
        answer = service.query(question, log=False)
        totals.append((time.perf_counter() - start) * 1000)
        hits += answer.cache_hit
        for stage, ms in answer.timings_ms.items():
            stages.setdefault(stage, []).append(ms)
        if n % 50 == 0:
            print(f"[latency] {n}/{len(workload)}", flush=True)
    return {"summary": latency_summary(totals), "cache_hit_rate": round(hits / len(workload), 4),
            "stages": {s: latency_summary(v)["p50_ms"] for s, v in stages.items()}}


def _embedding_batching(service: RAGService, n_texts: int) -> dict:
    chunks = service.store.scroll_chunks(limit=n_texts)
    texts = [c.embedding_text() for c in chunks]
    if not texts:
        return {}
    inner = service.embedder.inner

    t = time.perf_counter()
    for text in texts:
        inner.embed([text])
    one_by_one = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    for start in range(0, len(texts), service.settings.embedding_batch_size):
        inner.embed(texts[start:start + service.settings.embedding_batch_size])
    batched = (time.perf_counter() - t) * 1000

    # Re-embedding already-seen text (e.g. re-ingesting a document) is served from the cache.
    service.embedder.embed(texts)
    service.embedder.clear_memory()
    t = time.perf_counter()
    service.embedder.embed(texts)
    from_cache = (time.perf_counter() - t) * 1000

    return {"n_texts": len(texts), "unbatched_ms": round(one_by_one, 1), "batched_ms": round(batched, 1),
            "db_cache_ms": round(from_cache, 1),
            "batched_speedup": round(one_by_one / batched, 2) if batched else None}

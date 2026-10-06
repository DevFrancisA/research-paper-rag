"""Compares retrieval modes on the answerable benchmark items."""

from ..models import RetrievalMode
from ..pipeline import RAGService
from .benchmark import BenchmarkItem
from .metrics import hit_at_k, mean, ndcg_at_k, recall_at_k, reciprocal_rank

KS = (1, 3, 5, 10)


def evaluate_retrieval(service: RAGService, items: list[BenchmarkItem],
                       modes: list[RetrievalMode] | None = None) -> dict:
    modes = modes or list(RetrievalMode)
    answerable = [i for i in items if i.answerable]
    retriever = service.retriever
    retriever.cache_enabled = False  # measure retrieval itself, not the cache
    results: dict[str, dict] = {}
    try:
        for mode in modes:
            rows = []
            for n, item in enumerate(answerable, start=1):
                chunks = retriever.retrieve(item.question, top_k=max(KS), mode=mode).chunks
                ids = [c.chunk_id for c in chunks]
                pages = [f"{c.doc_id}:{c.page}" for c in chunks]
                gold = set(item.gold_chunk_ids)
                row = {"style": item.style}
                for k in KS:
                    row[f"recall@{k}"] = recall_at_k(ids, gold, k)
                    row[f"hit@{k}"] = hit_at_k(ids, gold, k)
                row["mrr@10"] = reciprocal_rank(ids, gold, 10)
                row["ndcg@10"] = ndcg_at_k(ids, gold, 10)
                row["page_hit@5"] = hit_at_k(pages, {f"{item.gold_doc_id}:{item.gold_page}"}, 5)
                rows.append(row)
                if n % 25 == 0:
                    print(f"[retrieval] {mode.value}: {n}/{len(answerable)}", flush=True)
            results[mode.value] = {"overall": _aggregate(rows),
                                   "by_style": {s: _aggregate([r for r in rows if r["style"] == s])
                                                for s in sorted({r["style"] for r in rows})}}
    finally:
        retriever.cache_enabled = True
    return {"n_queries": len(answerable), "modes": results}


def _aggregate(rows: list[dict]) -> dict:
    metrics = [k for k in rows[0] if k != "style"] if rows else []
    return {"n": len(rows), **{m: round(mean([r[m] for r in rows]), 4) for m in metrics}}

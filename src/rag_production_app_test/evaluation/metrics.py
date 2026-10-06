"""Ranking and latency metrics."""

import math
import statistics


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def latency_summary(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "p50_ms": round(statistics.median(values), 1) if values else 0.0,
            "p95_ms": round(percentile(values, 0.95), 1), "mean_ms": round(mean(values), 1)}


def relative_change(before: float, after: float) -> float:
    """(after - before) / before, e.g. -0.38 for a 38% reduction."""
    return (after - before) / before if before else 0.0

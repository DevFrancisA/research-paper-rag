"""Cross-encoder reranking (fastembed ONNX models, CPU-friendly)."""

import math
import threading
from typing import Protocol

from .models import RetrievedChunk


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-base", batch_size: int = 16):
        self._model_name = model
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(model_name=self._model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance probabilities in [0, 1] (sigmoid of the cross-encoder logit)."""
        if not texts:
            return []
        logits = self._get().rerank(query, texts, batch_size=self._batch_size)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def rerank(reranker: Reranker, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    scores = reranker.score(query, [c.embedding_text() for c in chunks])
    for chunk, s in zip(chunks, scores, strict=True):
        chunk.rerank_score = s
    return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

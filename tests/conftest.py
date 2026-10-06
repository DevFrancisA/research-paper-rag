import hashlib
import math
import re

import pytest

from rag_production_app_test.config import Settings
from rag_production_app_test.db import MemoryDatabase
from rag_production_app_test.embeddings import CachedEmbedder, SparseEncoder
from rag_production_app_test.models import Chunk
from rag_production_app_test.pipeline import RAGService
from rag_production_app_test.vector_store import VectorStore, make_client

DIM = 64


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, offline, roughly similarity-preserving."""

    model = "fake-embedding"
    dim = DIM

    def __init__(self):
        self.calls: list[int] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        out = []
        for text in texts:
            v = [0.0] * DIM
            for w in _words(text):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class FakeReranker:
    """Scores by query-word overlap, so unrelated chunks fall below the threshold."""

    def score(self, query, texts):
        q = set(_words(query))
        return [len(q & set(_words(t))) / max(len(q), 1) for t in texts]


class FakeLLM:
    def __init__(self, reply="Paris is the capital of France [1]."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


DOCS = {
    "geo.pdf": [
        (1, "Paris is the capital of France and its largest city, located on the Seine river."),
        (2, "Berlin is the capital of Germany. The city is known for its history and museums."),
    ],
    "cooking.pdf": [
        (1, "To bake sourdough bread you need a starter, flour, water and salt, and a long fermentation."),
        (3, "Risotto is made by slowly adding hot stock to arborio rice while stirring constantly."),
    ],
}


def make_chunks(source: str, pages: list[tuple[int, str]], category: str) -> list[Chunk]:
    doc_id = f"doc-{source}"
    return [Chunk(chunk_id=hashlib.md5(f"{doc_id}{i}".encode()).hexdigest(), doc_id=doc_id, source=source,
                  page=page, chunk_index=i, text=text, metadata={"category": category})
            for i, (page, text) in enumerate(pages)]


@pytest.fixture(scope="session")
def sparse():
    return SparseEncoder("Qdrant/bm25")


@pytest.fixture
def service(sparse):
    settings = Settings(embedding_model="fake-embedding", embedding_dim=DIM, database_url="",
                        qdrant_url=":memory:", relevance_threshold=0.3, top_k=3, candidate_k=10, rerank_k=10)
    db = MemoryDatabase()
    embedder = CachedEmbedder(FakeEmbedder(), db, batch_size=2)
    store = VectorStore(make_client(":memory:", ""), "test", DIM)
    svc = RAGService(settings, db, embedder, sparse, store, FakeReranker(), FakeLLM())
    for source, pages in DOCS.items():
        chunks = make_chunks(source, pages, "geography" if source == "geo.pdf" else "food")
        svc.ingest_chunks(chunks, chunks[0].doc_id, source, num_pages=len(pages),
                          metadata={"category": chunks[0].metadata["category"]})
    return svc

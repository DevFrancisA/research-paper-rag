import pytest

from rag_production_app_test.chunking import chunk_pages, clean_text
from rag_production_app_test.evaluation.benchmark import sample_chunks
from rag_production_app_test.evaluation.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rag_production_app_test.generation import REFUSAL, extract_citations, is_refusal
from rag_production_app_test.models import Chunk, RetrievedChunk


def _retrieved(n: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=f"c{i}", doc_id="d", source="a.pdf", page=i, chunk_index=i,
                           text=f"text {i}", score=1.0) for i in range(1, n + 1)]


def test_ranking_metrics():
    ranked = ["x", "gold", "y"]
    assert hit_at_k(ranked, {"gold"}, 1) == 0 and hit_at_k(ranked, {"gold"}, 2) == 1
    assert recall_at_k(ranked, {"gold", "other"}, 3) == 0.5
    assert reciprocal_rank(ranked, {"gold"}, 10) == 0.5
    assert ndcg_at_k(["gold"], {"gold"}, 10) == 1.0
    assert ndcg_at_k(ranked, {"gold"}, 10) == pytest.approx(1 / 1.58496, rel=1e-4)


def test_extract_citations_handles_lists_and_bad_numbers():
    text, cites = extract_citations("A [2, 1]. B [7]. C [2].", _retrieved(2))
    assert text == "A [2][1]. B. C [2]."
    assert [c.number for c in cites] == [2, 1]
    assert cites[0].page == 2


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal(f'"{REFUSAL}"')
    assert not is_refusal("Paris [1].")


def test_chunks_keep_pages_and_stable_ids():
    pages = [(1, "First page sentence. " * 50), (2, "Second page sentence. " * 50)]
    a = chunk_pages(pages, "doc", "f.pdf", chunk_size=128, chunk_overlap=16)
    b = chunk_pages(pages, "doc", "f.pdf", chunk_size=128, chunk_overlap=16)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert {c.page for c in a} == {1, 2}
    assert all(("First" in c.text) == (c.page == 1) for c in a)  # chunks never span pages


def test_clean_text_rejoins_hyphenation():
    assert clean_text("retrie-\nval   works\n\n\n\nok") == "retrieval works\n\nok"


def test_sample_chunks_spreads_across_documents():
    chunks = [Chunk(chunk_id=f"{d}{i}", doc_id=d, source=d, chunk_index=i, text="x" * 400)
              for d, n in (("big", 50), ("small", 3)) for i in range(n)]
    picked = sample_chunks(chunks, 6, seed=1)
    assert sum(c.doc_id == "small" for c in picked) == 3

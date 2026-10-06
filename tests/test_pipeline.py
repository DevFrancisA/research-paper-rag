from rag_production_app_test.generation import REFUSAL
from rag_production_app_test.models import MetadataFilter, RetrievalMode

from .conftest import FakeLLM


def test_lexical_signal_fixes_dense_ranking(service):
    q = "What is the capital of France?"
    retrieve = service.retriever.retrieve
    dense = [(c.source, c.page) for c in retrieve(q, top_k=3, mode=RetrievalMode.DENSE).chunks]
    assert ("geo.pdf", 1) in dense  # the fake dense embedder over-weights "is the capital of"
    # Dense and BM25 disagree on Paris vs Berlin, so RRF ties them; the cross-encoder settles it.
    hybrid = [(c.source, c.page) for c in retrieve(q, top_k=2, mode=RetrievalMode.HYBRID).chunks]
    assert ("geo.pdf", 1) in hybrid
    for mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID_RERANK):
        top = retrieve(q, top_k=3, mode=mode).chunks[0]
        assert (top.source, top.page) == ("geo.pdf", 1), mode


def test_bm25_matches_exact_rare_terms(service):
    chunks = service.retriever.retrieve("arborio", top_k=1, mode=RetrievalMode.SPARSE).chunks
    assert "Risotto" in chunks[0].text


def test_metadata_filters(service):
    retrieve = service.retriever.retrieve
    q = "capital city"
    only_food = retrieve(q, mode=RetrievalMode.HYBRID, flt=MetadataFilter(sources=["cooking.pdf"])).chunks
    assert only_food and {c.source for c in only_food} == {"cooking.pdf"}

    by_page = retrieve(q, mode=RetrievalMode.HYBRID, flt=MetadataFilter(page_gte=2)).chunks
    assert by_page and all(c.page >= 2 for c in by_page)

    by_meta = retrieve(q, mode=RetrievalMode.HYBRID, flt=MetadataFilter(metadata={"category": "food"})).chunks
    assert by_meta and all(c.metadata["category"] == "food" for c in by_meta)


def test_grounded_answer_has_validated_citations(service):
    answer = service.query("What is the capital of France?")
    assert not answer.refused
    assert answer.citations[0].source == "geo.pdf" and answer.citations[0].page == 1
    system, user = service.llm.calls[-1]
    assert '<source id="1" file="geo.pdf" page="1">' in user
    assert "ONLY the numbered sources" in system


def test_refuses_without_calling_llm_when_nothing_is_relevant(service):
    answer = service.query("quantum chromodynamics lattice gauge")
    assert answer.refused and answer.answer == REFUSAL
    assert answer.refusal_reason == "no_relevant_sources"
    assert service.llm.calls == []


def test_uncited_answers_become_refusals(service):
    service.llm = FakeLLM("Paris, obviously.")
    answer = service.query("What is the capital of France?")
    assert answer.refused and answer.refusal_reason == "uncited_answer"


def test_hallucinated_citation_numbers_are_dropped(service):
    service.llm = FakeLLM("Paris [1][9].")
    answer = service.query("What is the capital of France?")
    assert answer.answer == "Paris [1]."
    assert [c.number for c in answer.citations] == [1]


def test_retrieval_cache_hits_and_invalidates_on_ingest(service):
    q = "How do you make risotto?"
    assert not service.retriever.retrieve(q).cache_hit
    assert service.retriever.retrieve(q).cache_hit
    assert service.retriever.retrieve("  how do YOU make   risotto? ").cache_hit  # normalized key

    service.delete_document("doc-geo.pdf")  # bumps the corpus version
    assert not service.retriever.retrieve(q).cache_hit


def test_embeddings_are_cached_and_batched(service):
    fake = service.embedder.inner
    fake.calls.clear()
    texts = [f"text number {i}" for i in range(5)]
    service.embedder.embed(texts)
    assert fake.calls == [2, 2, 1]  # batch_size=2

    service.embedder.clear_memory()  # still served from the database cache
    service.embedder.embed(texts + texts)
    assert fake.calls == [2, 2, 1]


def test_reingest_replaces_chunks(service):
    before = service.store.count()
    doc = service.db.get_document("doc-geo.pdf")
    from .conftest import DOCS, make_chunks
    chunks = make_chunks("geo.pdf", DOCS["geo.pdf"][:1], "geography")
    service.ingest_chunks(chunks, "doc-geo.pdf", "geo.pdf", num_pages=1)
    assert service.store.count() == before - doc.num_chunks + 1

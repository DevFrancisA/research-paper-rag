"""RAGService: the ingestion and query pipelines used by the API, Inngest functions and evaluation."""

import time
from pathlib import Path
from typing import Any

from .chunking import chunk_pages, file_doc_id, load_pdf_pages
from .config import Settings
from .db import Database, make_database
from .embeddings import CachedEmbedder, OpenAIEmbedder, SparseEncoder
from .generation import (NAIVE_SYSTEM_PROMPT, REFUSAL, SYSTEM_PROMPT, ChatModel, OpenAIChat, build_naive_prompt,
                         build_user_prompt, extract_citations, is_refusal)
from .models import Answer, Chunk, DocumentInfo, IngestResult, MetadataFilter, RetrievalMode
from .reranker import CrossEncoderReranker, Reranker
from .retrieval import Retriever, apply_relevance_threshold
from .vector_store import VectorStore, make_client


class RAGService:
    def __init__(self, settings: Settings, db: Database, embedder: CachedEmbedder, sparse: SparseEncoder,
                 store: VectorStore, reranker: Reranker | None, llm: ChatModel):
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.sparse = sparse
        self.store = store
        self.llm = llm
        self.retriever = Retriever(store, embedder, sparse, reranker, db, candidate_k=settings.candidate_k,
                                   rerank_k=settings.rerank_k, cache_ttl_s=settings.retrieval_cache_ttl_s,
                                   reranker_name=settings.reranker_model if reranker else "")

    # ---------- ingestion ----------

    def ingest_pdf(self, path: str | Path, source: str | None = None, metadata: dict[str, Any] | None = None,
                   force: bool = False) -> IngestResult:
        start = time.perf_counter()
        path = Path(path)
        source = source or path.name
        doc_id = file_doc_id(path)

        existing = self.db.get_document(doc_id)
        if existing and not force:
            return IngestResult(doc_id=doc_id, source=existing.source, num_pages=existing.num_pages,
                                num_chunks=existing.num_chunks, embeddings_cached=0, embeddings_computed=0,
                                skipped=True)

        t = time.perf_counter()
        pages = load_pdf_pages(path)
        chunks = chunk_pages(pages, doc_id, source, self.settings.chunk_size, self.settings.chunk_overlap, metadata)
        timings = {"load_and_chunk": _ms(t)}
        result = self.ingest_chunks(chunks, doc_id, source, num_pages=len(pages), metadata=metadata)
        result.timings_ms = {**timings, **result.timings_ms, "total": _ms(start)}
        return result

    def ingest_chunks(self, chunks: list[Chunk], doc_id: str, source: str, num_pages: int,
                      metadata: dict[str, Any] | None = None) -> IngestResult:
        texts = [c.embedding_text() for c in chunks]

        t = time.perf_counter()
        dense, cached = self.embedder.embed_with_stats(texts) if texts else ([], 0)
        embed_ms = _ms(t)
        t = time.perf_counter()
        sparse = self.sparse.embed_documents(texts) if texts else []
        sparse_ms = _ms(t)

        t = time.perf_counter()
        self.store.delete_document(doc_id)  # replace any previous version of this document
        if chunks:
            self.store.upsert(chunks, dense, sparse)
        upsert_ms = _ms(t)

        self.db.upsert_document(DocumentInfo(doc_id=doc_id, source=source, num_pages=num_pages,
                                             num_chunks=len(chunks), metadata=metadata or {}))
        self.db.bump_corpus_version()
        return IngestResult(doc_id=doc_id, source=source, num_pages=num_pages, num_chunks=len(chunks),
                            embeddings_cached=cached, embeddings_computed=len(chunks) - cached,
                            timings_ms={"embed_dense": embed_ms, "embed_sparse": sparse_ms, "upsert": upsert_ms})

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        self.db.delete_document(doc_id)
        self.db.bump_corpus_version()

    def list_documents(self) -> list[DocumentInfo]:
        return self.db.list_documents()

    # ---------- querying ----------

    def query(self, question: str, top_k: int | None = None, flt: MetadataFilter | None = None,
              mode: RetrievalMode = RetrievalMode.HYBRID_RERANK, log: bool = True) -> Answer:
        start = time.perf_counter()
        top_k = top_k or self.settings.top_k
        retrieved = self.retriever.retrieve(question, top_k=top_k, mode=mode, flt=flt)
        sources = apply_relevance_threshold(retrieved.chunks, self.settings.relevance_threshold)
        timings = dict(retrieved.timings_ms)

        if not sources:
            answer = Answer(question=question, answer=REFUSAL, citations=[], sources=retrieved.chunks,
                            refused=True, refusal_reason="no_relevant_sources", cache_hit=retrieved.cache_hit)
        else:
            t = time.perf_counter()
            raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, sources))
            timings["generation"] = _ms(t)
            answer = self._finalize(question, raw, sources, retrieved.cache_hit)

        timings["total"] = _ms(start)
        answer.timings_ms = timings
        if log:
            self.db.log_query({
                "question": question, "mode": mode.value, "cache_hit": answer.cache_hit,
                "refused": answer.refused, "num_sources": len(answer.citations),
                "retrieval_ms": timings.get("retrieval_total"), "generation_ms": timings.get("generation"),
                "total_ms": timings["total"],
            })
        return answer

    def _finalize(self, question: str, raw: str, sources, cache_hit: bool) -> Answer:
        if is_refusal(raw):
            return Answer(question=question, answer=REFUSAL, citations=[], sources=sources, refused=True,
                          refusal_reason="model_refused", cache_hit=cache_hit)
        text, citations = extract_citations(raw, sources)
        if not citations and self.settings.require_citations:
            return Answer(question=question, answer=REFUSAL, citations=[], sources=sources, refused=True,
                          refusal_reason="uncited_answer", cache_hit=cache_hit)
        return Answer(question=question, answer=text, citations=citations, sources=sources, cache_hit=cache_hit)

    def naive_query(self, question: str, top_k: int | None = None) -> Answer:
        """Baseline for evaluation: dense retrieval, plain prompt, no threshold or citation checks."""
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=top_k or self.settings.top_k, mode=RetrievalMode.DENSE)
        raw = self.llm.complete(NAIVE_SYSTEM_PROMPT, build_naive_prompt(question, retrieved.chunks))
        return Answer(question=question, answer=raw, citations=[], sources=retrieved.chunks,
                      refused=is_refusal(raw), timings_ms={"total": _ms(start)})


def build_service(settings: Settings, llm: ChatModel | None = None) -> RAGService:
    db = make_database(settings.database_url, max_cache_items=settings.memory_cache_size * 10)
    embedder = CachedEmbedder(OpenAIEmbedder(settings.embedding_model, settings.embedding_dim), db,
                              batch_size=settings.embedding_batch_size, memory_size=settings.memory_cache_size)
    store = VectorStore(make_client(settings.qdrant_url, settings.qdrant_path), settings.collection,
                        settings.embedding_dim)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None
    return RAGService(settings, db, embedder, SparseEncoder(settings.sparse_model), store, reranker,
                      llm or OpenAIChat(settings.chat_model, settings.chat_temperature))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)

import asyncio
import json
import logging
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

import inngest
import inngest.fast_api
<<<<<<< HEAD
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
=======
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

>>>>>>> 887dabe57a6aba3b9c98461ffc930e31cf02fe74

from rag_production_app_test.config import settings
from rag_production_app_test.models import (Answer, DocumentInfo, IngestResult, MetadataFilter,
                                            RetrievalMode)
from rag_production_app_test.pipeline import RAGService, build_service

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

vector_storage = QdrantStorage()


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return build_service(settings)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        count=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
    limit=1,
    period=datetime.timedelta(seconds=20),
    key="event.data.source_id",
    )
)
async def rag_ingest_pdf(ctx: inngest.Context):
<<<<<<< HEAD
    data = ctx.event.data

    async def ingest() -> dict:
        result = await asyncio.to_thread(get_service().ingest_pdf, data["pdf_path"], data.get("source"),
                                         data.get("metadata"), bool(data.get("force", False)))
        return result.model_dump()

    return await ctx.step.run("ingest-pdf", ingest)
=======
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        vector_storage.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))


    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_text([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages":[
                {"role": "system", "content": "You answer questions using the only provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )


    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}
>>>>>>> 887dabe57a6aba3b9c98461ffc930e31cf02fe74


@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query")
)
async def rag_query(ctx: inngest.Context):
    data = ctx.event.data
    flt = MetadataFilter.model_validate(data["filter"]) if data.get("filter") else None

    async def answer() -> dict:
        result = await asyncio.to_thread(get_service().query, data["question"], data.get("top_k"), flt)
        return result.model_dump()

    return await ctx.step.run("retrieve-and-answer", answer)


<<<<<<< HEAD
app = FastAPI(title="Production RAG")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    filter: MetadataFilter | None = None
    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK


@app.get("/health")
def health() -> dict:
    service = get_service()
    return {"status": "ok", "chunks": service.store.count(), "documents": len(service.list_documents())}


@app.get("/documents")
def list_documents() -> list[DocumentInfo]:
    return get_service().list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    get_service().delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), metadata: str = Form("{}"), background: bool = Form(False),
                 force: bool = Form(False)) -> IngestResult | dict:
    """Upload a PDF. With background=true the work runs as a durable Inngest function."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"metadata must be a JSON object: {e}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    source = Path(file.filename).name
    path = upload_dir / f"{uuid.uuid4().hex}_{source}"
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    if background:
        ids = await inngest_client.send(inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": str(path.resolve()), "source": source, "metadata": meta, "force": force}))
        return {"queued": True, "event_ids": ids}
    return await asyncio.to_thread(get_service().ingest_pdf, path, source, meta, force)


@app.post("/query")
def query(req: QueryRequest) -> Answer:
    return get_service().query(req.question, req.top_k, req.filter, req.mode)


@app.get("/stats")
def stats() -> dict:
    service = get_service()
    return {"latency": service.db.query_latency_stats(), "embedding_cache": service.embedder.stats}


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query])
=======
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
>>>>>>> 887dabe57a6aba3b9c98461ffc930e31cf02fe74

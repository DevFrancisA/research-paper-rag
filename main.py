import asyncio
import json
import logging
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

import inngest
import inngest.fast_api
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

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


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return build_service(settings)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    data = ctx.event.data

    async def ingest() -> dict:
        result = await asyncio.to_thread(get_service().ingest_pdf, data["pdf_path"], data.get("source"),
                                         data.get("metadata"), bool(data.get("force", False)))
        return result.model_dump()

    return await ctx.step.run("ingest-pdf", ingest)


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

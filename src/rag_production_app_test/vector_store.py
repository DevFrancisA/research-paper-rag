"""Qdrant collection holding one point per chunk with a dense and a BM25 sparse vector."""

import warnings

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .models import Chunk, MetadataFilter, RetrievedChunk

DENSE = "dense"
SPARSE = "bm25"
INDEXED_FIELDS = {"doc_id": qm.PayloadSchemaType.KEYWORD, "source": qm.PayloadSchemaType.KEYWORD,
                  "page": qm.PayloadSchemaType.INTEGER}


def make_client(url: str, path: str) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url) if url else QdrantClient(path=path)


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            existing = self.client.get_collection(self.collection).config.params.vectors
            size = existing[DENSE].size if isinstance(existing, dict) else None
            if size != self.dim:
                raise ValueError(
                    f"Collection {self.collection!r} has dense size {size}, but EMBEDDING_DIM={self.dim}. "
                    "Use a new QDRANT_COLLECTION or delete the old one."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        with warnings.catch_warnings():  # embedded Qdrant warns that indexes are a no-op there
            warnings.simplefilter("ignore", UserWarning)
            for name, schema in INDEXED_FIELDS.items():
                self.client.create_payload_index(self.collection, name, field_schema=schema)

    def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[qm.SparseVector],
               batch_size: int = 256) -> None:
        points = [
            qm.PointStruct(id=c.chunk_id, vector={DENSE: d, SPARSE: s}, payload=_payload(c))
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[start:start + batch_size], wait=True)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])), wait=True)

    def search_dense(self, vector: list[float], limit: int, flt: MetadataFilter | None = None):
        return self._query(query=vector, using=DENSE, limit=limit, flt=flt)

    def search_sparse(self, vector: qm.SparseVector, limit: int, flt: MetadataFilter | None = None):
        return self._query(query=vector, using=SPARSE, limit=limit, flt=flt)

    def search_hybrid(self, dense: list[float], sparse: qm.SparseVector, limit: int, prefetch_limit: int,
                      flt: MetadataFilter | None = None) -> list[RetrievedChunk]:
        qfilter = build_filter(flt)
        response = self.client.query_points(
            self.collection,
            prefetch=[
                qm.Prefetch(query=dense, using=DENSE, limit=prefetch_limit, filter=qfilter),
                qm.Prefetch(query=sparse, using=SPARSE, limit=prefetch_limit, filter=qfilter),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved(p) for p in response.points]

    def _query(self, query, using: str, limit: int, flt: MetadataFilter | None) -> list[RetrievedChunk]:
        response = self.client.query_points(
            self.collection, query=query, using=using, limit=limit,
            query_filter=build_filter(flt), with_payload=True,
        )
        return [_to_retrieved(p) for p in response.points]

    def scroll_chunks(self, limit: int | None = None) -> list[Chunk]:
        """All chunks in the collection (used to build evaluation benchmarks)."""
        out: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True)
            out.extend(_payload_to_chunk(str(p.id), p.payload) for p in points)
            if offset is None or (limit and len(out) >= limit):
                return out[:limit] if limit else out

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count


def build_filter(flt: MetadataFilter | None) -> qm.Filter | None:
    if flt is None or flt.is_empty():
        return None
    must: list[qm.Condition] = []
    if flt.sources:
        must.append(qm.FieldCondition(key="source", match=qm.MatchAny(any=flt.sources)))
    if flt.doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=flt.doc_ids)))
    if flt.page_gte is not None or flt.page_lte is not None:
        must.append(qm.FieldCondition(key="page", range=qm.Range(gte=flt.page_gte, lte=flt.page_lte)))
    for key, value in (flt.metadata or {}).items():
        must.append(qm.FieldCondition(key=f"metadata.{key}", match=qm.MatchValue(value=value)))
    return qm.Filter(must=must)


def _payload(c: Chunk) -> dict:
    return {"doc_id": c.doc_id, "source": c.source, "page": c.page, "chunk_index": c.chunk_index,
            "text": c.text, "metadata": c.metadata}


def _payload_to_chunk(point_id: str, payload: dict) -> Chunk:
    return Chunk(chunk_id=point_id, doc_id=payload["doc_id"], source=payload["source"], page=payload.get("page"),
                 chunk_index=payload["chunk_index"], text=payload["text"], metadata=payload.get("metadata") or {})


def _to_retrieved(point) -> RetrievedChunk:
    chunk = _payload_to_chunk(str(point.id), point.payload)
    return RetrievedChunk(**chunk.model_dump(), score=point.score)

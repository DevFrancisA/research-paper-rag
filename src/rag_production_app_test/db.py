"""Relational storage: document registry, cache tables, corpus version and query log.

PostgresDatabase is used when DATABASE_URL is set; MemoryDatabase otherwise (tests, quick local runs).
"""

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

from .models import DocumentInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    num_pages    INT NOT NULL,
    num_chunks   INT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS cache (
    namespace   TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       BYTEA NOT NULL,
    expires_at  TIMESTAMPTZ,
    PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS counters (
    name   TEXT PRIMARY KEY,
    value  BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    id             BIGSERIAL PRIMARY KEY,
    question       TEXT NOT NULL,
    mode           TEXT NOT NULL,
    cache_hit      BOOLEAN NOT NULL,
    refused        BOOLEAN NOT NULL,
    num_sources    INT NOT NULL,
    retrieval_ms   REAL,
    generation_ms  REAL,
    total_ms       REAL NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Database(Protocol):
    def cache_get_many(self, namespace: str, keys: list[str]) -> dict[str, bytes]: ...
    def cache_set_many(self, namespace: str, items: dict[str, bytes], ttl_s: int | None = None) -> None: ...
    def cache_clear(self, namespace: str | None = None) -> None: ...
    def corpus_version(self) -> int: ...
    def bump_corpus_version(self) -> int: ...
    def upsert_document(self, doc: DocumentInfo) -> None: ...
    def get_document(self, doc_id: str) -> DocumentInfo | None: ...
    def delete_document(self, doc_id: str) -> None: ...
    def list_documents(self) -> list[DocumentInfo]: ...
    def log_query(self, record: dict[str, Any]) -> None: ...
    def query_latency_stats(self) -> dict[str, float]: ...


class MemoryDatabase:
    def __init__(self, max_cache_items: int = 100_000):
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], tuple[bytes, float | None]] = OrderedDict()
        self._max = max_cache_items
        self._docs: dict[str, DocumentInfo] = {}
        self._version = 0
        self._log: list[dict[str, Any]] = []

    def cache_get_many(self, namespace, keys):
        now = time.time()
        out = {}
        with self._lock:
            for key in keys:
                entry = self._cache.get((namespace, key))
                if entry is None:
                    continue
                value, expires = entry
                if expires is not None and expires < now:
                    del self._cache[(namespace, key)]
                    continue
                self._cache.move_to_end((namespace, key))
                out[key] = value
        return out

    def cache_set_many(self, namespace, items, ttl_s=None):
        expires = time.time() + ttl_s if ttl_s else None
        with self._lock:
            for key, value in items.items():
                self._cache[(namespace, key)] = (value, expires)
                self._cache.move_to_end((namespace, key))
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def cache_clear(self, namespace=None):
        with self._lock:
            if namespace is None:
                self._cache.clear()
            else:
                for k in [k for k in self._cache if k[0] == namespace]:
                    del self._cache[k]

    def corpus_version(self):
        return self._version

    def bump_corpus_version(self):
        with self._lock:
            self._version += 1
            return self._version

    def upsert_document(self, doc):
        self._docs[doc.doc_id] = doc

    def get_document(self, doc_id):
        return self._docs.get(doc_id)

    def delete_document(self, doc_id):
        self._docs.pop(doc_id, None)

    def list_documents(self):
        return sorted(self._docs.values(), key=lambda d: d.source)

    def log_query(self, record):
        self._log.append(record)

    def query_latency_stats(self):
        return _latency_stats([r["total_ms"] for r in self._log])


class PostgresDatabase:
    def __init__(self, url: str):
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(url, min_size=1, max_size=10, open=True)
        with self._pool.connection() as conn:
            conn.execute(SCHEMA)

    def cache_get_many(self, namespace, keys):
        if not keys:
            return {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM cache WHERE namespace = %s AND key = ANY(%s)"
                " AND (expires_at IS NULL OR expires_at > now())",
                (namespace, keys),
            ).fetchall()
        return {key: bytes(value) for key, value in rows}

    def cache_set_many(self, namespace, items, ttl_s=None):
        if not items:
            return
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO cache (namespace, key, value, expires_at)"
                " VALUES (%s, %s, %s, CASE WHEN %s::int IS NULL THEN NULL"
                "         ELSE now() + make_interval(secs => %s::int) END)"
                " ON CONFLICT (namespace, key) DO UPDATE"
                " SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at",
                [(namespace, k, v, ttl_s, ttl_s) for k, v in items.items()],
            )

    def cache_clear(self, namespace=None):
        with self._pool.connection() as conn:
            if namespace is None:
                conn.execute("DELETE FROM cache")
            else:
                conn.execute("DELETE FROM cache WHERE namespace = %s", (namespace,))

    def corpus_version(self):
        with self._pool.connection() as conn:
            row = conn.execute("SELECT value FROM counters WHERE name = 'corpus_version'").fetchone()
        return row[0] if row else 0

    def bump_corpus_version(self):
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO counters (name, value) VALUES ('corpus_version', 1)"
                " ON CONFLICT (name) DO UPDATE SET value = counters.value + 1 RETURNING value"
            ).fetchone()
        return row[0]

    def upsert_document(self, doc):
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, source, num_pages, num_chunks, metadata)"
                " VALUES (%s, %s, %s, %s, %s)"
                " ON CONFLICT (doc_id) DO UPDATE SET source = EXCLUDED.source,"
                " num_pages = EXCLUDED.num_pages, num_chunks = EXCLUDED.num_chunks,"
                " metadata = EXCLUDED.metadata, ingested_at = now()",
                (doc.doc_id, doc.source, doc.num_pages, doc.num_chunks, json.dumps(doc.metadata)),
            )

    def get_document(self, doc_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT doc_id, source, num_pages, num_chunks, metadata FROM documents WHERE doc_id = %s",
                (doc_id,),
            ).fetchone()
        return _row_to_doc(row) if row else None

    def delete_document(self, doc_id):
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def list_documents(self):
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT doc_id, source, num_pages, num_chunks, metadata FROM documents ORDER BY source"
            ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def log_query(self, record):
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO query_log (question, mode, cache_hit, refused, num_sources,"
                " retrieval_ms, generation_ms, total_ms)"
                " VALUES (%(question)s, %(mode)s, %(cache_hit)s, %(refused)s, %(num_sources)s,"
                " %(retrieval_ms)s, %(generation_ms)s, %(total_ms)s)",
                record,
            )

    def query_latency_stats(self):
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT total_ms FROM query_log ORDER BY id DESC LIMIT 5000").fetchall()
        return _latency_stats([r[0] for r in rows])


def _row_to_doc(row) -> DocumentInfo:
    doc_id, source, num_pages, num_chunks, metadata = row
    return DocumentInfo(doc_id=doc_id, source=source, num_pages=num_pages, num_chunks=num_chunks,
                        metadata=metadata or {})


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    s = sorted(values)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return {"count": len(s), "p50_ms": pct(0.5), "p95_ms": pct(0.95), "mean_ms": sum(s) / len(s)}


def make_database(url: str, max_cache_items: int = 100_000) -> Database:
    return PostgresDatabase(url) if url else MemoryDatabase(max_cache_items)

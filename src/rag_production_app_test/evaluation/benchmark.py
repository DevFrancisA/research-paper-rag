"""Synthetic benchmark generation from the indexed corpus.

Answerable items: one question per sampled chunk; that chunk is the gold retrieval target.
Unanswerable items: on-topic questions the corpus does not answer (verified against the top retrieved
chunks), used to measure refusals and hallucination.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ..models import Chunk, RetrievalMode
from ..pipeline import RAGService

T = TypeVar("T", bound=BaseModel)

STYLES = {
    "keyword": "Use the specific names, terms or numbers that appear in the passage, as someone who already "
               "knows the terminology would.",
    "paraphrase": "Do not copy more than three consecutive words from the passage. Use synonyms and describe "
                  "concepts in your own words, as someone unfamiliar with the document's terminology would.",
}


class BenchmarkItem(BaseModel):
    id: str
    question: str
    answerable: bool
    style: str
    reference_answer: str | None = None
    gold_chunk_ids: list[str] = []
    gold_doc_id: str | None = None
    gold_page: int | None = None


class _Generated(BaseModel):
    question: str
    answer: str


class _Unanswerable(BaseModel):
    question: str


class _Verdict(BaseModel):
    answered_by_sources: bool


def structured(client, model: str, system: str, user: str, schema: type[T]) -> T:
    response = client.chat.completions.parse(
        model=model, response_format=schema,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Model returned no parsable output: {response.choices[0].message.refusal}")
    return parsed


def sample_chunks(chunks: list[Chunk], n: int, seed: int, min_chars: int = 300) -> list[Chunk]:
    """Round-robin across documents so large documents don't dominate the benchmark.

    Prefers chunks of at least min_chars (enough content for a specific question), falling back to
    shorter ones when there aren't n of those.
    """
    rng = random.Random(seed)
    if sum(len(c.text) >= min_chars for c in chunks) < n:
        min_chars = 40
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text) >= min_chars:
            by_doc[c.doc_id].append(c)
    for group in by_doc.values():
        rng.shuffle(group)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    picked: list[Chunk] = []
    while len(picked) < n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(picked) < n:
                picked.append(by_doc[d].pop())
    return picked


def generate_benchmark(service: RAGService, client, model: str, n_answerable: int = 200,
                       n_unanswerable: int = 50, seed: int = 7) -> list[BenchmarkItem]:
    chunks = service.store.scroll_chunks()
    if not chunks:
        raise RuntimeError("The collection is empty. Ingest some PDFs first.")
    items: list[BenchmarkItem] = []
    seen_questions: set[str] = set()

    for i, chunk in enumerate(sample_chunks(chunks, n_answerable, seed)):
        style = "keyword" if i % 2 == 0 else "paraphrase"
        gen = structured(client, model,
                         "You write evaluation questions for a document search system.",
                         f"Passage (from {chunk.source}, page {chunk.page}):\n\"\"\"\n{chunk.text}\n\"\"\"\n\n"
                         "Write one question a real user might ask that this passage answers, and the answer "
                         "taken from the passage. The question must be specific enough that the passage is "
                         "clearly the place to find the answer, and must make sense without seeing the passage: "
                         f"never say 'the passage', 'this document' or 'the text'. {STYLES[style]}",
                         _Generated)
        key = gen.question.strip().lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        items.append(BenchmarkItem(id=f"a{i:04d}", question=gen.question.strip(), answerable=True, style=style,
                                   reference_answer=gen.answer.strip(), gold_chunk_ids=[chunk.chunk_id],
                                   gold_doc_id=chunk.doc_id, gold_page=chunk.page))
        print(f"[benchmark] answerable {len(items)}/{n_answerable}", flush=True)
    if len(items) < n_answerable:
        print(f"[benchmark] warning: only {len(items)} answerable questions; the corpus has too few chunks "
              f"for {n_answerable}. Ingest more documents for a more reliable benchmark.", flush=True)

    # Unanswerable: generate from a passage's topic, then discard any the corpus can actually answer.
    rng = random.Random(seed + 1)
    attempts, made = 0, 0
    while made < n_unanswerable and attempts < n_unanswerable * 3:
        attempts += 1
        chunk = rng.choice(chunks)
        gen = structured(client, model,
                         "You write evaluation questions for a document search system.",
                         f"Passage (from {chunk.source}):\n\"\"\"\n{chunk.text}\n\"\"\"\n\n"
                         "Write one question on the same topic, using similar vocabulary, that this passage does "
                         "NOT answer and that the rest of the document is unlikely to answer either (for example, "
                         "asking for a detail, figure, comparison or date that is never given). It should sound "
                         "like a reasonable question a user would ask. Never say 'the passage' or 'the document'.",
                         _Unanswerable)
        question = gen.question.strip()
        if question.lower() in seen_questions:
            continue
        top = service.retriever.retrieve(question, top_k=8, mode=RetrievalMode.HYBRID_RERANK).chunks
        context = "\n\n".join(f"[{j}] {c.text}" for j, c in enumerate(top, start=1))
        verdict = structured(client, model,
                             "You check whether a question can be answered from given sources.",
                             f"Sources:\n{context}\n\nQuestion: {question}\n\n"
                             "Do the sources contain the information needed to answer this question, fully or "
                             "partly?", _Verdict)
        if verdict.answered_by_sources:
            continue
        seen_questions.add(question.lower())
        items.append(BenchmarkItem(id=f"u{made:04d}", question=question, answerable=False, style="unanswerable"))
        made += 1
        print(f"[benchmark] unanswerable {made}/{n_unanswerable}", flush=True)
    return items


def save_benchmark(items: list[BenchmarkItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8")


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [BenchmarkItem.model_validate(json.loads(line)) for line in lines if line.strip()]

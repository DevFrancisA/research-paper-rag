"""Hallucination and grounded-answer accuracy: naive baseline vs the grounded pipeline.

Definitions (per benchmark item):
- hallucinated: the system answered (did not decline) and the answer contains a claim the context it was
  given does not support.
- grounded-correct: answerable -> answered, fully supported by its context and consistent with the
  reference answer; unanswerable -> declined.
"""

from pydantic import BaseModel

from ..models import Answer
from ..pipeline import RAGService
from .benchmark import BenchmarkItem, structured
from .metrics import mean, relative_change

JUDGE_SYSTEM = """You grade answers produced by a document question-answering system. Be strict and literal.

- declined: the answer says it cannot answer / does not know, and gives no substantive answer.
- supported: every factual claim in the answer is stated in, or directly follows from, the CONTEXT. Claims from
  general knowledge that are not in the CONTEXT count as unsupported. A declined answer is supported.
- correct: the answer is consistent with the REFERENCE answer and contains its key facts. If there is no
  reference answer (the question is unanswerable from the documents), correct means the answer declined."""


class Judgement(BaseModel):
    declined: bool
    supported: bool
    correct: bool
    unsupported_claims: list[str]


def judge(client, model: str, item: BenchmarkItem, answer: Answer) -> Judgement:
    if answer.refused and answer.refusal_reason in ("no_relevant_sources", "uncited_answer"):
        # Declined by the pipeline itself; nothing for the judge to read.
        return Judgement(declined=True, supported=True, correct=not item.answerable, unsupported_claims=[])
    context = "\n\n".join(f"[{i}] ({c.source}, p.{c.page}) {c.text}" for i, c in enumerate(answer.sources, 1))
    reference = item.reference_answer if item.answerable else "(none - not answerable from the documents)"
    return structured(client, model, JUDGE_SYSTEM,
                      f"QUESTION: {item.question}\n\nREFERENCE: {reference}\n\nCONTEXT:\n{context}\n\n"
                      f"ANSWER:\n{answer.answer}", Judgement)


def evaluate_answers(service: RAGService, items: list[BenchmarkItem], client, judge_model: str) -> dict:
    service.retriever.cache_enabled = False
    systems = {"naive": lambda q: service.naive_query(q),
               "grounded": lambda q: service.query(q, log=False)}
    per_system: dict[str, list[dict]] = {name: [] for name in systems}
    try:
        for n, item in enumerate(items, start=1):
            for name, run in systems.items():
                answer = run(item.question)
                j = judge(client, judge_model, item, answer)
                hallucinated = not j.declined and not j.supported
                if item.answerable:
                    grounded_correct = not j.declined and j.supported and j.correct
                else:
                    grounded_correct = j.declined
                per_system[name].append({
                    "id": item.id, "answerable": item.answerable, "declined": j.declined,
                    "supported": j.supported, "correct": j.correct, "hallucinated": hallucinated,
                    "grounded_correct": grounded_correct, "unsupported_claims": j.unsupported_claims,
                    "answer": answer.answer, "num_citations": len(answer.citations),
                })
            if n % 10 == 0:
                print(f"[answers] {n}/{len(items)}", flush=True)
    finally:
        service.retriever.cache_enabled = True

    summary = {name: _summarize(rows) for name, rows in per_system.items()}
    summary["comparison"] = {
        "hallucination_rate_change": round(relative_change(summary["naive"]["hallucination_rate"],
                                                           summary["grounded"]["hallucination_rate"]), 4),
        "grounded_accuracy_change_pts": round(summary["grounded"]["grounded_accuracy"]
                                              - summary["naive"]["grounded_accuracy"], 4),
    }
    return {"summary": summary, "rows": per_system}


def _summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "n": len(rows),
        "hallucination_rate": round(mean([r["hallucinated"] for r in rows]), 4),
        "grounded_accuracy": round(mean([r["grounded_correct"] for r in rows]), 4),
        "answerable_accuracy": round(mean([r["grounded_correct"] for r in answerable]), 4),
        "answerable_decline_rate": round(mean([r["declined"] for r in answerable]), 4),
        "unanswerable_decline_rate": round(mean([r["declined"] for r in unanswerable]), 4),
    }

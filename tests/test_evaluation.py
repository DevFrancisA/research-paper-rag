from types import SimpleNamespace

from rag_production_app_test.evaluation.answer_eval import Judgement, evaluate_answers
from rag_production_app_test.evaluation.benchmark import (BenchmarkItem, generate_benchmark, load_benchmark,
                                                          save_benchmark)
from rag_production_app_test.evaluation.latency_eval import evaluate_latency
from rag_production_app_test.evaluation.retrieval_eval import evaluate_retrieval


class FakeOpenAI:
    """Stands in for openai.OpenAI().chat.completions.parse: returns a canned object per schema."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))
        self.n = 0

    def _parse(self, model, response_format, messages):
        self.n += 1
        name = response_format.__name__
        if name == "_Generated":
            value = response_format(question=f"What is the capital of France? ({self.n})", answer="Paris")
        elif name == "_Unanswerable":
            value = response_format(question=f"What is the population of Lyon in 1850? ({self.n})")
        elif name == "_Verdict":
            value = response_format(answered_by_sources=False)
        else:
            value = Judgement(declined=False, supported=True, correct=True, unsupported_claims=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=value, refusal=None))])


def test_benchmark_roundtrip_and_evals(service, tmp_path):
    items = generate_benchmark(service, FakeOpenAI(), "fake", n_answerable=3, n_unanswerable=2, seed=1)
    assert sum(i.answerable for i in items) == 3 and sum(not i.answerable for i in items) == 2
    path = tmp_path / "bench.jsonl"
    save_benchmark(items, path)
    assert load_benchmark(path) == items

    retrieval = evaluate_retrieval(service, items)
    assert retrieval["n_queries"] == 3
    assert set(retrieval["modes"]) == {"dense", "sparse", "hybrid", "hybrid_rerank"}
    assert 0 <= retrieval["modes"]["hybrid_rerank"]["overall"]["recall@5"] <= 1

    answers = evaluate_answers(service, items, FakeOpenAI(), "fake")
    assert set(answers["summary"]) == {"naive", "grounded", "comparison"}
    assert len(answers["rows"]["grounded"]) == len(items)

    latency = evaluate_latency(service, items, n_requests=20, n_embed_texts=3)
    assert latency["baseline"]["n"] == 20 and latency["cached"]["n"] == 20
    assert latency["cache_hit_rate"] > 0  # 20 requests over 5 questions must repeat
    assert service.retriever.cache_enabled and service.retriever.embedder is service.embedder


def test_unanswerable_judgement_rules(service):
    item = BenchmarkItem(id="u1", question="quantum chromodynamics lattice", answerable=False, style="unanswerable")
    result = evaluate_answers(service, [item], FakeOpenAI(), "fake")
    grounded = result["rows"]["grounded"][0]
    assert grounded["declined"] and grounded["grounded_correct"] and not grounded["hallucinated"]

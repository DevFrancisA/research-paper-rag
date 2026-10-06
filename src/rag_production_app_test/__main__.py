"""Command line: `uv run python -m rag_production_app_test <command>`.

  ingest <folder-or-pdf>   index PDFs
  generate-benchmark       build eval/benchmark.jsonl from the indexed corpus
  eval-retrieval           Recall@k / MRR / nDCG for each retrieval mode
  eval-answers             hallucination rate and grounded-answer accuracy, naive vs grounded
  eval-latency             end-to-end latency with/without caching, embedding batching
  eval-all                 all three evaluations, then the report
  report                   rebuild eval/results/REPORT.md from saved results
"""

import argparse
import json
from pathlib import Path

from .config import settings

RESULTS = Path("eval/results")
BENCHMARK = Path("eval/benchmark.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_production_app_test")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("path")
    p.add_argument("--metadata", default="{}", help='JSON object stored on every chunk, e.g. {"category":"hr"}')
    p.add_argument("--force", action="store_true", help="re-index files that are already indexed")

    p = sub.add_parser("generate-benchmark")
    p.add_argument("--answerable", type=int, default=200)
    p.add_argument("--unanswerable", type=int, default=50)
    p.add_argument("--model", default=None, help="question-writing model (default: JUDGE_MODEL or gpt-4.1)")
    p.add_argument("--out", default=str(BENCHMARK))

    for name in ("eval-retrieval", "eval-answers", "eval-latency", "eval-all"):
        p = sub.add_parser(name)
        p.add_argument("--benchmark", default=str(BENCHMARK))
        p.add_argument("--judge-model", default=None)
        p.add_argument("--requests", type=int, default=300, help="eval-latency workload size")
        p.add_argument("--limit", type=int, default=None, help="only use the first N benchmark items")

    sub.add_parser("report")
    args = parser.parse_args()

    if args.command == "report":
        write_report()
        return

    from .pipeline import build_service

    service = build_service(settings)
    try:
        _run(args, service)
    finally:
        service.store.client.close()


def _run(args, service) -> None:
    if args.command == "ingest":
        _ingest(service, Path(args.path), json.loads(args.metadata), args.force)
        return

    import os

    from openai import OpenAI

    client = OpenAI()
    judge_model = getattr(args, "judge_model", None) or getattr(args, "model", None) \
        or os.getenv("JUDGE_MODEL", "gpt-4.1")

    from .evaluation.benchmark import generate_benchmark, load_benchmark, save_benchmark

    if args.command == "generate-benchmark":
        items = generate_benchmark(service, client, judge_model, args.answerable, args.unanswerable)
        save_benchmark(items, args.out)
        print(f"Wrote {len(items)} items to {args.out}")
        return

    items = load_benchmark(args.benchmark)[:args.limit]
    run = {"eval-retrieval": ["retrieval"], "eval-answers": ["answers"], "eval-latency": ["latency"],
           "eval-all": ["retrieval", "answers", "latency"]}[args.command]
    config = {"embedding_model": settings.embedding_model, "chat_model": settings.chat_model,
              "reranker_model": settings.reranker_model, "judge_model": judge_model,
              "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
              "candidate_k": settings.candidate_k, "rerank_k": settings.rerank_k, "top_k": settings.top_k,
              "relevance_threshold": settings.relevance_threshold, "chunks_indexed": service.store.count(),
              "documents": len(service.list_documents())}

    if "retrieval" in run:
        from .evaluation.retrieval_eval import evaluate_retrieval
        _save("retrieval", {"config": config, **evaluate_retrieval(service, items)})
    if "answers" in run:
        from .evaluation.answer_eval import evaluate_answers
        _save("answers", {"config": config, **evaluate_answers(service, items, client, judge_model)})
    if "latency" in run:
        from .evaluation.latency_eval import evaluate_latency
        _save("latency", {"config": config, **evaluate_latency(service, items, n_requests=args.requests)})
    write_report()


def _ingest(service, path: Path, metadata: dict, force: bool) -> None:
    files = sorted(path.rglob("*.pdf")) if path.is_dir() else [path]
    for f in files:
        r = service.ingest_pdf(f, metadata=metadata, force=force)
        status = "skipped (already indexed)" if r.skipped else (
            f"{r.num_chunks} chunks, {r.embeddings_computed} embedded, {r.embeddings_cached} from cache, "
            f"{r.timings_ms.get('total', 0):.0f} ms")
        print(f"{f.name}: {status}")


def _save(name: str, data: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {out}")


def _load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report() -> None:
    lines = ["# Evaluation report", "", "Generated by `python -m rag_production_app_test report`.", ""]

    if r := _load("retrieval"):
        c = r["config"]
        lines += [f"## Retrieval ({r['n_queries']} queries, {c['chunks_indexed']} chunks, "
                  f"{c['documents']} documents)", "",
                  "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Page hit@5 |",
                  "|---|---|---|---|---|---|---|"]
        for mode, m in r["modes"].items():
            o = m["overall"]
            lines.append(f"| {mode} | {_pct(o['recall@1'])} | {_pct(o['recall@5'])} | {_pct(o['recall@10'])} | "
                         f"{o['mrr@10']:.3f} | {o['ndcg@10']:.3f} | {_pct(o['page_hit@5'])} |")
        lines += ["", "Recall@5 by question style:", "", "| Mode | keyword | paraphrase |", "|---|---|---|"]
        for mode, m in r["modes"].items():
            s = m["by_style"]
            lines.append(f"| {mode} | " + " | ".join(_pct(s[k]["recall@5"]) if k in s else "-"
                                                     for k in ("keyword", "paraphrase")) + " |")
        lines.append("")

    if r := _load("answers"):
        s = r["summary"]
        lines += [f"## Answers ({s['naive']['n']} questions, judge: {r['config']['judge_model']})", "",
                  "| System | Hallucination rate | Grounded accuracy | Answerable accuracy | "
                  "Declined (answerable) | Declined (unanswerable) |", "|---|---|---|---|---|---|"]
        for name in ("naive", "grounded"):
            m = s[name]
            lines.append(f"| {name} | {_pct(m['hallucination_rate'])} | {_pct(m['grounded_accuracy'])} | "
                         f"{_pct(m['answerable_accuracy'])} | {_pct(m['answerable_decline_rate'])} | "
                         f"{_pct(m['unanswerable_decline_rate'])} |")
        cmp = s["comparison"]
        lines += ["", f"Hallucination rate change: {_pct(cmp['hallucination_rate_change'])}. "
                      f"Grounded accuracy change: {cmp['grounded_accuracy_change_pts'] * 100:+.1f} pts.", ""]

    if r := _load("latency"):
        b, c = r["baseline"], r["cached"]
        lines += [f"## Latency ({r['n_requests']} requests, {r['unique_questions']} unique questions)", "",
                  "| Config | p50 | p95 | mean |", "|---|---|---|---|",
                  f"| no cache | {b['p50_ms']:.0f} ms | {b['p95_ms']:.0f} ms | {b['mean_ms']:.0f} ms |",
                  f"| cached | {c['p50_ms']:.0f} ms | {c['p95_ms']:.0f} ms | {c['mean_ms']:.0f} ms |", "",
                  f"Median change: {_pct(r['p50_change'])}. Cache hit rate: {_pct(r['cache_hit_rate'])}.", ""]
        if e := r.get("embedding_batching"):
            lines += [f"Embedding {e['n_texts']} chunks: {e['unbatched_ms']:.0f} ms one at a time, "
                      f"{e['batched_ms']:.0f} ms batched ({e['batched_speedup']}x), "
                      f"{e['db_cache_ms']:.0f} ms from the database cache.", ""]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()

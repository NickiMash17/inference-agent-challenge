"""End-to-end evaluation: runs the full answer() pipeline (router -> SQL or
retrieve/rewrite_retrieve/id_lookup -> grounded answer) against every item
in evals/kb_gold.jsonl and scores correctness, route accuracy, citation
validity, refusal behaviour, latency and cost.

NEVER run this against evals/kb_heldout.jsonl - see CLAUDE.md. This script
only ever reads evals/kb_gold.jsonl; there is no --heldout flag here at all.

Costs real money (the ANSWER_MODEL and ROUTER_MODEL are real Anthropic
models). Aborts as soon as cumulative cost would exceed
Settings.max_cost_per_eval_run_usd.

Usage:
    uv run python evals/e2e_eval.py [--reps N] [--save-baseline NAME]

--reps > 1 requires --save-baseline (a multi-rep run is for recording a
baseline, not quick iteration).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.ingest import load_all  # noqa: E402
from rag.pipeline import AnswerResult, build_answer_pipelines  # noqa: E402
from rag.pipeline import answer as answer_query  # noqa: E402
from rag.sql_store import describe_schema, get_connection  # noqa: E402
from shared.config import Settings  # noqa: E402
from shared.llm import AnthropicClient, LLMClient  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
EVALS_DIR = ROOT / "evals"
GOLD_PATH = EVALS_DIR / "kb_gold.jsonl"
INDEX_CACHE_DIR = ROOT / "data" / "index"
RESULTS_DIR = EVALS_DIR / "results"
BASELINES_DIR = EVALS_DIR / "baselines"

#: Matches our ID formats: used to decide whether a "lookup" question is
#: ID-anchored (and should therefore route to id_lookup).
_ID_IN_TEXT = re.compile(r"\b(?:S-\d{3}|SP-\d{3}|R-\d{2})\b", re.IGNORECASE)

#: The original spec for this eval said "all 25 items"; kb_gold.jsonl has
#: since grown to ~30 (a `semantic` category was added, and G-016 moved
#: cross_doc -> aggregation). This runs over every item currently in the
#: file rather than hard-capping at a stale count.


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_number(text: str) -> float | None:
    match = re.search(r"-?\d[\d,]*\.?\d*", text)
    if not match:
        return None
    return float(match.group().replace(",", ""))


def check_correctness(item: dict, result: AnswerResult) -> bool:
    expected_value = item["expected_value"]

    if expected_value == "NOT_IN_KB":
        return result.not_in_kb is True

    if isinstance(expected_value, bool):
        return _normalize(str(expected_value)) in _normalize(result.answer)

    if isinstance(expected_value, int | float):
        got = _extract_number(result.answer)
        if got is None:
            return False
        tolerance = item.get("tolerance", 0)
        return abs(got - float(expected_value)) <= tolerance

    expected_str = str(expected_value)
    if expected_str in ("Yes", "No"):
        return bool(re.search(rf"\b{re.escape(expected_str.lower())}\b", _normalize(result.answer)))

    return _normalize(expected_str) in _normalize(result.answer)


def expected_routes(item: dict) -> set[str] | None:
    """None means "any route is acceptable" (only true for negative items,
    which are instead checked on not_in_kb, not on route)."""
    answer_type = item["answer_type"]
    if answer_type == "lookup":
        return {"id_lookup"} if _ID_IN_TEXT.search(item["question"]) else {"id_lookup", "retrieve"}
    if answer_type == "aggregation":
        return {"sql"}
    if answer_type == "semantic":
        return {"rewrite_retrieve", "retrieve"}
    if answer_type in ("policy", "cross_doc"):
        return {"id_lookup", "retrieve"}
    if answer_type == "negative":
        return None
    return None


def git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


class CostLimitExceeded(RuntimeError):
    pass


def run_eval(llm: LLMClient, settings: Settings, reps: int) -> dict:
    gold = load_gold()
    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))

    chunks = load_all(KB_DIR)
    con = get_connection(KB_DIR)
    schema = describe_schema(con)
    pipelines = build_answer_pipelines(chunks, cache_dir=INDEX_CACHE_DIR)

    total_cost = 0.0
    per_item_records: dict[str, list[dict]] = defaultdict(list)  # item id -> [record per rep]
    latencies_s: list[float] = []

    for _rep in range(reps):
        for item in gold:
            result = answer_query(
                item["question"],
                con=con,
                schema=schema,
                retrieval_pipelines=pipelines,
                llm=llm,
                router_model=settings.router_model,
                answer_model=settings.answer_model,
            )

            total_cost += result.cost_usd
            if total_cost > settings.max_cost_per_eval_run_usd:
                raise CostLimitExceeded(
                    f"cumulative cost ${total_cost:.4f} exceeds "
                    f"MAX_COST_PER_EVAL_RUN_USD (${settings.max_cost_per_eval_run_usd:.2f}); "
                    "aborting the eval run."
                )

            correct = check_correctness(item, result)
            routes_ok = expected_routes(item)
            route_correct = routes_ok is None or result.route in routes_ok
            end_to_end_latency = sum(result.stage_latencies_s.values())
            latencies_s.append(end_to_end_latency)

            per_item_records[item["id"]].append(
                {
                    "question": item["question"],
                    "answer_type": item["answer_type"],
                    "expected_value": item["expected_value"],
                    "answer": result.answer,
                    "route": result.route,
                    "route_correct": route_correct,
                    "citations": result.citations,
                    "invalid_citations": result.invalid_citations,
                    "not_in_kb": result.not_in_kb,
                    "low_confidence": result.low_confidence,
                    "correct": correct,
                    "cost_usd": result.cost_usd,
                    "latency_s": end_to_end_latency,
                    "sql": result.sql_result.sql if result.sql_result else None,
                    "errors": result.errors,
                }
            )

    return _summarize(gold, per_item_records, total_cost, latencies_s, reps, manifest, settings)


def _summarize(
    gold: list[dict],
    per_item_records: dict[str, list[dict]],
    total_cost: float,
    latencies_s: list[float],
    reps: int,
    manifest: dict,
    settings: Settings,
) -> dict:
    by_type_correct: dict[str, list[int]] = defaultdict(list)
    by_type_n: dict[str, int] = defaultdict(int)
    route_correct_count = 0
    route_total = 0
    citation_valid_count = 0
    citation_total = 0
    negative_refused = 0
    negative_total = 0
    false_refusals = 0
    inconsistent_items = []

    for item in gold:
        records = per_item_records[item["id"]]
        correctness_values = [r["correct"] for r in records]
        by_type_correct[item["answer_type"]].extend(correctness_values)
        by_type_n[item["answer_type"]] += len(records)

        if len(set(correctness_values)) > 1:
            inconsistent_items.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "correctness_by_rep": correctness_values,
                }
            )

        for r in records:
            route_total += 1
            route_correct_count += int(r["route_correct"])

            if r["citations"] or r["invalid_citations"]:
                citation_total += 1
                citation_valid_count += int(bool(r["citations"]) or r["not_in_kb"])

            if item["answer_type"] == "negative":
                negative_total += 1
                negative_refused += int(r["not_in_kb"])
            elif r["not_in_kb"]:
                false_refusals += 1

    by_type_summary = {
        answer_type: {"correct": sum(values), "n": len(values)}
        for answer_type, values in by_type_correct.items()
    }

    lat_ms = sorted(s * 1000 for s in latencies_s) if latencies_s else [0.0]
    n_queries = sum(by_type_n.values())

    return {
        "kb_version": manifest["kb_version"],
        "git_commit_sha": git_commit_sha(),
        "router_model": settings.router_model,
        "answer_model": settings.answer_model,
        "reps": reps,
        "n_gold_items": len(gold),
        "n_queries_total": n_queries,
        "by_answer_type": by_type_summary,
        "route_accuracy": {"correct": route_correct_count, "n": route_total},
        "citation_validity": {"valid_or_refused": citation_valid_count, "n": citation_total},
        "negative_refusal": {"refused": negative_refused, "n": negative_total},
        "false_refusals_on_answerable_items": false_refusals,
        "latency_ms": {
            "p50": round(float(np.percentile(lat_ms, 50)), 2),
            "p95": round(float(np.percentile(lat_ms, 95)), 2),
        },
        "cost_usd": {
            "total": round(total_cost, 4),
            "per_query": round(total_cost / n_queries, 6) if n_queries else 0.0,
        },
        "inconsistent_items": inconsistent_items,
        "per_item_records": {k: v for k, v in per_item_records.items()},
    }


def print_report(summary: dict) -> None:
    print(
        f"kb_version={summary['kb_version']}  git={summary['git_commit_sha'][:12]}  "
        f"router={summary['router_model']}  answer={summary['answer_model']}  "
        f"reps={summary['reps']}"
    )
    print()
    print("Correctness by answer_type:")
    for answer_type, counts in summary["by_answer_type"].items():
        print(f"  {answer_type}: {counts['correct']}/{counts['n']}")
    print()
    ra = summary["route_accuracy"]
    print(f"Route accuracy: {ra['correct']}/{ra['n']}")
    cv = summary["citation_validity"]
    print(f"Citation validity: {cv['valid_or_refused']}/{cv['n']}")
    nr = summary["negative_refusal"]
    print(f"Refusal on negatives: {nr['refused']}/{nr['n']}")
    print(f"False refusals on answerable items: {summary['false_refusals_on_answerable_items']}")
    print(f"Latency: p50={summary['latency_ms']['p50']}ms p95={summary['latency_ms']['p95']}ms")
    print(
        f"Cost: total=${summary['cost_usd']['total']:.4f} "
        f"per_query=${summary['cost_usd']['per_query']:.6f}"
    )

    if summary["inconsistent_items"]:
        print(f"\n{len(summary['inconsistent_items'])} item(s) inconsistent across reps:")
        for inc in summary["inconsistent_items"]:
            print(f"  [{inc['id']}] {inc['question']} -> {inc['correctness_by_rep']}")

    failures = [
        {"id": item_id, **rec}
        for item_id, records in summary["per_item_records"].items()
        for rec in records
        if not rec["correct"]
    ]
    if failures:
        print(f"\n{len(failures)} failing record(s):")
        for f in failures:
            print(f"  [{f['id']}] {f['question']}")
            print(f"      expected: {f['expected_value']!r}")
            print(f"      answer:   {f['answer']!r}")
            print(f"      route:    {f['route']}  citations: {f['citations']}")
            if f["sql"]:
                print(f"      sql:      {f['sql']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--save-baseline", default=None, help="write evals/baselines/<name>.json")
    args = parser.parse_args()

    if args.reps > 1 and not args.save_baseline:
        parser.error("--reps > 1 requires --save-baseline (a multi-rep run records a baseline)")

    settings = Settings()
    if not settings.atlas_anthropic_api_key:
        parser.error("ATLAS_ANTHROPIC_API_KEY is not set - cannot run a live e2e eval")

    llm = AnthropicClient(settings)

    try:
        summary = run_eval(llm, settings, args.reps)
    except CostLimitExceeded as exc:
        print(f"ABORTED: {exc}")
        sys.exit(1)

    print_report(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {"timestamp": timestamp, **summary}
    out_path = RESULTS_DIR / f"e2e_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")

    if args.save_baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINES_DIR / f"{args.save_baseline}.json"
        baseline_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {baseline_path}")


if __name__ == "__main__":
    main()

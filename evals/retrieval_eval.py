"""Retrieval evaluation: recall@1/3/5 and MRR for a configurable retrieval
pipeline (see rag/pipeline.py), over the lookup/semantic/cross_doc/policy
gold items.

Aggregation items are skipped here; they're checked against ground truth
directly via SQL in evals/sql_sanity.py instead.

Usage:
    uv run python evals/retrieval_eval.py [--config NAME] [--save-baseline NAME]

Every run writes a scratch copy to evals/results/ (gitignored). Passing
--save-baseline additionally writes a committed copy to evals/baselines/.

evals/kb_heldout.jsonl is evaluated once, at submission time, by a human -
see CLAUDE.md and README "Evaluation". --heldout refuses to run unless
--final is also passed AND the HELDOUT_CONFIRM environment variable is set
to the exact current kb_version, e.g.:
    HELDOUT_CONFIRM=1.0.2 uv run python evals/retrieval_eval.py \\
        --config v6_id_pinning --heldout --final
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.index import NumpyVectorIndex  # noqa: E402
from rag.ingest import load_all  # noqa: E402
from rag.pipeline import CONFIGS, Pipeline, RetrievedItem  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
EVALS_DIR = ROOT / "evals"
GOLD_PATH = EVALS_DIR / "kb_gold.jsonl"
HELDOUT_PATH = EVALS_DIR / "kb_heldout.jsonl"
INDEX_CACHE_DIR = ROOT / "data" / "index"
RESULTS_DIR = EVALS_DIR / "results"
BASELINES_DIR = EVALS_DIR / "baselines"

#: answer_types evaluated by retrieval here; "aggregation" goes through
#: evals/sql_sanity.py instead, and "negative" has no source_refs to score.
EVAL_ANSWER_TYPES = ("lookup", "semantic", "cross_doc", "policy")

K_VALUES = (1, 3, 5)
MAX_K = max(K_VALUES)


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ref_matches(gold_ref: str, chunk_ref: str) -> bool:
    """A gold ref matches a chunk ref exactly, or (for a table-level gold
    ref like "venue_guide.pdf#rooms") matches any row-level refinement of
    it (e.g. "venue_guide.pdf#rooms:R-04")."""
    if gold_ref == chunk_ref:
        return True
    return ":" not in gold_ref and chunk_ref.startswith(gold_ref + ":")


def refs_visible_at_k(retrieved: list[RetrievedItem], k: int) -> list[str]:
    """Refs "visible" at cutoff k: base (non-expanded) items ranked <= k,
    plus any FK-expanded item whose parent base item is ranked <= k. For
    configs without FK expansion this is just the ordinary top-k refs."""
    visible = []
    base_ids_within_k = set()
    for item in retrieved:
        if not item.expanded and item.rank <= k:
            visible.append(item.chunk.source_ref)
            base_ids_within_k.add(item.chunk.chunk_id)
    for item in retrieved:
        if item.expanded and item.expanded_from in base_ids_within_k:
            visible.append(item.chunk.source_ref)
    return visible


def first_hit_rank(source_refs: list[str], base_items: list[RetrievedItem]) -> int | None:
    """Rank of the first base (non-expanded) item that matches a gold ref.
    Used for MRR, which reflects ranking quality and is left unaffected by
    FK expansion (a context-enrichment side channel, not a ranking change)."""
    for item in sorted(base_items, key=lambda i: i.rank):
        if any(ref_matches(g, item.chunk.source_ref) for g in source_refs):
            return item.rank
    return None


def evaluate_retriever(name: str, pipeline: Pipeline, items: list[dict]) -> dict:
    per_item = []
    for item in items:
        retrieved = pipeline.search(item["question"], k=MAX_K)
        unresolved_ids = list(pipeline.last_unresolved_ids)
        base_items = [r for r in retrieved if not r.expanded]

        rank = first_hit_rank(item["source_refs"], base_items)
        hit_at = {
            k: any(
                ref_matches(g, ref) for ref in refs_visible_at_k(retrieved, k)
                for g in item["source_refs"]
            )
            for k in K_VALUES
        }
        all_refs_at_5 = (
            all(
                any(ref_matches(g, ref) for ref in refs_visible_at_k(retrieved, 5))
                for g in item["source_refs"]
            )
            if item["answer_type"] == "cross_doc"
            else None
        )

        per_item.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer_type": item["answer_type"],
                "source_refs": item["source_refs"],
                "rank": rank,
                "hit_at": hit_at,
                "all_refs_at_5": all_refs_at_5,
                "unresolved_ids": unresolved_ids,
                "top5": [
                    {"source_ref": r.chunk.source_ref, "score": round(r.score, 4)}
                    for r in sorted(base_items, key=lambda i: i.rank)[:5]
                ],
            }
        )

    def metrics_for(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for k in K_VALUES:
            hits = sum(1 for r in subset if r["hit_at"][k])
            out[f"recall@{k}"] = round(hits / n, 3)
        mrr = sum((1.0 / r["rank"]) if r["rank"] is not None else 0.0 for r in subset) / n
        out["mrr"] = round(mrr, 3)
        return out

    by_type = {}
    for answer_type in EVAL_ANSWER_TYPES:
        subset = [r for r in per_item if r["answer_type"] == answer_type]
        metrics = metrics_for(subset)
        if answer_type == "cross_doc" and subset:
            metrics["all_refs@5"] = round(
                sum(1 for r in subset if r["all_refs_at_5"]) / len(subset), 3
            )
        by_type[answer_type] = metrics

    result = {
        "retriever": name,
        "overall": metrics_for(per_item),
        "by_answer_type": by_type,
        "per_item": per_item,
    }

    def latency_stats(latencies_s: list[float]) -> dict:
        lat_ms = sorted(s * 1000 for s in latencies_s)
        return {
            "p50": round(float(np.percentile(lat_ms, 50)), 2),
            "p95": round(float(np.percentile(lat_ms, 95)), 2),
        }

    if pipeline.query_latencies:
        result["query_latency_ms"] = latency_stats(pipeline.query_latencies)
    if pipeline.rerank_latencies:
        result["rerank_latency_ms"] = latency_stats(pipeline.rerank_latencies)
    return result


def results_table(result: dict) -> pd.DataFrame:
    rows = []
    for answer_type, metrics in result["by_answer_type"].items():
        if metrics["n"] == 0:
            continue
        rows.append({"answer_type": answer_type, **metrics})
    rows.append({"answer_type": "OVERALL", **result["overall"]})
    df = pd.DataFrame(rows)
    cols = ["answer_type", "n", "recall@1", "recall@3", "recall@5", "mrr", "all_refs@5"]
    return df.reindex(columns=[c for c in cols if c in df.columns])


def print_failure_report(result: dict) -> None:
    failures = [r for r in result["per_item"] if not r["hit_at"][1] or not r["hit_at"][5]]
    if not failures:
        print("\nNo misses at @1 or @5.")
        return
    print(f"\n{len(failures)} item(s) missed at @1 or @5:")
    for r in failures:
        print(f"  [{r['id']}] {r['question']}")
        print(f"      expected: {r['source_refs']}")
        if r["unresolved_ids"]:
            print(f"      unresolved IDs in query: {r['unresolved_ids']}")
        for rank, hit in enumerate(r["top5"], start=1):
            print(f"      #{rank} {hit['source_ref']}  (score={hit['score']})")


def git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run_config(config_name: str, gold_path: Path = GOLD_PATH) -> dict:
    config = CONFIGS[config_name]
    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))

    chunks = load_all(KB_DIR)
    gold = [item for item in load_gold(gold_path) if item["answer_type"] in EVAL_ANSWER_TYPES]

    pipeline = Pipeline(config, chunks, cache_dir=INDEX_CACHE_DIR)
    result = evaluate_retriever(config_name, pipeline, gold)

    print(f"=== {config_name} (gold set: {gold_path.name}) ===")
    print(results_table(result).to_string(index=False))
    if "query_latency_ms" in result:
        print(f"\nquery latency: p50={result['query_latency_ms']['p50']}ms "
              f"p95={result['query_latency_ms']['p95']}ms")
    print_failure_report(result)

    return {
        "config": config_name,
        "gold_set": gold_path.name,
        "kb_version": manifest["kb_version"],
        "embedding_model": NumpyVectorIndex.MODEL_NAME,
        "git_commit_sha": git_commit_sha(),
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "chunk_count": len(chunks),
        "gold_item_count": len(gold),
        "result": result,
    }


def heldout_guard_error(
    heldout: bool, final: bool, kb_version: str, env: dict[str, str]
) -> str | None:
    """Returns a refusal message if a --heldout run should be blocked, or
    None if it's allowed to proceed. A pure function (no file I/O, no
    search), so it can be tested exhaustively without ever triggering an
    actual held-out evaluation.

    Two independent confirmations are required, on top of --heldout itself:
      1. --final - states intent to run the real, one-time evaluation.
      2. HELDOUT_CONFIRM env var set to the exact current kb_version - a
         value that can't be set by habit or muscle memory, so it only
         ever matches when someone deliberately checked the KB version and
         typed it in right before running.
    """
    if not heldout:
        return None
    if not final:
        return (
            "--heldout refuses to run without --final. The held-out set "
            "(evals/kb_heldout.jsonl) is evaluated once, at submission time - "
            "iterate against evals/kb_gold.jsonl instead."
        )
    confirm = env.get("HELDOUT_CONFIRM")
    if confirm != kb_version:
        return (
            "--heldout --final also requires the HELDOUT_CONFIRM environment "
            f"variable set to the exact current kb_version ({kb_version!r}); "
            f"got {confirm!r}. This is deliberately not something you'd set by "
            "accident - only set it when you mean to run the real, one-time "
            "submission evaluation."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v0_baseline", choices=sorted(CONFIGS))
    parser.add_argument(
        "--save-baseline", default=None, help="also write evals/baselines/<name>.json"
    )
    parser.add_argument(
        "--heldout", action="store_true",
        help="evaluate against evals/kb_heldout.jsonl instead of kb_gold.jsonl",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="required alongside --heldout; the held-out set is evaluated once, at "
        "submission time, not during iteration",
    )
    args = parser.parse_args()

    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))
    error = heldout_guard_error(args.heldout, args.final, manifest["kb_version"], os.environ)
    if error:
        parser.error(error)

    gold_path = HELDOUT_PATH if args.heldout else GOLD_PATH
    payload = run_config(args.config, gold_path=gold_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_{payload['timestamp']}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    if args.save_baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINES_DIR / f"{args.save_baseline}.json"
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {baseline_path}")


if __name__ == "__main__":
    main()

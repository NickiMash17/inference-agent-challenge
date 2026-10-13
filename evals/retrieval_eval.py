"""Retrieval evaluation: recall@1/3/5 and MRR for BM25, vector and hybrid
retrievers, over the lookup/semantic/cross_doc/policy gold items.

Aggregation items are skipped here; they're checked against ground truth
directly via SQL in evals/sql_sanity.py instead.

Usage:
    uv run python evals/retrieval_eval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.index import BM25Index, HybridRetriever, NumpyVectorIndex  # noqa: E402
from rag.ingest import load_all  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
EVALS_DIR = ROOT / "evals"
GOLD_PATH = EVALS_DIR / "kb_gold.jsonl"
INDEX_CACHE_DIR = ROOT / "data" / "index"
RESULTS_DIR = EVALS_DIR / "results"

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


def first_hit_rank(source_refs: list[str], retrieved_refs: list[str]) -> int | None:
    for rank, ref in enumerate(retrieved_refs, start=1):
        if any(ref_matches(g, ref) for g in source_refs):
            return rank
    return None


def all_refs_hit_within_k(source_refs: list[str], retrieved_refs: list[str], k: int) -> bool:
    top_k = retrieved_refs[:k]
    return all(any(ref_matches(g, r) for r in top_k) for g in source_refs)


def evaluate_retriever(name: str, retriever, items: list[dict]) -> dict:
    per_item = []
    for item in items:
        results = retriever.search(item["question"], k=MAX_K)
        retrieved_refs = [c.source_ref for c, _score in results]
        rank = first_hit_rank(item["source_refs"], retrieved_refs)
        per_item.append(
            {
                "id": item["id"],
                "answer_type": item["answer_type"],
                "rank": rank,
                "all_refs_at_5": (
                    all_refs_hit_within_k(item["source_refs"], retrieved_refs, 5)
                    if item["answer_type"] == "cross_doc"
                    else None
                ),
            }
        )

    def metrics_for(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for k in K_VALUES:
            hits = sum(1 for r in subset if r["rank"] is not None and r["rank"] <= k)
            out[f"recall@{k}"] = round(hits / n, 3)
        mrr = sum((1.0 / r["rank"]) if r["rank"] is not None else 0.0 for r in subset) / n
        out["mrr"] = round(mrr, 3)
        return out

    by_type = {}
    for answer_type in EVAL_ANSWER_TYPES:
        subset = [r for r in per_item if r["answer_type"] == answer_type]
        metrics = metrics_for(subset)
        if answer_type == "cross_doc" and subset:
            all_hit = sum(1 for r in subset if r["all_refs_at_5"]) / len(subset)
            metrics["all_refs@5"] = round(all_hit, 3)
        by_type[answer_type] = metrics

    return {
        "retriever": name,
        "overall": metrics_for(per_item),
        "by_answer_type": by_type,
        "per_item": per_item,
    }


def results_table(all_results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in all_results:
        for answer_type, metrics in res["by_answer_type"].items():
            if metrics["n"] == 0:
                continue
            row = {"retriever": res["retriever"], "answer_type": answer_type, **metrics}
            rows.append(row)
        rows.append({"retriever": res["retriever"], "answer_type": "OVERALL", **res["overall"]})
    df = pd.DataFrame(rows)
    cols = [
        "retriever", "answer_type", "n", "recall@1", "recall@3", "recall@5", "mrr", "all_refs@5",
    ]
    return df.reindex(columns=[c for c in cols if c in df.columns])


def git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))
    kb_version = manifest["kb_version"]

    chunks = load_all(KB_DIR)
    gold = [item for item in load_gold() if item["answer_type"] in EVAL_ANSWER_TYPES]

    bm25 = BM25Index()
    bm25.add(chunks)
    vector = NumpyVectorIndex(cache_dir=INDEX_CACHE_DIR)
    vector.add(chunks)
    hybrid = HybridRetriever(bm25, vector)

    all_results = [
        evaluate_retriever("bm25", bm25, gold),
        evaluate_retriever("vector", vector, gold),
        evaluate_retriever("hybrid", hybrid, gold),
    ]

    table = results_table(all_results)
    print(table.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"retrieval_{timestamp}.json"
    payload = {
        "kb_version": kb_version,
        "embedding_model": NumpyVectorIndex.MODEL_NAME,
        "git_commit_sha": git_commit_sha(),
        "timestamp": timestamp,
        "chunk_count": len(chunks),
        "gold_item_count": len(gold),
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

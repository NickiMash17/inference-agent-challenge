"""Proves the DuckDB store agrees with the frozen gold set: for every
aggregation gold item, runs the equivalent hand-written SQL against
sessions_active (or sessions, where the question is explicitly about
cancellations) and asserts it equals expected_value, within tolerance.

Usage:
    uv run python evals/sql_sanity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.sql_store import get_connection  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
GOLD_PATH = ROOT / "evals" / "kb_gold.jsonl"


def load_aggregation_items(path: Path = GOLD_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return [item for item in items if item["answer_type"] == "aggregation"]


def sql_for_item(item: dict) -> str:
    """One hand-written query per gold item ID, proving the SQL store
    agrees with ground truth. Not a general text-to-SQL translator."""
    gid = item["id"]
    if gid == "G-007":
        return "SELECT COUNT(*) FROM sessions_active WHERE day = 1"
    if gid == "G-008":
        return "SELECT COUNT(*) FROM sessions_active WHERE track = 'Safety'"
    if gid == "G-009":
        return "SELECT ROUND(AVG(capacity), 1) FROM sessions_active"
    if gid == "G-010":
        return (
            "SELECT session_id FROM sessions_active "
            "ORDER BY capacity DESC, session_id ASC LIMIT 1"
        )
    if gid == "G-011":
        return "SELECT COUNT(*) FROM sessions WHERE status = 'Cancelled'"
    if gid == "G-012":
        match = re.search(r"SP-\d+", item["question"])
        if not match:
            raise ValueError(f"{gid}: couldn't find a speaker ID in the question")
        speaker_id = match.group()
        return (
            "SELECT COUNT(DISTINCT ss.session_id) FROM sessions_active sa "
            "JOIN session_speakers ss ON sa.session_id = ss.session_id "
            f"WHERE ss.speaker_id = '{speaker_id}'"
        )
    raise ValueError(f"No hand-written SQL registered for aggregation item {gid!r}")


def check(item: dict, actual) -> bool:
    expected = item["expected_value"]
    tolerance = item.get("tolerance")
    if tolerance is not None:
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def main() -> int:
    con = get_connection(KB_DIR)
    items = load_aggregation_items()

    all_passed = True
    for item in items:
        sql = sql_for_item(item)
        actual = con.execute(sql).fetchone()[0]
        passed = check(item, actual)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {item['id']}: {item['question']}")
        print(f"       sql:      {sql}")
        print(f"       actual:   {actual!r}")
        print(f"       expected: {item['expected_value']!r}" + (
            f" (tolerance {item['tolerance']})" if "tolerance" in item else ""
        ))

    print()
    print("ALL PASSED" if all_passed else "SOME FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

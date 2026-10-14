"""Tests for the generated Part 1 knowledge base (data/kb/ + evals/kb_gold.jsonl)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_kb  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
EVALS_DIR = ROOT / "evals"

PDF_SECTIONS = {"rooms", "tickets", "policies"}


def _read_sessions(kb_dir: Path = KB_DIR) -> list[dict]:
    with (kb_dir / "sessions.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_speakers(kb_dir: Path = KB_DIR) -> list[dict]:
    with (kb_dir / "speakers.json").open(encoding="utf-8") as f:
        return json.load(f)


def _read_gold(evals_dir: Path = EVALS_DIR) -> list[dict]:
    with (evals_dir / "kb_gold.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_heldout(evals_dir: Path = EVALS_DIR) -> list[dict]:
    with (evals_dir / "kb_heldout.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def sessions() -> list[dict]:
    return _read_sessions()


@pytest.fixture(scope="module")
def speakers() -> list[dict]:
    return _read_speakers()


@pytest.fixture(scope="module")
def gold() -> list[dict]:
    return _read_gold()


@pytest.fixture(scope="module")
def heldout() -> list[dict]:
    return _read_heldout()


def test_kb_files_exist():
    assert (KB_DIR / "sessions.csv").exists()
    assert (KB_DIR / "speakers.json").exists()
    assert (KB_DIR / "venue_guide.pdf").exists()
    assert (KB_DIR / "manifest.json").exists()
    assert (EVALS_DIR / "kb_gold.jsonl").exists()
    assert (EVALS_DIR / "kb_heldout.jsonl").exists()


def test_foreign_keys_resolve(sessions, speakers):
    speaker_ids = {sp["speaker_id"] for sp in speakers}
    session_ids = {s["session_id"] for s in sessions}
    room_ids = {r["room_id"] for r in generate_kb.ROOMS}

    for s in sessions:
        assert s["room_id"] in room_ids, f"{s['session_id']} references unknown room {s['room_id']}"
        for sid in s["speaker_ids"].split(";"):
            assert sid in speaker_ids, f"{s['session_id']} references unknown speaker {sid}"

    # reverse direction: every speaker's session_ids must exist, and must
    # actually list that speaker as a presenter of that session.
    sessions_by_id = {s["session_id"]: s for s in sessions}
    for sp in speakers:
        for sid in sp["session_ids"]:
            assert sid in session_ids, f"{sp['speaker_id']} references unknown session {sid}"
            assert sp["speaker_id"] in sessions_by_id[sid]["speaker_ids"].split(";")

    # and every session's speakers must list that session back.
    speakers_by_id = {sp["speaker_id"]: sp for sp in speakers}
    for s in sessions:
        for sid in s["speaker_ids"].split(";"):
            assert s["session_id"] in speakers_by_id[sid]["session_ids"]


def test_no_room_overlaps_among_non_cancelled_sessions(sessions):
    def to_minutes(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    by_room_day: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for s in sessions:
        if s["status"] == "Cancelled":
            continue
        key = (s["room_id"], s["day"])
        by_room_day.setdefault(key, []).append(
            (to_minutes(s["start_time"]), to_minutes(s["end_time"]), s["session_id"])
        )

    for key, intervals in by_room_day.items():
        intervals.sort()
        for (_, end1, id1), (start2, _, id2) in zip(intervals, intervals[1:], strict=False):
            assert end1 <= start2, f"Overlap in room/day {key}: {id1} and {id2}"


def test_all_emails_use_example_domains(speakers):
    for sp in speakers:
        email = sp["email"]
        assert email.endswith("@example.com") or email.endswith("@example.org"), (
            f"{sp['speaker_id']} has a non-example email: {email}"
        )


def test_generator_is_deterministic(tmp_path):
    kb_a = tmp_path / "a" / "kb"
    evals_a = tmp_path / "a" / "evals"
    kb_b = tmp_path / "b" / "kb"
    evals_b = tmp_path / "b" / "evals"

    generate_kb.generate(kb_a, evals_a, seed=generate_kb.SEED)
    generate_kb.generate(kb_b, evals_b, seed=generate_kb.SEED)

    for name in ("sessions.csv", "speakers.json", "venue_guide.pdf", "manifest.json"):
        hash_a = hashlib.sha256((kb_a / name).read_bytes()).hexdigest()
        hash_b = hashlib.sha256((kb_b / name).read_bytes()).hexdigest()
        assert hash_a == hash_b, f"{name} is not byte-identical across runs"

    for name in ("kb_gold.jsonl", "kb_heldout.jsonl"):
        hash_a = hashlib.sha256((evals_a / name).read_bytes()).hexdigest()
        hash_b = hashlib.sha256((evals_b / name).read_bytes()).hexdigest()
        assert hash_a == hash_b, f"{name} is not byte-identical across runs"


def test_gold_source_refs_resolve(sessions, speakers, gold):
    session_ids = {s["session_id"] for s in sessions}
    speaker_ids = {sp["speaker_id"] for sp in speakers}
    kb_files = {"sessions.csv", "speakers.json", "venue_guide.pdf"}

    for item in gold:
        for ref in item["source_refs"]:
            assert "#" in ref or ref in kb_files, f"{item['id']}: malformed ref {ref!r}"
            if "#" in ref:
                file_name, anchor = ref.split("#", 1)
                assert file_name in kb_files, f"{item['id']}: unknown file in ref {ref!r}"
                if file_name == "sessions.csv":
                    assert anchor in session_ids, f"{item['id']}: unknown session {anchor}"
                elif file_name == "speakers.json":
                    assert anchor in speaker_ids, f"{item['id']}: unknown speaker {anchor}"
                elif file_name == "venue_guide.pdf":
                    assert anchor in PDF_SECTIONS, f"{item['id']}: unknown PDF section {anchor}"


def test_gold_set_coverage(gold):
    # G-016 was relabelled cross_doc -> aggregation in kb_version 1.0.1 (it's
    # fundamentally a COUNT+GROUP BY+JOIN query, not a simple two-hop
    # lookup), so aggregation's floor moved up and cross_doc's moved down.
    counts = Counter(item["answer_type"] for item in gold)
    assert counts["aggregation"] >= 7
    assert counts["lookup"] >= 6
    assert counts["cross_doc"] >= 4
    assert counts["policy"] >= 4
    assert counts["negative"] >= 3
    assert counts["semantic"] >= 5

    assert any(
        "cancelled" in item["question"].lower() or item["expected_answer"] == "Cancelled"
        for item in gold
    )
    assert any("moved" in item["question"].lower() for item in gold)


def test_gold_items_have_expected_value(gold):
    for item in gold:
        assert "expected_value" in item, f"{item['id']} is missing expected_value"
        assert item["expected_value"] not in (None, ""), f"{item['id']} has an empty expected_value"


def test_negative_items_use_not_in_kb_sentinel(gold):
    negatives = [item for item in gold if item["answer_type"] == "negative"]
    assert negatives, "expected at least one negative item"
    for item in negatives:
        assert item["expected_value"] == "NOT_IN_KB", (
            f"{item['id']} is negative but expected_value is {item['expected_value']!r}"
        )


def test_semantic_items_avoid_shared_key_terms_and_ids(sessions, gold):
    import re

    session_ids = {s["session_id"] for s in sessions}
    sessions_by_id = {s["session_id"]: s for s in sessions}

    semantics = [item for item in gold if item["answer_type"] == "semantic"]
    assert len(semantics) >= 5

    for item in semantics:
        target = sessions_by_id[item["expected_value"]]
        question_lower = item["question"].lower()

        assert not any(sid in item["question"] for sid in session_ids), (
            f"{item['id']} leaks a session ID in its question"
        )

        title_words = {w for w in re.findall(r"[a-z]+", target["title"].lower()) if len(w) > 3}
        question_words = set(re.findall(r"[a-z]+", question_lower))
        shared = title_words & question_words
        assert not shared, f"{item['id']} shares key terms {shared} with its target title"


def test_planted_edge_cases_present(sessions, speakers):
    assert any(s["status"] == "Cancelled" for s in sessions)
    assert any(s["status"] == "Moved" and "Moved from" in s["notes"] for s in sessions)
    assert any(len(s["speaker_ids"].split(";")) >= 2 for s in sessions)

    speaker_session_counts: dict[str, int] = {}
    for s in sessions:
        for sid in s["speaker_ids"].split(";"):
            speaker_session_counts[sid] = speaker_session_counts.get(sid, 0) + 1
    assert sum(1 for c in speaker_session_counts.values() if c >= 2) >= 2

    non_accessible_rooms = {
        r["room_id"] for r in generate_kb.ROOMS if r["wheelchair_accessible"] == "No"
    }
    assert any(
        s["room_id"] in non_accessible_rooms and s["status"] == "Scheduled" for s in sessions
    )

    names = [sp["full_name"] for sp in speakers]
    assert "Jon Mokoena" in names and "John Mokoena" in names
    orgs = {sp["full_name"]: sp["organisation"] for sp in speakers}
    assert orgs["Jon Mokoena"] != orgs["John Mokoena"]


def test_conference_dates_are_fictional_and_fixed():
    assert generate_kb.CONFERENCE_DATES == {1: "2026-11-18", 2: "2026-11-19"}
    for date_str in generate_kb.CONFERENCE_DATES.values():
        datetime.strptime(date_str, "%Y-%m-%d")


def test_per_table_rngs_are_independent():
    rng_speakers = generate_kb.make_rng(generate_kb.SEED, "speakers")
    rng_sessions = generate_kb.make_rng(generate_kb.SEED, "sessions")
    assert rng_speakers.getstate() != rng_sessions.getstate()
    # and each is independently reproducible from its own seed string
    assert generate_kb.make_rng(generate_kb.SEED, "speakers").getstate() == rng_speakers.getstate()


def test_speaker_field_changes_do_not_affect_sessions(tmp_path, monkeypatch):
    """Proves the per-table RNG split: a new/changed speaker field must not
    shift a single byte of sessions.csv, since they now draw from
    independent random streams."""
    kb_before = tmp_path / "before" / "kb"
    generate_kb.generate(kb_before, tmp_path / "before" / "evals", seed=generate_kb.SEED)
    sessions_before = (kb_before / "sessions.csv").read_bytes()
    speakers_before = (kb_before / "speakers.json").read_bytes()

    original_make_bio = generate_kb.make_bio

    def make_bio_with_extra_field(rng, name, job_title, org):
        rng.randint(0, 1_000_000)  # simulate a new speaker field consuming extra randomness
        return original_make_bio(rng, name, job_title, org)

    monkeypatch.setattr(generate_kb, "make_bio", make_bio_with_extra_field)

    kb_after = tmp_path / "after" / "kb"
    generate_kb.generate(kb_after, tmp_path / "after" / "evals", seed=generate_kb.SEED)
    sessions_after = (kb_after / "sessions.csv").read_bytes()
    speakers_after = (kb_after / "speakers.json").read_bytes()

    assert sessions_after == sessions_before, (
        "sessions.csv changed after perturbing speaker generation; RNGs are not isolated"
    )
    # sanity: the perturbation should actually have changed something (bios),
    # otherwise this test would pass vacuously
    assert speakers_after != speakers_before


def test_highest_capacity_session_is_unique(sessions):
    going = [s for s in sessions if s["status"] != "Cancelled"]
    max_capacity = max(int(s["capacity"]) for s in going)
    winners = [s["session_id"] for s in going if int(s["capacity"]) == max_capacity]
    assert len(winners) == 1, f"capacity tie for the max among {winners}"


def test_most_sessions_speaker_is_unique(sessions):
    counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        if s["status"] == "Cancelled":
            continue
        for sid in s["speaker_ids"].split(";"):
            counts[sid] += 1
    max_count = max(counts.values())
    winners = [sid for sid, c in counts.items() if c == max_count]
    assert len(winners) == 1, f"speaker tie for the most sessions among {winners}"


def test_gold_max_questions_match_the_unique_winner(sessions, gold):
    going = [s for s in sessions if s["status"] != "Cancelled"]

    capacity_item = next(
        item
        for item in gold
        if item["question"].startswith("Which session (by ID) has the highest capacity")
    )
    max_capacity = max(int(s["capacity"]) for s in going)
    winner = next(s["session_id"] for s in going if int(s["capacity"]) == max_capacity)
    assert capacity_item["expected_value"] == winner

    speaker_item = next(
        item
        for item in gold
        if item["question"].startswith("Which speaker (by ID) has given the most sessions")
    )
    counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        if s["status"] == "Cancelled":
            continue
        for sid in s["speaker_ids"].split(";"):
            counts[sid] += 1
    max_count = max(counts.values())
    speaker_winner = next(sid for sid, c in counts.items() if c == max_count)
    assert speaker_item["expected_value"] == speaker_winner


def test_average_capacity_gold_item_has_tolerance(sessions, gold):
    item = next(item for item in gold if "average capacity" in item["question"].lower())
    assert "tolerance" in item
    assert item["tolerance"] == pytest.approx(0.05)

    going = [s for s in sessions if s["status"] != "Cancelled"]
    recomputed = round(sum(int(s["capacity"]) for s in going) / len(going), 1)
    assert abs(item["expected_value"] - recomputed) <= item["tolerance"] + 1e-9


def test_planted_edge_cases_have_disjoint_speakers():
    """No speaker may carry more than one planted edge case. Rebuilds the
    internal (pre-CSV) session objects, which still carry the `_role` tag
    stripped from sessions.csv, so each trap's exact designated session can
    be identified rather than any session that incidentally matches."""
    rng_speakers = generate_kb.make_rng(generate_kb.SEED, "speakers")
    rng_sessions = generate_kb.make_rng(generate_kb.SEED, "sessions")
    fake_speakers = generate_kb.make_faker(generate_kb.SEED, "speakers")
    speakers = generate_kb.build_speakers(rng_speakers, fake_speakers)
    sessions = generate_kb.build_sessions(rng_sessions, speakers)

    by_role = {s["_role"]: s for s in sessions if s.get("_role")}
    confusable_ids = {
        sp["speaker_id"] for sp in speakers if sp["full_name"] in ("Jon Mokoena", "John Mokoena")
    }

    going_counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        if s["status"] == "Cancelled":
            continue
        for sid in s["speaker_ids"].split(";"):
            going_counts[sid] += 1
    max_count = max(going_counts.values())
    top_speaker_id = next(sid for sid, c in going_counts.items() if c == max_count)

    edge_case_speakers = {
        "confusable_pair": confusable_ids,
        "forced_accessible": set(by_role["forced_accessible"]["speaker_ids"].split(";")),
        "cospeaker": set(by_role["cospeaker"]["speaker_ids"].split(";")),
        "cancelled": set(by_role["cancelled"]["speaker_ids"].split(";")),
        "moved": set(by_role["moved"]["speaker_ids"].split(";")),
        "top_speaker": {top_speaker_id},
    }

    for a, b in combinations(edge_case_speakers, 2):
        overlap = edge_case_speakers[a] & edge_case_speakers[b]
        assert not overlap, f"edge cases {a!r} and {b!r} share speaker(s) {overlap}"


def test_manifest_hashes_match_files():
    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kb_version"] == generate_kb.KB_VERSION
    assert manifest["generator_seed"] == generate_kb.SEED

    file_paths = {
        "sessions.csv": KB_DIR / "sessions.csv",
        "speakers.json": KB_DIR / "speakers.json",
        "venue_guide.pdf": KB_DIR / "venue_guide.pdf",
        "kb_gold.jsonl": EVALS_DIR / "kb_gold.jsonl",
        "kb_heldout.jsonl": EVALS_DIR / "kb_heldout.jsonl",
    }
    for name, path in file_paths.items():
        expected = manifest["files"][name]["sha256"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{name} does not match manifest.json - bump kb_version and regenerate"
        )


def test_heldout_set_shape_and_disjointness_from_gold(gold, heldout):
    assert len(heldout) == 15
    counts = Counter(item["answer_type"] for item in heldout)
    assert counts == {
        "lookup": 3,
        "aggregation": 3,
        "cross_doc": 3,
        "policy": 2,
        "semantic": 2,
        "negative": 2,
    }

    gold_questions = {item["question"] for item in gold}
    heldout_questions = {item["question"] for item in heldout}
    assert gold_questions.isdisjoint(heldout_questions), "held-out item repeats a gold question"

    def is_row_ref(ref: str) -> bool:
        # Row-specific refs must not repeat between gold and held-out.
        # Bare filenames ("sessions.csv") and PDF table-level refs
        # ("venue_guide.pdf#policies", no row key) are whole-table
        # references, not one specific fact, so they're expected to appear
        # in both sets.
        if "#" not in ref:
            return False
        file_name, anchor = ref.split("#", 1)
        if file_name == "venue_guide.pdf":
            return ":" in anchor
        return True

    gold_refs = {ref for item in gold for ref in item["source_refs"] if is_row_ref(ref)}
    heldout_refs = {ref for item in heldout for ref in item["source_refs"] if is_row_ref(ref)}
    assert gold_refs.isdisjoint(heldout_refs), "held-out item reuses a gold row-level source_ref"

    assert all(item["id"].startswith("H-") for item in heldout)

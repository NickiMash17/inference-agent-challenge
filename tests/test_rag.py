"""Tests for the ingestion, SQL store and indexing layers (rag/)."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from rag.index import NumpyVectorIndex, tokenize
from rag.ingest import load_all, parse_venue_pdf
from rag.models import Chunk
from rag.sql_store import get_connection

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return load_all(KB_DIR)


@pytest.fixture(scope="module")
def sessions_rows() -> list[dict]:
    with (KB_DIR / "sessions.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def speakers_rows() -> list[dict]:
    return json.loads((KB_DIR / "speakers.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def venue_tables() -> dict[str, list[dict]]:
    return parse_venue_pdf(KB_DIR / "venue_guide.pdf")


def test_chunk_counts_match_source_rows_plus_summaries(
    chunks, sessions_rows, speakers_rows, venue_tables
):
    counts = Counter(c.doc_type for c in chunks)
    assert counts["session"] == len(sessions_rows)
    assert counts["speaker"] == len(speakers_rows)
    assert counts["room"] == len(venue_tables["rooms"]) + 1  # +1 summary chunk
    assert counts["ticket"] == len(venue_tables["tickets"]) + 1
    assert counts["policy"] == len(venue_tables["policies"]) + 1
    assert len(chunks) == sum(counts.values())


def test_every_chunk_has_a_valid_source_ref(chunks, sessions_rows, speakers_rows, venue_tables):
    session_ids = {row["session_id"] for row in sessions_rows}
    speaker_ids = {row["speaker_id"] for row in speakers_rows}
    room_ids = {row["room_id"] for row in venue_tables["rooms"]}
    tiers = {row["tier"] for row in venue_tables["tickets"]}
    policies = {row["policy"] for row in venue_tables["policies"]}

    ref_re = re.compile(r"^(sessions\.csv|speakers\.json|venue_guide\.pdf)#(.+)$")

    for c in chunks:
        match = ref_re.match(c.source_ref)
        assert match, f"{c.chunk_id}: malformed source_ref {c.source_ref!r}"
        file_name, anchor = match.groups()

        if file_name == "sessions.csv":
            assert anchor in session_ids
        elif file_name == "speakers.json":
            assert anchor in speaker_ids
        else:
            table, _, row_key = anchor.partition(":")
            assert table in {"rooms", "tickets", "policies"}
            if not row_key:
                continue  # table-level summary ref
            if table == "rooms":
                assert row_key in room_ids
            elif table == "tickets":
                assert row_key in tiers
            else:
                assert row_key in policies


def test_no_chunk_text_contains_email_or_phone(chunks, speakers_rows):
    emails = {sp["email"] for sp in speakers_rows}
    phones = {sp["phone"] for sp in speakers_rows}
    for c in chunks:
        assert "@" not in c.text, f"{c.chunk_id} text contains an '@' (possible email leak)"
        for phone in phones:
            assert phone not in c.text, f"{c.chunk_id} text contains phone number {phone!r}"
        for email in emails:
            assert email not in c.text, f"{c.chunk_id} text contains email {email!r}"


def test_sessions_active_excludes_exactly_the_cancelled_sessions(sessions_rows):
    con = get_connection(KB_DIR)
    rows = con.execute("SELECT session_id FROM sessions_active").fetchall()
    active_ids = {row[0] for row in rows}
    expected_active = {
        row["session_id"] for row in sessions_rows if row["status"] != "Cancelled"
    }
    expected_cancelled = {
        row["session_id"] for row in sessions_rows if row["status"] == "Cancelled"
    }

    assert active_ids == expected_active
    assert active_ids.isdisjoint(expected_cancelled)
    assert len(expected_cancelled) >= 1


def test_vector_cache_invalidates_on_text_change(tmp_path):
    chunk_v1 = Chunk(
        chunk_id="x", text="hello world", source_file="f", source_ref="f#x",
        doc_type="session", metadata={},
    )
    index_v1 = NumpyVectorIndex(cache_dir=tmp_path)
    index_v1.add([chunk_v1])
    key_v1 = index_v1._cache_key([chunk_v1])
    assert (tmp_path / f"{key_v1}.npz").exists()

    chunk_v2 = chunk_v1.model_copy(update={"text": "a completely different sentence"})
    index_v2 = NumpyVectorIndex(cache_dir=tmp_path)
    index_v2.add([chunk_v2])
    key_v2 = index_v2._cache_key([chunk_v2])

    assert key_v1 != key_v2, "cache key did not change when chunk text changed"
    assert (tmp_path / f"{key_v2}.npz").exists()

    # same chunks again -> same key -> reuses the cached embeddings (no recompute)
    index_v1_again = NumpyVectorIndex(cache_dir=tmp_path)
    index_v1_again.add([chunk_v1])
    assert (index_v1_again.embeddings == index_v1.embeddings).all()


def test_ids_tokenize_cleanly_despite_surrounding_punctuation():
    """Regression test for the G-004 diagnosis: IDs must stay a single
    token and never absorb or get split by surrounding punctuation, while
    still keeping their internal hyphen."""
    assert tokenize("Session S-015: has been cancelled.") == [
        "session", "s-015", "has", "been", "cancelled",
    ]
    assert tokenize("(SP-001) presented") == ["sp-001", "presented"]
    assert tokenize("in room R-04.") == ["in", "room", "r-04"]
    assert tokenize("S-015, S-016, and S-017") == ["s-015", "s-016", "and", "s-017"]

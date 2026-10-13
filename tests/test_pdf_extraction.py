"""Tests that venue_guide.pdf's tables extract cleanly with pdfplumber.

If these fail because of merged cells or split/wrapped headers, the fix is
to adjust table styling/column widths in scripts/generate_kb.py, not to
loosen these assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_kb  # noqa: E402

PDF_PATH = ROOT / "data" / "kb" / "venue_guide.pdf"

ROOMS_HEADER = [
    "Room ID",
    "Name",
    "Floor",
    "Capacity",
    "Wheelchair Accessible",
    "AV Equipment",
    "Hearing Loop",
]
TICKETS_HEADER = ["Tier", "Price (ZAR)", "Includes", "Refund Policy"]
POLICIES_HEADER = ["Policy", "Rule"]


@pytest.fixture(scope="module")
def tables() -> dict[str, list[list[str]]]:
    extracted: dict[str, list[list[str]]] = {}
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                header = table[0]
                if header == ROOMS_HEADER:
                    extracted["rooms"] = table
                elif header == TICKETS_HEADER:
                    extracted["tickets"] = table
                elif header == POLICIES_HEADER:
                    extracted["policies"] = table
    return extracted


def test_all_three_tables_are_found(tables):
    assert set(tables) == {"rooms", "tickets", "policies"}


def test_rooms_table_headers_and_row_count(tables):
    table = tables["rooms"]
    assert table[0] == ROOMS_HEADER
    assert len(table) - 1 == len(generate_kb.ROOMS)


def test_tickets_table_headers_and_row_count(tables):
    table = tables["tickets"]
    assert table[0] == TICKETS_HEADER
    assert len(table) - 1 == len(generate_kb.TICKET_TIERS)


def test_policies_table_headers_and_row_count(tables):
    table = tables["policies"]
    assert table[0] == POLICIES_HEADER
    assert len(table) - 1 == len(generate_kb.KEY_POLICIES)


def test_rooms_table_room_ids_match_source(tables):
    extracted_ids = [row[0] for row in tables["rooms"][1:]]
    assert extracted_ids == [r["room_id"] for r in generate_kb.ROOMS]


def test_policies_table_names_match_source(tables):
    extracted_policies = [row[0] for row in tables["policies"][1:]]
    assert extracted_policies == [p["policy"] for p in generate_kb.KEY_POLICIES]

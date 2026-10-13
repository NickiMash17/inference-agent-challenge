"""Load each Part 1 knowledge-base source into retrievable Chunks.

Source refs follow the same "file#anchor" convention as evals/kb_gold.jsonl:
  - "sessions.csv#S-009"               (row-level, matches gold exactly)
  - "speakers.json#SP-001"             (row-level, matches gold exactly)
  - "venue_guide.pdf#rooms"            (table-level summary chunk)
  - "venue_guide.pdf#rooms:R-04"       (row-level, refines the table-level ref)
A gold item's table-level ref ("...#rooms") is satisfied by either the
summary chunk (exact match) or any row-level chunk under that table (prefix
match) — see evals/retrieval_eval.py's `ref_matches`.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pdfplumber

from rag.models import Chunk

#: Mirrors scripts/generate_kb.py's CONFERENCE_DATES (the KB is frozen, so
#: this small constant is safe to duplicate rather than importing a
#: one-off data generator script into the runtime RAG package).
CONFERENCE_DATES = {"1": "2026-11-18", "2": "2026-11-19"}

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


def _normalize_header(header: str) -> str:
    key = re.sub(r"[^\w]+", "_", header.strip().lower())
    return re.sub(r"_+", "_", key).strip("_")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def _clean_cell(value: str) -> str:
    """pdfplumber preserves the wrapped line breaks inside a table cell;
    collapse them back into normal inter-word spacing."""
    return re.sub(r"\s+", " ", value).strip()


def parse_venue_pdf(pdf_path: Path) -> dict[str, list[dict]]:
    """Extracts the Rooms, Tickets and Policies tables, keyed by normalized,
    snake_case column names (e.g. "Room ID" -> "room_id")."""
    found: dict[str, list[dict]] = {}
    headers_by_table = {
        "rooms": ROOMS_HEADER,
        "tickets": TICKETS_HEADER,
        "policies": POLICIES_HEADER,
    }
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                header, *rows = table
                for name, expected_header in headers_by_table.items():
                    if header == expected_header:
                        keys = [_normalize_header(h) for h in header]
                        found[name] = [
                            dict(zip(keys, (_clean_cell(v) for v in row), strict=True))
                            for row in rows
                        ]
    missing = set(headers_by_table) - set(found)
    if missing:
        raise ValueError(f"venue_guide.pdf is missing expected table(s): {sorted(missing)}")
    return found


def load_sessions(
    sessions_rows: list[dict], speakers_by_id: dict[str, dict], rooms_by_id: dict[str, dict]
) -> list[Chunk]:
    chunks = []
    for row in sessions_rows:
        session_id = row["session_id"]
        speaker_ids = row["speaker_ids"].split(";")
        speaker_descs = [
            f"{speakers_by_id[sid]['full_name']} ({sid}) of {speakers_by_id[sid]['organisation']}"
            for sid in speaker_ids
        ]
        presenters = " and ".join(speaker_descs)
        room = rooms_by_id[row["room_id"]]
        date = CONFERENCE_DATES[row["day"]]

        article = "an" if row["level"][0] in "AEIOU" else "a"
        parts = []
        if row["status"] == "Cancelled":
            parts.append("This session has been cancelled.")
        parts.append(
            f'Session {session_id}: "{row["title"]}" is {article} {row["level"]}-level '
            f'{row["track"]} track session on Day {row["day"]} ({date}), '
            f'{row["start_time"]}-{row["end_time"]}, '
            f"in {room['name']} (room {row['room_id']}), with a session capacity of "
            f"{row['capacity']}."
        )
        if row["status"] == "Moved" and row["notes"]:
            parts.append(row["notes"])
        parts.append(f"Presented by {presenters}.")
        parts.append(f"Status: {row['status']}.")
        text = " ".join(parts)

        chunks.append(
            Chunk(
                chunk_id=session_id,
                text=text,
                source_file="sessions.csv",
                source_ref=f"sessions.csv#{session_id}",
                doc_type="session",
                metadata={
                    "session_id": session_id,
                    "track": row["track"],
                    "level": row["level"],
                    "day": row["day"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "room_id": row["room_id"],
                    "status": row["status"],
                    "speaker_ids": speaker_ids,
                },
            )
        )
    return chunks


def load_speakers(speakers_rows: list[dict], sessions_by_id: dict[str, dict]) -> list[Chunk]:
    chunks = []
    for sp in speakers_rows:
        speaker_id = sp["speaker_id"]
        session_descs = [
            f'"{sessions_by_id[sid]["title"]}" ({sid})' for sid in sp["session_ids"]
        ]
        sessions_part = (
            f"Presenting: {', '.join(session_descs)}." if session_descs else "Not presenting."
        )
        # Email and phone are intentionally excluded from chunk text; they
        # stay in metadata only, since contact details aren't needed for Q&A.
        text = (
            f"Speaker {speaker_id}: {sp['full_name']} is a {sp['job_title']} at "
            f"{sp['organisation']}, based in {sp['country']}. {sp['bio']} {sessions_part}"
        )
        chunks.append(
            Chunk(
                chunk_id=speaker_id,
                text=text,
                source_file="speakers.json",
                source_ref=f"speakers.json#{speaker_id}",
                doc_type="speaker",
                metadata={
                    "speaker_id": speaker_id,
                    "organisation": sp["organisation"],
                    "country": sp["country"],
                    "email": sp["email"],
                    "phone": sp["phone"],
                    "session_ids": sp["session_ids"],
                },
            )
        )
    return chunks


#: Table name (plural, matches the PDF section / gold source_ref) -> doc_type
#: (singular, per rag.models.DocType).
_TABLE_DOC_TYPE = {"rooms": "room", "tickets": "ticket", "policies": "policy"}


def _venue_row_chunks(
    table_name: str, rows: list[dict], key_field: str, row_text_fn, summary_text_fn
) -> list[Chunk]:
    doc_type = _TABLE_DOC_TYPE[table_name]
    chunks = []
    keys = []
    for row in rows:
        key = row[key_field]
        keys.append(key)
        chunks.append(
            Chunk(
                chunk_id=f"{table_name}:{_slugify(key)}",
                text=row_text_fn(row),
                source_file="venue_guide.pdf",
                source_ref=f"venue_guide.pdf#{table_name}:{key}",
                doc_type=doc_type,
                metadata=dict(row),
            )
        )
    chunks.append(
        Chunk(
            chunk_id=f"{table_name}:summary",
            text=summary_text_fn(keys),
            source_file="venue_guide.pdf",
            source_ref=f"venue_guide.pdf#{table_name}",
            doc_type=doc_type,
            metadata={"keys": keys},
        )
    )
    return chunks


def load_venue_chunks(tables: dict[str, list[dict]]) -> list[Chunk]:
    chunks = []

    chunks += _venue_row_chunks(
        "rooms",
        tables["rooms"],
        key_field="room_id",
        row_text_fn=lambda r: (
            f"Venue room {r['room_id']}: {r['name']}, floor {r['floor']}, capacity "
            f"{r['capacity']}. Wheelchair accessible: {r['wheelchair_accessible']}. "
            f"AV equipment: {r['av_equipment']}. Hearing loop: {r['hearing_loop']}."
        ),
        summary_text_fn=lambda keys: (
            f"Venue rooms: {', '.join(keys)}. See each room's own entry for capacity, "
            "wheelchair accessibility, AV equipment and hearing loop availability."
        ),
    )

    chunks += _venue_row_chunks(
        "tickets",
        tables["tickets"],
        key_field="tier",
        row_text_fn=lambda r: (
            f"Ticket tier {r['tier']}: ZAR {r['price_zar']}. Includes: {r['includes']}. "
            f"Refund policy: {r['refund_policy']}"
        ),
        summary_text_fn=lambda keys: (
            f"Ticket tiers: {', '.join(keys)}. See each tier's own entry for price, "
            "inclusions and refund policy."
        ),
    )

    chunks += _venue_row_chunks(
        "policies",
        tables["policies"],
        key_field="policy",
        row_text_fn=lambda r: f"Venue policy - {r['policy']}: {r['rule']}",
        summary_text_fn=lambda keys: (
            f"Venue policies: {', '.join(keys)}. See each policy's own entry for the full rule."
        ),
    )

    return chunks


def load_all(kb_dir: Path) -> list[Chunk]:
    with (kb_dir / "sessions.csv").open(encoding="utf-8", newline="") as f:
        sessions_rows = list(csv.DictReader(f))
    speakers_rows = json.loads((kb_dir / "speakers.json").read_text(encoding="utf-8"))
    tables = parse_venue_pdf(kb_dir / "venue_guide.pdf")

    speakers_by_id = {sp["speaker_id"]: sp for sp in speakers_rows}
    sessions_by_id = {row["session_id"]: row for row in sessions_rows}
    rooms_by_id = {row["room_id"]: row for row in tables["rooms"]}

    chunks: list[Chunk] = []
    chunks += load_sessions(sessions_rows, speakers_by_id, rooms_by_id)
    chunks += load_speakers(speakers_rows, sessions_by_id)
    chunks += load_venue_chunks(tables)
    return chunks

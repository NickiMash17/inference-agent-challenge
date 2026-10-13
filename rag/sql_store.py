"""In-memory DuckDB store over the Part 1 knowledge base, for aggregation
and lookup queries that the CSV/JSON/PDF chunks aren't well suited for."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from rag.ingest import parse_venue_pdf


def _sessions_speakers_bridge(sessions_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"session_id": session_id, "speaker_id": speaker_id}
        for session_id, speaker_ids in zip(
            sessions_df["session_id"], sessions_df["speaker_ids"], strict=True
        )
        for speaker_id in str(speaker_ids).split(";")
    ]
    return pd.DataFrame(rows, columns=["session_id", "speaker_id"])


def get_connection(kb_dir: Path) -> duckdb.DuckDBPyConnection:
    """Builds a fresh in-memory DuckDB database from the KB files.

    Tables: sessions, speakers, session_speakers (bridge), rooms, tickets,
    policies. View: sessions_active (sessions excluding status='Cancelled'),
    which encodes the "going ahead" counting convention from the gold set
    directly in the data layer.
    """
    con = duckdb.connect(":memory:")

    def _load(table_name: str, df: pd.DataFrame) -> None:
        tmp = f"_{table_name}_df"
        con.register(tmp, df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {tmp}")
        con.unregister(tmp)

    sessions_df = pd.read_csv(kb_dir / "sessions.csv", dtype={"day": "Int64"})
    _load("sessions", sessions_df)

    speakers_raw = json.loads((kb_dir / "speakers.json").read_text(encoding="utf-8"))
    speakers_df = pd.DataFrame(speakers_raw).drop(columns=["session_ids"])
    _load("speakers", speakers_df)

    _load("session_speakers", _sessions_speakers_bridge(sessions_df))

    tables = parse_venue_pdf(kb_dir / "venue_guide.pdf")
    for name in ("rooms", "tickets", "policies"):
        df = pd.DataFrame(tables[name])
        if name == "rooms":
            df["floor"] = df["floor"].astype(int)
            df["capacity"] = df["capacity"].astype(int)
        _load(name, df)

    con.execute("CREATE VIEW sessions_active AS SELECT * FROM sessions WHERE status != 'Cancelled'")

    return con


def describe_schema(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[str, str]]]:
    """Returns {table_or_view_name: [(column_name, column_type), ...]} for
    every table and view in the database, for later text-to-SQL prompting."""
    names = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' "
        "ORDER BY table_name"
    ).fetchall()
    schema = {}
    for (name,) in names:
        columns = con.execute(f"PRAGMA table_info('{name}')").fetchall()
        schema[name] = [(col[1], col[2]) for col in columns]
    return schema

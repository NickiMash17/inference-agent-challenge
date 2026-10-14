"""Tests for rag/sql_agent.py: the pre-execution sqlglot guard, the
engine-level DuckDB lockdown, and the generate-execute-retry flow. No
network - uses FakeLLMClient throughout."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from rag.sql_agent import (
    ALLOWED_TABLES,
    GeneratedSQL,
    SQLGuardError,
    generate_and_run_sql,
    guard_sql,
)
from rag.sql_store import get_connection
from shared.llm import FakeLLMClient, LLMResult, LLMUsage

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"


def _llm_result(parsed) -> LLMResult:
    return LLMResult(parsed=parsed, usage=LLMUsage(10, 5, 0.01, 0.0), model="fake")


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return get_connection(KB_DIR)


# --- the sqlglot guard -------------------------------------------------------


def test_guard_accepts_a_normal_select():
    guard_sql("SELECT COUNT(*) FROM sessions_active WHERE day = 1")


def test_guard_accepts_a_select_with_a_cte():
    guard_sql(
        "WITH by_track AS (SELECT track, COUNT(*) AS n FROM sessions_active GROUP BY track) "
        "SELECT * FROM by_track ORDER BY n DESC"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sessions",
        "DELETE FROM sessions WHERE session_id = 'S-001'",
        "UPDATE sessions SET title = 'x'",
        "INSERT INTO sessions VALUES (1)",
    ],
)
def test_guard_rejects_ddl_and_dml(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)


def test_guard_rejects_multiple_statements():
    with pytest.raises(SQLGuardError, match="exactly one statement"):
        guard_sql("SELECT 1; DROP TABLE sessions")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('x.csv')",
        "SELECT * FROM read_csv_auto('x.csv')",
        "SELECT * FROM read_parquet('x.parquet')",
        "SELECT * FROM read_json('x.json')",
        "SELECT glob('*.csv')",
    ],
)
def test_guard_rejects_external_data_functions(sql):
    with pytest.raises(SQLGuardError, match="blocked function"):
        guard_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH 'evil.db' AS e",
        "COPY sessions TO 'out.csv'",
        "INSTALL httpfs",
        "PRAGMA database_list",
        "SET enable_external_access = true",
    ],
)
def test_guard_rejects_non_select_commands(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)


def test_guard_rejects_unknown_tables():
    with pytest.raises(SQLGuardError, match="allow-list"):
        guard_sql("SELECT * FROM secret_table")


def test_guard_allowed_tables_matches_sql_store_tables():
    # Sanity check that the allow-list hasn't drifted from what
    # rag/sql_store.py actually creates.
    assert ALLOWED_TABLES == {
        "sessions",
        "sessions_active",
        "speakers",
        "session_speakers",
        "rooms",
        "tickets",
        "policies",
    }


# --- engine-level defence ----------------------------------------------------


def test_engine_level_lockdown_blocks_read_csv_even_if_the_guard_is_bypassed(con):
    """Proves the defence works at the DuckDB connection level, independent
    of the sqlglot guard - i.e. even if guard_sql() were buggy or skipped,
    this connection still can't read an external file."""
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT * FROM read_csv_auto('{(KB_DIR / 'sessions.csv').as_posix()}')")


def test_engine_level_lockdown_cannot_be_reversed(con):
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")


# --- generate -> guard -> execute --------------------------------------------


def test_generate_and_run_sql_happy_path(con):
    schema = {"sessions_active": [("session_id", "VARCHAR")]}
    fake = FakeLLMClient(
        scripted=[_llm_result(GeneratedSQL(sql="SELECT COUNT(*) AS n FROM sessions_active"))]
    )
    result = generate_and_run_sql("how many sessions?", con, fake, "model", schema)

    assert result.error is None
    assert result.retried is False
    assert result.columns == ["n"]
    assert len(result.rows) == 1


def test_generate_and_run_sql_rejects_guard_failure_and_retries(con):
    schema = {"sessions_active": [("session_id", "VARCHAR")]}
    fake = FakeLLMClient(
        scripted=[
            _llm_result(GeneratedSQL(sql="SELECT * FROM secret_table")),  # fails the guard
            _llm_result(GeneratedSQL(sql="SELECT COUNT(*) AS n FROM sessions_active")),  # retry
        ]
    )
    result = generate_and_run_sql("how many sessions?", con, fake, "model", schema)

    assert result.retried is True
    assert result.error is None
    assert result.columns == ["n"]


def test_generate_and_run_sql_gives_up_gracefully_after_one_failed_retry(con):
    schema = {"sessions_active": [("session_id", "VARCHAR")]}
    fake = FakeLLMClient(
        scripted=[
            _llm_result(GeneratedSQL(sql="SELECT * FROM secret_table")),
            _llm_result(GeneratedSQL(sql="SELECT * FROM still_bad_table")),
        ]
    )
    result = generate_and_run_sql("how many sessions?", con, fake, "model", schema)

    assert result.retried is True
    assert result.error is not None
    assert result.rows == []

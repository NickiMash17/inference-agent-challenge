"""Guarded text-to-SQL: ANSWER_MODEL generates SQL from the live schema,
which is validated by a sqlglot-based guard BEFORE execution (allow-listed
tables, SELECT-only, no external-data functions), then run with a row cap
and a wall-clock timeout against a DuckDB connection that already has
external access disabled and locked at the engine level (see
rag/sql_store.py's get_connection())."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp

from shared.llm import LLMClient, LLMError

ALLOWED_TABLES = {
    "sessions",
    "sessions_active",
    "speakers",
    "session_speakers",
    "rooms",
    "tickets",
    "policies",
}

#: Table/scalar functions that read external data or perform I/O. Checked
#: anywhere in the query tree, not only in FROM-clause position, since
#: e.g. a SELECT-list call is just as capable of reading a file.
BLOCKED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_parquet_auto",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_text_auto",
    "glob",
    "httpfs",
}

ROW_CAP = 100
QUERY_TIMEOUT_S = 5.0


class SQLGuardError(Exception):
    """Raised when generated SQL fails the pre-execution guard. The SQL is
    never executed when this is raised."""


class GeneratedSQL(BaseModel):
    sql: str = Field(description="A single read-only DuckDB SELECT statement.")


SQL_SYSTEM_PROMPT_TEMPLATE = """You write DuckDB SQL for a conference knowledge base.

Schema:
{schema_description}

Conventions:
- Use `sessions_active` (which already excludes status='Cancelled') unless the \
question is explicitly about cancelled sessions, in which case use `sessions`.
- Round averages with ROUND(x, 1).
- Return only the columns needed to answer the question.
- Write exactly one SELECT statement (CTEs are fine). Never write DDL, DML, \
ATTACH, COPY, INSTALL, LOAD, PRAGMA, or SET, and never call a function that \
reads external files (read_csv, read_parquet, etc.) - these are rejected \
before execution regardless of whether you write them."""


def _describe_schema_text(schema: dict[str, list[tuple[str, str]]]) -> str:
    lines = []
    for table, columns in schema.items():
        cols = ", ".join(f"{name} {dtype}" for name, dtype in columns)
        lines.append(f"- {table}({cols})")
    return "\n".join(lines)


def guard_sql(sql: str) -> None:
    """Raises SQLGuardError if `sql` fails any check. Pure validation - this
    function never executes anything."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except Exception as exc:
        raise SQLGuardError(f"could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise SQLGuardError(f"expected exactly one statement, got {len(statements)}")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise SQLGuardError(f"statement must be a SELECT, got {type(stmt).__name__}")

    # Checked before the table allow-list: a table-valued function call
    # (e.g. read_csv_auto(...)) shows up as an unnamed/empty table reference
    # in the allow-list check, which is technically still rejected but with
    # a confusing message - checking functions first gives a clear reason.
    for func in stmt.find_all((exp.Func, exp.Anonymous)):
        if isinstance(func, exp.Anonymous):
            name = str(func.this).lower()
        else:
            name = (func.sql_name() or "").lower()
        if name in BLOCKED_FUNCTIONS:
            raise SQLGuardError(f"query calls a blocked function: {name}")

    cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    table_names = {t.name.lower() for t in stmt.find_all(exp.Table)} - cte_names
    disallowed_tables = table_names - ALLOWED_TABLES
    if disallowed_tables:
        raise SQLGuardError(
            f"query references table(s) not in the allow-list: {sorted(disallowed_tables)}"
        )

    # Defence in depth: httpfs is an extension name (loaded via INSTALL/LOAD,
    # already rejected above since those don't parse as a Select) rather
    # than a function - checked in the raw text too in case of some syntax
    # path the structural checks above don't cover.
    if "httpfs" in sql.lower():
        raise SQLGuardError("query references httpfs")


def _wrap_with_row_cap(sql: str, cap: int = ROW_CAP) -> str:
    return f"SELECT * FROM ({sql.rstrip(' ;')}) AS _capped LIMIT {cap}"


def _execute_with_timeout(
    con: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = QUERY_TIMEOUT_S
) -> tuple[list[tuple], list[str], str | None]:
    capped_sql = _wrap_with_row_cap(sql)
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        result = con.execute(capped_sql)
        rows = result.fetchall()
        columns = [d[0] for d in result.description]
        return rows, columns, None
    except duckdb.Error as exc:
        return [], [], str(exc)
    finally:
        timer.cancel()


@dataclass(frozen=True)
class SQLResult:
    sql: str
    rows: list[tuple]
    columns: list[str]
    error: str | None
    retried: bool


def generate_and_run_sql(
    query: str,
    con: duckdb.DuckDBPyConnection,
    llm: LLMClient,
    model: str,
    schema: dict[str, list[tuple[str, str]]],
) -> SQLResult:
    system = SQL_SYSTEM_PROMPT_TEMPLATE.format(schema_description=_describe_schema_text(schema))

    try:
        result = llm.generate(
            model=model,
            system=system,
            messages=[{"role": "user", "content": query}],
            output_model=GeneratedSQL,
            max_tokens=512,
        )
    except LLMError as exc:
        return SQLResult(
            sql="", rows=[], columns=[], error=f"SQL generation failed: {exc}", retried=False
        )

    sql = result.parsed.sql
    outcome = _guard_and_run(sql, con)
    if isinstance(outcome, str):
        return _retry_once(query, sql, outcome, con, llm, model, system)

    rows, columns = outcome
    return SQLResult(sql=sql, rows=rows, columns=columns, error=None, retried=False)


def _guard_and_run(
    sql: str, con: duckdb.DuckDBPyConnection
) -> tuple[list[tuple], list[str]] | str:
    """Returns (rows, columns) on success, or an error message string."""
    try:
        guard_sql(sql)
    except SQLGuardError as exc:
        return f"SQL guard rejected query: {exc}"
    rows, columns, exec_error = _execute_with_timeout(con, sql)
    if exec_error is not None:
        return exec_error
    return rows, columns


def _retry_once(
    query: str,
    failed_sql: str,
    error: str,
    con: duckdb.DuckDBPyConnection,
    llm: LLMClient,
    model: str,
    system: str,
) -> SQLResult:
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"SQL: {failed_sql}"},
        {
            "role": "user",
            "content": (
                f"That query failed with this error:\n{error}\n\n"
                "Fix it and return corrected SQL."
            ),
        },
    ]
    try:
        retry_result = llm.generate(
            model=model,
            system=system,
            messages=messages,
            output_model=GeneratedSQL,
            max_tokens=512,
        )
    except LLMError as exc:
        return SQLResult(
            sql=failed_sql, rows=[], columns=[], error=f"{error}; retry failed: {exc}", retried=True
        )

    retry_sql = retry_result.parsed.sql
    outcome = _guard_and_run(retry_sql, con)
    if isinstance(outcome, str):
        return SQLResult(sql=retry_sql, rows=[], columns=[], error=outcome, retried=True)
    rows, columns = outcome
    return SQLResult(sql=retry_sql, rows=rows, columns=columns, error=None, retried=True)

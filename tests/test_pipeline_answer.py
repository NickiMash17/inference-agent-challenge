"""Tests for rag/pipeline.py's answer() end-to-end orchestration. No
network - uses FakeLLMClient throughout."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.answer import AnswerOut
from rag.ingest import load_all
from rag.pipeline import answer, build_answer_pipelines
from rag.router import RouteDecision
from rag.sql_agent import GeneratedSQL
from rag.sql_store import describe_schema, get_connection
from shared.llm import FakeLLMClient, LLMError, LLMResult, LLMUsage

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"


def _llm_result(parsed) -> LLMResult:
    return LLMResult(parsed=parsed, usage=LLMUsage(10, 5, 0.01, 0.0), model="fake")


@pytest.fixture(scope="module")
def chunks():
    return load_all(KB_DIR)


@pytest.fixture(scope="module")
def con():
    return get_connection(KB_DIR)


@pytest.fixture(scope="module")
def schema(con):
    return describe_schema(con)


@pytest.fixture(scope="module")
def pipelines(chunks, tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("index_cache")
    return build_answer_pipelines(chunks, cache_dir=cache_dir)


def test_answer_never_raises_when_every_llm_call_times_out(con, schema, pipelines):
    """The router falls back to "retrieve" on its own LLM failure, but the
    final answer-generation call also fails - the whole pipeline must still
    return a safe AnswerResult rather than raising."""
    fake = FakeLLMClient(respond=lambda **kw: LLMError("simulated timeout"))

    result = answer(
        "What is the weather like?",
        con=con,
        schema=schema,
        retrieval_pipelines=pipelines,
        llm=fake,
        router_model="router-model",
        answer_model="answer-model",
    )

    assert result.low_confidence is True
    assert result.answer  # some user-safe message, not an exception
    assert any("LLM error" in e or "timeout" in e for e in result.errors)


def test_answer_id_lookup_route_skips_router_llm_call(con, schema, pipelines):
    fake = FakeLLMClient(
        scripted=[_llm_result(AnswerOut(answer="Cancelled.", citations=[], not_in_kb=False))]
    )
    result = answer(
        "What is the status of session S-015?",
        con=con,
        schema=schema,
        retrieval_pipelines=pipelines,
        llm=fake,
        router_model="router-model",
        answer_model="answer-model",
    )

    assert result.route == "id_lookup"
    assert len(fake.calls) == 1  # only the final answer call, no router LLM call


def test_answer_sql_route_runs_guarded_sql_and_grounds_the_answer(con, schema, pipelines):
    fake = FakeLLMClient(
        scripted=[
            _llm_result(RouteDecision(route="sql", rationale="counting")),
            _llm_result(
                GeneratedSQL(sql="SELECT COUNT(*) AS n FROM sessions_active WHERE day = 1")
            ),
            _llm_result(AnswerOut(answer="19.", citations=[], not_in_kb=False)),
        ]
    )
    result = answer(
        "How many sessions are going ahead on Day 1?",
        con=con,
        schema=schema,
        retrieval_pipelines=pipelines,
        llm=fake,
        router_model="router-model",
        answer_model="answer-model",
    )

    assert result.route == "sql"
    assert result.sql_result is not None
    assert result.sql_result.error is None
    assert result.errors == []


def test_answer_records_cost_and_tokens_across_stages(con, schema, pipelines):
    fake = FakeLLMClient(
        scripted=[_llm_result(AnswerOut(answer="Cancelled.", citations=[], not_in_kb=False))]
    )
    result = answer(
        "What is the status of session S-015?",
        con=con,
        schema=schema,
        retrieval_pipelines=pipelines,
        llm=fake,
        router_model="router-model",
        answer_model="answer-model",
    )

    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert "retrieve" in result.stage_latencies_s
    assert "answer" in result.stage_latencies_s

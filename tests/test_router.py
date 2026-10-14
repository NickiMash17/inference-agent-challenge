"""Tests for rag/router.py. No network - uses FakeLLMClient throughout."""

from __future__ import annotations

from rag.router import RouteDecision, route
from shared.llm import FakeLLMClient, LLMError, LLMResult, LLMUsage


def _llm_result(parsed) -> LLMResult:
    return LLMResult(parsed=parsed, usage=LLMUsage(10, 5, 0.01, 0.0), model="fake")


def test_id_pattern_routes_without_any_llm_call():
    fake = FakeLLMClient()  # no scripted responses - any call raises AssertionError
    outcome = route("What is the status of session S-015?", fake, "model", ["sessions"])

    assert outcome.route == "id_lookup"
    assert outcome.fallback_reason is None
    assert outcome.llm_result is None
    assert fake.calls == [], "the deterministic ID pre-route must never call the LLM"


def test_id_pattern_matches_speaker_and_room_ids_too():
    fake = FakeLLMClient()
    for query in ("Who is SP-001?", "Is room R-04 accessible?"):
        outcome = route(query, fake, "model", ["sessions"])
        assert outcome.route == "id_lookup"
        assert fake.calls == []


def test_llm_classifies_a_non_id_query():
    fake = FakeLLMClient(
        scripted=[_llm_result(RouteDecision(route="sql", rationale="requires counting"))]
    )
    outcome = route("How many sessions are in the RAG track?", fake, "model", ["sessions_active"])

    assert outcome.route == "sql"
    assert outcome.rationale == "requires counting"
    assert outcome.fallback_reason is None
    assert len(fake.calls) == 1


def test_falls_back_to_retrieve_on_invalid_llm_output():
    # The LLM returned something that isn't a RouteDecision at all.
    fake = FakeLLMClient(
        respond=lambda **kw: LLMResult(
            parsed="not a RouteDecision", usage=LLMUsage(1, 1, 0.01, 0.0), model="fake"
        )
    )
    outcome = route("Tell me about session trends", fake, "model", ["sessions"])

    assert outcome.route == "retrieve"
    assert outcome.fallback_reason is not None
    assert "invalid output" in outcome.fallback_reason


def test_falls_back_to_retrieve_on_llm_error():
    fake = FakeLLMClient(scripted=[LLMError("connection timed out")])
    outcome = route("Tell me about session trends", fake, "model", ["sessions"])

    assert outcome.route == "retrieve"
    assert outcome.fallback_reason is not None
    assert "LLM error" in outcome.fallback_reason
    assert outcome.llm_result is None

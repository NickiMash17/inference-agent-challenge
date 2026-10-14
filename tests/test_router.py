"""Tests for rag/router.py. No network - uses FakeLLMClient throughout."""

from __future__ import annotations

import pytest

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


def test_id_with_aggregation_language_goes_to_the_llm_router_not_id_lookup():
    """An ID plus aggregation language (e.g. "how many sessions is SP-001
    presenting") is a GROUP BY/COUNT question that names an entity, not a
    simple lookup - it must not short-circuit to id_lookup."""
    fake = FakeLLMClient(
        scripted=[_llm_result(RouteDecision(route="sql", rationale="counting"))]
    )
    outcome = route(
        "How many sessions is SP-001 presenting?", fake, "model", ["sessions_active"]
    )

    assert outcome.route == "sql"
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "query",
    [
        "How many sessions is SP-001 presenting?",
        "What is the total number of sessions for S-015's track?",
        "Count the sessions in room R-01.",
        "What is the average capacity of rooms like R-04?",
        "Which has the most sessions, SP-001 or SP-002?",
        "What is the least popular room, R-01 or R-02?",
    ],
)
def test_various_id_plus_aggregation_phrasings_avoid_id_lookup(query):
    fake = FakeLLMClient(
        scripted=[_llm_result(RouteDecision(route="sql", rationale="counting"))]
    )
    outcome = route(query, fake, "model", ["sessions_active"])
    assert outcome.route != "id_lookup"
    assert len(fake.calls) == 1


def test_id_without_aggregation_language_still_short_circuits():
    fake = FakeLLMClient()  # any LLM call would raise AssertionError
    outcome = route("What room is session S-015 in?", fake, "model", ["sessions"])
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

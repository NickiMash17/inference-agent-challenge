"""Tests for evals/e2e_eval.py's pure scoring logic and CLI guards.

No network, and this never calls run_eval()/AnthropicClient - those would
require a real API key and make real (billed) calls. Only the
correctness/routing helper functions and the CLI's argument guards are
tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.e2e_eval import check_correctness, expected_routes, main  # noqa: E402
from rag.pipeline import AnswerResult  # noqa: E402


def _result(answer: str, not_in_kb: bool = False, route: str = "retrieve") -> AnswerResult:
    return AnswerResult(
        query="q",
        route=route,
        fallback_reason=None,
        rewrites=[],
        retrieved=[],
        sql_result=None,
        answer=answer,
        citations=[],
        not_in_kb=not_in_kb,
        low_confidence=False,
        invalid_citations=[],
        stage_latencies_s={},
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        errors=[],
    )


# --- check_correctness --------------------------------------------------------


def test_integer_expected_value_requires_exact_match_by_default():
    item = {"expected_value": 19, "answer_type": "aggregation"}
    assert check_correctness(item, _result("There are 19 sessions going ahead."))
    assert not check_correctness(item, _result("There are 20 sessions going ahead."))


def test_float_expected_value_uses_the_items_own_tolerance():
    item = {"expected_value": 115.3, "tolerance": 0.05, "answer_type": "aggregation"}
    assert check_correctness(item, _result("The average capacity is 115.32."))
    assert not check_correctness(item, _result("The average capacity is 116.0."))


def test_yes_no_expected_value_matches_as_a_word_not_a_substring():
    item = {"expected_value": "No", "answer_type": "cross_doc"}
    assert check_correctness(item, _result("No, it is not wheelchair accessible."))
    assert not check_correctness(item, _result("Yes, it is accessible."))


def test_string_expected_value_uses_normalized_containment():
    item = {"expected_value": "S-015", "answer_type": "lookup"}
    assert check_correctness(item, _result("The session in question is S-015."))
    assert not check_correctness(item, _result("The session in question is S-010."))


def test_not_in_kb_sentinel_requires_the_not_in_kb_flag():
    item = {"expected_value": "NOT_IN_KB", "answer_type": "negative"}
    assert check_correctness(item, _result("Not in the knowledge base.", not_in_kb=True))
    assert not check_correctness(item, _result("The wifi password is 'conference2026'."))


# --- expected_routes -----------------------------------------------------------


def test_expected_routes_lookup_with_id_in_question():
    item = {"answer_type": "lookup", "question": "What is the title of session S-001?"}
    assert expected_routes(item) == {"id_lookup"}


def test_expected_routes_aggregation_is_sql():
    item = {"answer_type": "aggregation", "question": "How many sessions on Day 1?"}
    assert expected_routes(item) == {"sql"}


def test_expected_routes_semantic_is_retrieve_family():
    item = {"answer_type": "semantic", "question": "Which talk covers X?"}
    assert expected_routes(item) == {"rewrite_retrieve", "retrieve"}


def test_expected_routes_policy_and_cross_doc():
    for answer_type in ("policy", "cross_doc"):
        item = {"answer_type": answer_type, "question": "irrelevant"}
        assert expected_routes(item) == {"id_lookup", "retrieve"}


def test_expected_routes_negative_is_unconstrained():
    item = {"answer_type": "negative", "question": "What is the wifi password?"}
    assert expected_routes(item) is None


# --- CLI guards (never reach run_eval / AnthropicClient) ----------------------


def test_reps_greater_than_one_requires_save_baseline(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["e2e_eval.py", "--reps", "3"])
    with pytest.raises(SystemExit):
        main()
    assert "--save-baseline" in capsys.readouterr().err


def test_missing_api_key_refuses_to_run(monkeypatch, capsys):
    monkeypatch.delenv("ATLAS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["e2e_eval.py"])
    with pytest.raises(SystemExit):
        main()
    assert "ATLAS_ANTHROPIC_API_KEY" in capsys.readouterr().err

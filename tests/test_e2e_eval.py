"""Tests for evals/e2e_eval.py's pure scoring logic and CLI guards.

No network. This never calls main(), run_eval(), or AnthropicClient - only
the pure correctness/routing/guard functions are tested here, so there is
no path from this file to a real, billed API call even if the safety nets
in tests/conftest.py weren't present.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.e2e_eval import (  # noqa: E402
    _extract_number,
    check_correctness,
    expected_routes,
    missing_api_key_error,
    reps_guard_error,
)
from rag.pipeline import AnswerResult  # noqa: E402
from shared.config import Settings  # noqa: E402


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


def test_number_extraction_ignores_ids_dates_and_times():
    """Regression test for the G-012 bug: a hyphen directly attached to a
    letter (as in an ID) is not a minus sign, and digits inside an ID, a
    date, or a time/time-range must never be mistaken for the answer."""
    assert _extract_number("SP-001 is presenting 2 sessions") == 2.0
    assert _extract_number("S-006 and S-030 are both correct, 2 in total") == 2.0
    assert (
        _extract_number(
            "Session S-015, on 2026-11-18, 13:30-14:30, in R-01, has 2 co-presenters."
        )
        == 2.0
    )
    assert _extract_number("There are 19 sessions going ahead.") == 19.0
    assert _extract_number("The average capacity is 115.32.") == 115.32


def test_number_expected_value_scores_correctly_with_an_id_prefixed_answer():
    item = {"expected_value": 2, "answer_type": "aggregation"}
    answer = (
        "Speaker SP-001 (Jon Mokoena) is presenting 2 sessions that are going ahead: "
        'S-006 "AI-Assisted Customer Support at Scale" and S-030 "Data Verification '
        'Pipelines for Agent Outputs".'
    )
    assert check_correctness(item, _result(answer))


def test_float_expected_value_uses_the_items_own_tolerance():
    item = {"expected_value": 115.3, "tolerance": 0.05, "answer_type": "aggregation"}
    assert check_correctness(item, _result("The average capacity is 115.32."))
    assert not check_correctness(item, _result("The average capacity is 116.0."))


def test_yes_no_expected_value_matches_as_a_word_not_a_substring():
    item = {"expected_value": "No", "answer_type": "cross_doc"}
    assert check_correctness(item, _result("No, it is not wheelchair accessible."))
    assert not check_correctness(item, _result("Yes, it is accessible."))


def test_list_valued_expected_value_accepts_ids_or_resolved_names():
    item = {"expected_value": "SP-004;SP-019", "answer_type": "cross_doc"}
    names = {"SP-004": "Donald Johnson", "SP-019": "Joseph Torres"}

    # both as raw IDs
    assert check_correctness(
        item, _result("Co-speakers: SP-004 and SP-019."), entity_names=names
    )
    # both as resolved full names (the realistic LLM-prose case)
    assert check_correctness(
        item,
        _result(
            "The session is presented by Donald Johnson of BrightPath AI and "
            "Joseph Torres of Horizon Cognition."
        ),
        entity_names=names,
    )
    # one missing entirely -> incorrect
    assert not check_correctness(
        item, _result("The session is presented by Donald Johnson."), entity_names=names
    )


def test_list_valued_expected_value_without_entity_names_falls_back_to_id_only():
    item = {"expected_value": "SP-004;SP-019", "answer_type": "cross_doc"}
    assert check_correctness(item, _result("Co-speakers: SP-004 and SP-019."))
    assert not check_correctness(
        item, _result("Co-speakers: Donald Johnson and Joseph Torres.")
    )


def test_policy_items_score_on_must_contain_not_whole_sentence():
    item = {
        "expected_value": (
            "Badges must be worn and visible at all times inside the venue. Lost badges "
            "must be reported to the registration desk immediately and may incur a "
            "replacement fee of ZAR 150."
        ),
        "answer_type": "policy",
        "must_contain": ["worn and visible", "ZAR 150"],
    }
    # a correct paraphrase (reworded, reordered) must still score correct
    paraphrase = (
        "Lost badges must be reported to the registration desk immediately, and a "
        "replacement fee of ZAR 150 may be charged. (Badges must also be worn and "
        "visible at all times inside the venue.)"
    )
    assert check_correctness(item, _result(paraphrase))
    # missing one of the key facts -> incorrect
    assert not check_correctness(
        item, _result("Lost badges must be reported to the registration desk.")
    )


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


# --- CLI guards --------------------------------------------------------------
#
# These call the pure guard functions directly, never main(): main() is the
# thing that would actually construct a real AnthropicClient and start a
# live eval, so no test is allowed to invoke it at all (the conftest.py
# safety nets would also catch a slip here, but the point is to not depend
# on that - this guard logic is pure and should be tested as such).


def test_reps_guard_blocks_multi_rep_without_baseline():
    assert reps_guard_error(3, None) is not None
    assert "--save-baseline" in reps_guard_error(3, None)


def test_reps_guard_allows_multi_rep_with_baseline():
    assert reps_guard_error(3, "e2e_v1") is None


def test_reps_guard_allows_single_rep_without_baseline():
    assert reps_guard_error(1, None) is None


def test_missing_api_key_error_detects_absence():
    settings = Settings(_env_file=None, atlas_anthropic_api_key=None)
    error = missing_api_key_error(settings)
    assert error is not None
    assert "ATLAS_ANTHROPIC_API_KEY" in error


def test_missing_api_key_error_allows_a_present_key():
    settings = Settings(_env_file=None, atlas_anthropic_api_key="sk-fake-for-this-test")
    assert missing_api_key_error(settings) is None

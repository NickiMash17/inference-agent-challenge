"""Tests for evals/retrieval_eval.py's held-out-set guard.

These tests cover ONLY refusal paths. heldout_guard_error is a pure
function (no file I/O, no retrieval) so none of these tests ever execute,
or come close to executing, an actual held-out evaluation. Per CLAUDE.md,
no test in this repo may run an evaluation against evals/kb_heldout.jsonl -
it is evaluated once, by a human, at submission time.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.retrieval_eval import heldout_guard_error  # noqa: E402

KB_VERSION = "1.0.2"


def test_non_heldout_runs_are_never_blocked():
    assert heldout_guard_error(heldout=False, final=False, kb_version=KB_VERSION, env={}) is None
    # even with --final and a correct-looking env var, a non-heldout run is a no-op check
    assert (
        heldout_guard_error(
            heldout=False, final=True, kb_version=KB_VERSION,
            env={"HELDOUT_CONFIRM": KB_VERSION},
        )
        is None
    )


def test_heldout_without_final_is_refused():
    error = heldout_guard_error(heldout=True, final=False, kb_version=KB_VERSION, env={})
    assert error is not None
    assert "--final" in error


def test_heldout_and_final_without_env_var_is_refused():
    error = heldout_guard_error(heldout=True, final=True, kb_version=KB_VERSION, env={})
    assert error is not None
    assert "HELDOUT_CONFIRM" in error


def test_heldout_and_final_with_empty_env_var_is_refused():
    error = heldout_guard_error(
        heldout=True, final=True, kb_version=KB_VERSION, env={"HELDOUT_CONFIRM": ""}
    )
    assert error is not None
    assert "HELDOUT_CONFIRM" in error


def test_heldout_and_final_with_stale_kb_version_is_refused():
    """A HELDOUT_CONFIRM left over from a previous kb_version (e.g. set once
    in a shell and forgotten) must not silently authorize a run against the
    current one."""
    error = heldout_guard_error(
        heldout=True, final=True, kb_version=KB_VERSION, env={"HELDOUT_CONFIRM": "1.0.1"}
    )
    assert error is not None
    assert "HELDOUT_CONFIRM" in error


def test_heldout_and_final_with_wrong_case_or_whitespace_is_refused():
    for bad_value in (" 1.0.2", "1.0.2 ", "V1.0.2", "1.0.2\n"):
        error = heldout_guard_error(
            heldout=True, final=True, kb_version=KB_VERSION, env={"HELDOUT_CONFIRM": bad_value}
        )
        assert error is not None, f"{bad_value!r} should not satisfy the guard"

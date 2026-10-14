"""Tests for the Streamlit app (app/).

No network: most tests here only exercise module-level imports (proving the
sys.path setup works from a repo-root pytest run) and the pure helper
functions that back the question-length and session-count limits. The one
test that runs the full page (test_sidebar_question_counter_matches_number_
asked) uses Streamlit's AppTest with get_app_resources() and rag.pipeline.
answer() both mocked, so it never constructs a real AnthropicClient or
calls the Anthropic API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from rag.pipeline import AnswerResult

ROOT = Path(__file__).resolve().parent.parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_resources_imports_from_repo_root():
    module = _load_module(ROOT / "app" / "resources.py", "test_app_resources")
    assert module.ROOT == ROOT
    assert callable(module.get_app_resources)


def test_home_page_imports_without_errors():
    _load_module(ROOT / "app" / "Home.py", "test_app_home")


def test_part1_rag_page_imports_without_errors():
    _load_module(ROOT / "app" / "pages" / "1_Part1_RAG.py", "test_app_part1_rag")


def test_part1_rag_limits_match_the_spec():
    module = _load_module(ROOT / "app" / "pages" / "1_Part1_RAG.py", "test_app_part1_rag_limits")
    assert module.MAX_CHARS == 500
    assert module.MAX_QUESTIONS == 20


def test_is_over_char_limit():
    module = _load_module(ROOT / "app" / "pages" / "1_Part1_RAG.py", "test_app_char_limit")
    assert module.is_over_char_limit("a" * 500) is False
    assert module.is_over_char_limit("a" * 501) is True


def test_is_over_question_limit():
    module = _load_module(ROOT / "app" / "pages" / "1_Part1_RAG.py", "test_app_question_limit")
    assert module.is_over_question_limit(19) is False
    assert module.is_over_question_limit(20) is True
    assert module.is_over_question_limit(21) is True


def _fake_answer_result() -> AnswerResult:
    return AnswerResult(
        query="q",
        route="retrieve",
        fallback_reason=None,
        rewrites=[],
        retrieved=[],
        sql_result=None,
        answer="fake answer",
        citations=[],
        not_in_kb=False,
        low_confidence=False,
        invalid_citations=[],
        stage_latencies_s={"total": 0.01},
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0001,
        errors=[],
    )


def test_sidebar_question_counter_matches_number_asked():
    """Regression test: the sidebar used to show one less than the real
    count (e.g. "2/20" right after the 3rd question), because it read
    st.session_state.question_count before handle_question() incremented it
    later in the same script run. Asks 3 questions and checks the sidebar
    reads "3/20", not "2/20", straight after the 3rd."""
    fake_resources = {
        "settings": type("FakeSettings", (), {"router_model": "x", "answer_model": "y"})(),
        "chunks": [],
        "chunk_by_ref": {},
        "con": None,
        "schema": "",
        "pipelines": None,
        "llm": object(),
        "llm_error": None,
    }

    with (
        patch("app.resources.get_app_resources", return_value=fake_resources),
        patch("rag.pipeline.answer", return_value=_fake_answer_result()),
    ):
        at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Part1_RAG.py"))
        at.run()
        at.chat_input[0].set_value("question one").run()
        at.chat_input[0].set_value("question two").run()
        at.chat_input[0].set_value("question three").run()

    assert not at.exception
    captions = [c.value for c in at.sidebar.caption]
    assert "Questions asked this session: 3/20" in captions
    assert "Questions asked this session: 2/20" not in captions

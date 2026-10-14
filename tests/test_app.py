"""Tests for the Streamlit app (app/).

No network: nothing here calls get_app_resources() or main(), so the real
AnthropicClient is never constructed. These tests only exercise module-level
imports (proving the sys.path setup works from a repo-root pytest run) and
the pure helper functions that back the question-length and session-count
limits.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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

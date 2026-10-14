"""Shared, cached resources for the Streamlit app.

Streamlit runs each page as its own script, with that script's own
directory on sys.path - not the repo root. Without this, `import rag` /
`import shared` fails once deployed (Streamlit Community Cloud runs from a
fresh checkout where the repo root is never implicitly on sys.path). Every
app entry point (Home.py and each page) adds the repo root before importing
anything from this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from rag.ingest import load_all  # noqa: E402
from rag.models import Chunk  # noqa: E402
from rag.pipeline import build_answer_pipelines  # noqa: E402
from rag.sql_store import describe_schema, get_connection  # noqa: E402
from shared.config import Settings  # noqa: E402
from shared.llm import AnthropicClient, LLMError  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
EVALS_DIR = ROOT / "evals"
INDEX_CACHE_DIR = ROOT / "data" / "index"


@st.cache_resource(show_spinner="Loading knowledge base and models (first run only)...")
def get_app_resources() -> dict:
    """Builds every heavy resource once per app worker: the chunk corpus,
    the DuckDB store, and both retrieval pipelines (which is also where the
    embedding model downloads/loads - doing that here, at cache-build time,
    rather than lazily on first query, is what keeps the first *question*
    fast even on a cold Streamlit Cloud instance)."""
    settings = Settings()

    chunks = load_all(KB_DIR)
    chunk_by_ref: dict[str, Chunk] = {c.source_ref: c for c in chunks}

    con = get_connection(KB_DIR)
    schema = describe_schema(con)

    pipelines = build_answer_pipelines(chunks, cache_dir=INDEX_CACHE_DIR)

    llm = None
    llm_error: str | None = None
    if settings.atlas_anthropic_api_key:
        try:
            llm = AnthropicClient(settings)
        except LLMError as exc:
            llm_error = str(exc)
    else:
        llm_error = "ATLAS_ANTHROPIC_API_KEY is not set."

    return {
        "settings": settings,
        "chunks": chunks,
        "chunk_by_ref": chunk_by_ref,
        "con": con,
        "schema": schema,
        "pipelines": pipelines,
        "llm": llm,
        "llm_error": llm_error,
    }

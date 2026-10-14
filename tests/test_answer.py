"""Tests for rag/answer.py: context building and citation validation.
No network - uses FakeLLMClient throughout."""

from __future__ import annotations

from rag.answer import AnswerOut, answer_from_context, build_context_from_chunks
from rag.models import Chunk
from rag.pipeline import RetrievedItem
from rag.sql_agent import SQLResult
from shared.llm import FakeLLMClient, LLMResult, LLMUsage


def _llm_result(parsed) -> LLMResult:
    return LLMResult(parsed=parsed, usage=LLMUsage(10, 5, 0.01, 0.0), model="fake")


def _chunk(chunk_id: str, source_ref: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}.",
        source_file=source_ref.split("#")[0],
        source_ref=source_ref,
        doc_type="session",
        metadata={},
    )


def test_citation_validator_strips_invented_refs():
    items = [RetrievedItem(chunk=_chunk("S-015", "sessions.csv#S-015"), score=1.0, rank=1)]
    context, valid_refs = build_context_from_chunks(items)

    fake = FakeLLMClient(
        scripted=[
            _llm_result(
                AnswerOut(
                    answer="Cancelled.",
                    citations=["sessions.csv#S-015", "sessions.csv#S-999", "made.up#ref"],
                    not_in_kb=False,
                )
            )
        ]
    )
    result = answer_from_context("status?", context, valid_refs, fake, "model")

    assert result.citations == ["sessions.csv#S-015"]
    assert set(result.invalid_citations) == {"sessions.csv#S-999", "made.up#ref"}
    assert result.low_confidence is False  # a valid citation remains


def test_low_confidence_when_no_valid_citation_remains_and_not_in_kb_is_false():
    items = [RetrievedItem(chunk=_chunk("S-015", "sessions.csv#S-015"), score=1.0, rank=1)]
    context, valid_refs = build_context_from_chunks(items)

    fake = FakeLLMClient(
        scripted=[
            _llm_result(
                AnswerOut(answer="Something.", citations=["invented#ref"], not_in_kb=False)
            )
        ]
    )
    result = answer_from_context("status?", context, valid_refs, fake, "model")

    assert result.citations == []
    assert result.invalid_citations == ["invented#ref"]
    assert result.low_confidence is True


def test_not_low_confidence_when_not_in_kb_is_true_even_with_no_citations():
    fake = FakeLLMClient(
        scripted=[
            _llm_result(
                AnswerOut(answer="Not in the knowledge base.", citations=[], not_in_kb=True)
            )
        ]
    )
    result = answer_from_context("random trivia?", "(no context)", set(), fake, "model")

    assert result.not_in_kb is True
    assert result.low_confidence is False


def test_sql_context_block_uses_sql_hash_source_ref():
    sql_result = SQLResult(
        sql="SELECT COUNT(*) AS n FROM sessions_active",
        rows=[(19,)],
        columns=["n"],
        error=None,
        retried=False,
    )
    from rag.answer import build_context_from_sql

    context, valid_refs = build_context_from_sql(sql_result)

    assert len(valid_refs) == 1
    (ref,) = valid_refs
    assert ref.startswith("sql:")
    assert ref in context
    assert "19" in context

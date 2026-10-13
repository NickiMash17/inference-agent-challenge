"""Shared data model for retrieval chunks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DocType = Literal["session", "speaker", "room", "ticket", "policy"]


class Chunk(BaseModel):
    """A single retrievable unit of text, traceable back to its source row."""

    chunk_id: str
    text: str
    source_file: str
    #: Same format as evals/kb_gold.jsonl's source_refs, e.g. "sessions.csv#S-009"
    #: or "venue_guide.pdf#rooms:R-04" for a specific PDF table row.
    source_ref: str
    doc_type: DocType
    #: Structured fields kept for filtering and citation (not necessarily
    #: repeated verbatim in `text`).
    metadata: dict

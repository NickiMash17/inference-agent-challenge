"""Tests for the configurable ablation pipeline (rag/pipeline.py)."""

from __future__ import annotations

from rag.models import Chunk
from rag.pipeline import (
    CONFIGS,
    MultiRepVectorIndex,
    RetrievedItem,
    expand_with_foreign_keys,
    tokenize_no_stopwords,
    weights_for_query,
)


def _chunk(chunk_id: str, source_ref: str, doc_type: str, metadata: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="placeholder text",
        source_file=source_ref.split("#")[0],
        source_ref=source_ref,
        doc_type=doc_type,
        metadata=metadata or {},
    )


# --- v1: stopword tokenizer --------------------------------------------------


def test_stopword_tokenizer_keeps_ids_intact():
    text = "Which session is presented by SP-001 in room R-04 on day 1?"
    tokens = tokenize_no_stopwords(text)
    assert "sp-001" in tokens
    assert "r-04" in tokens


def test_stopword_tokenizer_removes_english_and_domain_stopwords():
    text = "Which session is presented by SP-001 in room R-04 on day 1?"
    tokens = tokenize_no_stopwords(text)
    for stopword in ("which", "is", "by", "in", "on", "session", "room", "day"):
        assert stopword not in tokens, f"{stopword!r} should have been removed"


# --- v2: weighted RRF ---------------------------------------------------------


def test_all_config_weights_sum_to_one():
    for config in CONFIGS.values():
        assert abs(sum(config.id_weights) - 1.0) < 1e-9
        assert abs(sum(config.text_weights) - 1.0) < 1e-9


def test_weights_switch_on_id_detection():
    config = CONFIGS["v2_weighted_rrf"]
    assert weights_for_query(config, "What is the status of session S-015?") == config.id_weights
    assert weights_for_query(config, "Who is speaker SP-001?") == config.id_weights
    assert weights_for_query(config, "Is room R-04 accessible?") == config.id_weights
    assert weights_for_query(config, "What is the catering policy?") == config.text_weights


# --- v4: FK expansion ---------------------------------------------------------


def test_fk_expansion_attaches_room_and_speakers_without_duplicates():
    session = _chunk(
        "S-001", "sessions.csv#S-001", "session",
        {"room_id": "R-04", "speaker_ids": ["SP-001", "SP-002"]},
    )
    room = _chunk("rooms:r-04", "venue_guide.pdf#rooms:R-04", "room")
    sp1 = _chunk("SP-001", "speakers.json#SP-001", "speaker")
    sp2 = _chunk("SP-002", "speakers.json#SP-002", "speaker")
    chunk_by_ref = {c.source_ref: c for c in (room, sp1, sp2)}

    base = [RetrievedItem(chunk=session, score=1.0, rank=1)]
    expanded = expand_with_foreign_keys(base, chunk_by_ref)

    assert len(expanded) == 4  # session + room + 2 speakers
    new_ids = {item.chunk.chunk_id for item in expanded[1:]}
    assert new_ids == {"rooms:r-04", "SP-001", "SP-002"}
    assert all(item.expanded and item.expanded_from == "S-001" for item in expanded[1:])
    # chunk identities are unique: no chunk appears twice
    all_ids = [item.chunk.chunk_id for item in expanded]
    assert len(all_ids) == len(set(all_ids))


def test_fk_expansion_never_invents_a_chunk_for_a_missing_fk():
    session = _chunk(
        "S-001", "sessions.csv#S-001", "session",
        {"room_id": "R-99", "speaker_ids": ["SP-001"]},  # R-99 doesn't exist
    )
    sp1 = _chunk("SP-001", "speakers.json#SP-001", "speaker")
    chunk_by_ref = {sp1.source_ref: sp1}

    base = [RetrievedItem(chunk=session, score=1.0, rank=1)]
    expanded = expand_with_foreign_keys(base, chunk_by_ref)

    assert len(expanded) == 2  # session + the one resolvable speaker, no invented room
    assert {item.chunk.chunk_id for item in expanded} == {"S-001", "SP-001"}


def test_fk_expansion_skips_a_speaker_already_in_the_base_results():
    session = _chunk(
        "S-001", "sessions.csv#S-001", "session", {"room_id": None, "speaker_ids": ["SP-001"]}
    )
    sp1 = _chunk("SP-001", "speakers.json#SP-001", "speaker")
    chunk_by_ref = {sp1.source_ref: sp1}

    base = [
        RetrievedItem(chunk=session, score=1.0, rank=1),
        RetrievedItem(chunk=sp1, score=0.5, rank=2),  # already retrieved directly
    ]
    expanded = expand_with_foreign_keys(base, chunk_by_ref)

    assert len(expanded) == 2  # nothing new appended
    all_ids = [item.chunk.chunk_id for item in expanded]
    assert len(all_ids) == len(set(all_ids))


# --- v3: multi-representation (title) vectors --------------------------------


def test_title_vectors_map_back_to_parent_chunk_without_duplicates(tmp_path):
    session_a = Chunk(
        chunk_id="S-001",
        text="Session S-001: a long description about enterprise retrieval systems and search.",
        source_file="sessions.csv",
        source_ref="sessions.csv#S-001",
        doc_type="session",
        metadata={"title": "Retrieval-Augmented Generation for Enterprise Search"},
    )
    session_b = Chunk(
        chunk_id="S-002",
        text="Session S-002: a long description about managing cloud infrastructure spending.",
        source_file="sessions.csv",
        source_ref="sessions.csv#S-002",
        doc_type="session",
        metadata={"title": "Cost Control for Production Inference"},
    )

    index = MultiRepVectorIndex(cache_dir=tmp_path)
    index.add([session_a, session_b])

    results = index.search("Retrieval-Augmented Generation for Enterprise Search", k=2)
    chunk_ids = [c.chunk_id for c, _score in results]

    assert chunk_ids[0] == "S-001"
    assert len(chunk_ids) == len(set(chunk_ids)), "a chunk must not be returned more than once"

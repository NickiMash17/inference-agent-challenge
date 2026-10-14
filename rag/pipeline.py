"""Configurable retrieval pipeline used for the ablation study in
evals/retrieval_eval.py. Each named RetrievalConfig in CONFIGS corresponds
to one row of evals/ABLATION.md; v1-v5 build cumulatively on v0_baseline."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.index import BM25Index, HybridRetriever, NumpyVectorIndex, _EmbeddingModelCache, tokenize
from rag.models import Chunk

# --- v1: stopword-aware tokenizer -------------------------------------------

#: A standard English stopword list (no external dependency), plus a few
#: domain-generic terms that show up in nearly every chunk in this corpus
#: and add noise rather than signal to BM25.
_ENGLISH_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours
    ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're they've
    this those through to too under until up very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's while
    who who's whom why why's with won't would wouldn't you you'd you'll
    you're you've your yours yourself yourselves
    """.split()
)
DOMAIN_STOPWORDS = frozenset({"session", "talk", "speaker", "which", "track", "room", "day"})
ALL_STOPWORDS = _ENGLISH_STOPWORDS | DOMAIN_STOPWORDS


def tokenize_no_stopwords(text: str) -> list[str]:
    return tokenize(text, stopwords=ALL_STOPWORDS)


# --- v2: query-aware weighted RRF -------------------------------------------

#: Matches our ID formats exactly: S-009, SP-001, R-04 (case-insensitive).
ID_PATTERN = re.compile(r"\b(?:S-\d{3}|SP-\d{3}|R-\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalConfig:
    """One named point in the ablation. Every tunable lives here, not as a
    scattered module-level constant, so a config is fully self-describing."""

    name: str
    remove_stopwords: bool = False
    weighted_rrf: bool = False
    rrf_k: int = 60
    #: (bm25_weight, vector_weight) used when the query contains an ID.
    id_weights: tuple[float, float] = (0.7, 0.3)
    #: (bm25_weight, vector_weight) used otherwise.
    text_weights: tuple[float, float] = (0.3, 0.7)
    title_vectors: bool = False
    fk_expansion: bool = False
    rerank: bool = False
    rerank_pool: int = 20
    id_pinning: bool = False
    #: Diagnostic-only: bypasses BM25/fusion entirely and searches the
    #: vector index alone (used for the vector-only ablation rows).
    vector_only: bool = False

    def __post_init__(self) -> None:
        for weights in (self.id_weights, self.text_weights):
            assert abs(sum(weights) - 1.0) < 1e-9, f"weights must sum to 1, got {weights}"


CONFIGS: dict[str, RetrievalConfig] = {
    "v0_baseline": RetrievalConfig(name="v0_baseline"),
    "v1_bm25_stopwords": RetrievalConfig(name="v1_bm25_stopwords", remove_stopwords=True),
    "v2_weighted_rrf": RetrievalConfig(
        name="v2_weighted_rrf", remove_stopwords=True, weighted_rrf=True
    ),
    "v3_title_vectors": RetrievalConfig(
        name="v3_title_vectors", remove_stopwords=True, weighted_rrf=True, title_vectors=True
    ),
    "v4_fk_expansion": RetrievalConfig(
        name="v4_fk_expansion",
        remove_stopwords=True,
        weighted_rrf=True,
        title_vectors=True,
        fk_expansion=True,
    ),
    "v5_rerank": RetrievalConfig(
        name="v5_rerank",
        remove_stopwords=True,
        weighted_rrf=True,
        title_vectors=True,
        fk_expansion=True,
        rerank=True,
    ),
    # On top of v4 (not v5 - id_pinning and the cross-encoder are two
    # independent answers to the same "exact ID" problem; v6 isolates the
    # cheaper, deterministic one).
    "v6_id_pinning": RetrievalConfig(
        name="v6_id_pinning",
        remove_stopwords=True,
        weighted_rrf=True,
        title_vectors=True,
        fk_expansion=True,
        id_pinning=True,
    ),
    # Diagnostic-only rows (not part of the v0->v6 progression): isolate
    # the vector index alone, to show where hybrid fusion helps or hurts
    # relative to vector search by itself.
    "v0_vector_only": RetrievalConfig(name="v0_vector_only", vector_only=True),
    "v3_vector_only": RetrievalConfig(
        name="v3_vector_only", title_vectors=True, vector_only=True
    ),
    # Used by the "retrieve" and "rewrite_retrieve" routes (rag/rewrite.py):
    # these routes are definitionally non-ID, semantic-leaning queries, so
    # fusion is vector-led (0.2 BM25 / 0.8 vector) rather than the 0.3/0.7
    # default. "id_lookup" uses v6_id_pinning instead, not this config.
    "v7_vector_led": RetrievalConfig(
        name="v7_vector_led",
        remove_stopwords=True,
        weighted_rrf=True,
        text_weights=(0.2, 0.8),
        title_vectors=True,
        fk_expansion=True,
    ),
}


def weights_for_query(config: RetrievalConfig, query: str) -> tuple[float, float]:
    return config.id_weights if ID_PATTERN.search(query) else config.text_weights


class WeightedHybridRetriever:
    """Reciprocal Rank Fusion with query-aware weights (see
    RetrievalConfig.id_weights / text_weights), instead of HybridRetriever's
    fixed 1:1 weighting."""

    def __init__(self, bm25: BM25Index, vector, config: RetrievalConfig) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.config = config

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        w_bm25, w_vector = weights_for_query(self.config, query)
        pool_size = len(self.bm25.chunks)
        bm25_results = self.bm25.search(query, k=pool_size)
        vector_results = self.vector.search(query, k=pool_size)

        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for weight, results in ((w_bm25, bm25_results), (w_vector, vector_results)):
            for rank, (chunk, _score) in enumerate(results, start=1):
                scores[chunk.chunk_id] = (
                    scores.get(chunk.chunk_id, 0.0) + weight / (self.config.rrf_k + rank)
                )
                chunk_by_id[chunk.chunk_id] = chunk

        ranked_ids = sorted(scores, key=lambda cid: -scores[cid])[:k]
        return [(chunk_by_id[cid], scores[cid]) for cid in ranked_ids]


# --- v6: exact-ID pinning -----------------------------------------------------

#: Same shapes as ID_PATTERN, but captured individually so each match's
#: exact ID text (and its order of appearance in the query) is available.
_ID_TOKEN_RE = re.compile(r"\b(S-\d{3}|SP-\d{3}|R-\d{2})\b", re.IGNORECASE)


def find_ids_in_order(query: str) -> list[str]:
    """IDs mentioned in the query, upper-cased, in first-appearance order,
    without duplicates (a repeated mention doesn't pin the same chunk twice)."""
    seen: list[str] = []
    for match in _ID_TOKEN_RE.finditer(query):
        id_str = match.group(1).upper()
        if id_str not in seen:
            seen.append(id_str)
    return seen


def id_to_source_ref(id_str: str) -> str | None:
    """Maps an ID string to the source_ref of the chunk it exactly
    identifies, or None if the ID doesn't match a known format."""
    if re.fullmatch(r"SP-\d{3}", id_str):
        return f"speakers.json#{id_str}"
    if re.fullmatch(r"S-\d{3}", id_str):
        return f"sessions.csv#{id_str}"
    if re.fullmatch(r"R-\d{2}", id_str):
        return f"venue_guide.pdf#rooms:{id_str}"
    return None


class IdPinningRetriever:
    """Wraps another retriever: any ID mentioned in the query (S-009,
    SP-001, R-04) is resolved by exact metadata lookup and pinned at the
    top of the results, in order of appearance, ahead of anything fusion
    would have ranked. Remaining slots are filled from the wrapped
    retriever's results, deduplicated against the pinned chunks. An ID that
    doesn't resolve to a real chunk (e.g. S-999) never crashes the search —
    it's simply recorded in `last_unresolved_ids` for the caller to inspect."""

    def __init__(self, inner, chunk_by_ref: dict[str, Chunk]) -> None:
        self.inner = inner
        self.chunk_by_ref = chunk_by_ref
        self.last_unresolved_ids: list[str] = []

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        pinned: list[Chunk] = []
        unresolved: list[str] = []
        seen_ids: set[str] = set()

        for id_str in find_ids_in_order(query):
            ref = id_to_source_ref(id_str)
            chunk = self.chunk_by_ref.get(ref) if ref else None
            if chunk is None:
                unresolved.append(id_str)
                continue
            pinned.append(chunk)
            seen_ids.add(chunk.chunk_id)

        self.last_unresolved_ids = unresolved

        results: list[tuple[Chunk, float]] = [(c, 1.0) for c in pinned]
        if len(results) < k:
            for chunk, score in self.inner.search(query, k=k + len(pinned)):
                if len(results) >= k:
                    break
                if chunk.chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk.chunk_id)
                results.append((chunk, score))

        return results[:k]


# --- v3: multi-representation vectors ---------------------------------------


def build_representations(chunk: Chunk) -> list[str]:
    """Extra short representations embedded alongside a chunk's full text,
    so a terse query (e.g. just a title) can still score a strong match.
    A chunk's final vector score is the max over all of its
    representations; search() still returns only the parent chunk."""
    reps = [chunk.text]
    md = chunk.metadata
    if chunk.doc_type == "session" and "title" in md:
        reps.append(md["title"])
    elif chunk.doc_type == "speaker" and {"full_name", "job_title", "organisation"} <= md.keys():
        reps.append(f"{md['full_name']}, {md['job_title']} at {md['organisation']}")
    return reps


class MultiRepVectorIndex(_EmbeddingModelCache):
    """Like NumpyVectorIndex, but each chunk may contribute more than one
    embedded representation (see build_representations); a chunk's score is
    the max over its representations, and results are deduplicated back to
    the parent chunk (never returned more than once)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        self.chunks: list[Chunk] = []
        self._rep_chunk_idx: np.ndarray | None = None
        self._rep_embeddings: np.ndarray | None = None

    def _cache_key(self, representations: list[str]) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(self.MODEL_NAME.encode("utf-8"))
        h.update(b"multi-rep\0")
        for rep in representations:
            h.update(rep.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        all_reps: list[str] = []
        rep_chunk_idx: list[int] = []
        for i, chunk in enumerate(self.chunks):
            for rep in build_representations(chunk):
                all_reps.append(rep)
                rep_chunk_idx.append(i)
        self._rep_chunk_idx = np.array(rep_chunk_idx)

        key = self._cache_key(all_reps)
        cached = self._load_cached(key)
        if cached is not None:
            self._rep_embeddings = cached
            return

        vecs = np.array(list(self._get_model().passage_embed(all_reps)))
        self._rep_embeddings = self._normalize(vecs)
        self._save_cache(key, self._rep_embeddings)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        assert self._rep_embeddings is not None, "call add() before search()"
        qvec = self._normalize(np.array(list(self._get_model().query_embed([query])))[0])
        rep_scores = self._rep_embeddings @ qvec

        chunk_scores = np.full(len(self.chunks), -np.inf)
        np.maximum.at(chunk_scores, self._rep_chunk_idx, rep_scores)

        top = np.argsort(-chunk_scores)[:k]
        return [(self.chunks[i], float(chunk_scores[i])) for i in top]


# --- v4: foreign-key expansion ----------------------------------------------


@dataclass
class RetrievedItem:
    """A single retrieval result. `rank` is the item's 1-based position
    among the *base* (non-expanded) results; expanded items keep the rank
    they would need their parent to have been within, for recall purposes,
    via `expanded_from` rather than their own position in the list."""

    chunk: Chunk
    score: float
    rank: int
    expanded: bool = False
    expanded_from: str | None = None


def expand_with_foreign_keys(
    base_items: list[RetrievedItem], chunk_by_ref: dict[str, Chunk]
) -> list[RetrievedItem]:
    """Appends each base session item's linked room and speaker chunks
    (via its metadata foreign keys), deduplicated, marked expanded=True.
    Never duplicates a chunk already present, and never invents a chunk for
    a foreign key that doesn't resolve to one. Expanded items are appended
    after the base items; they never displace or reorder them."""
    expanded: list[RetrievedItem] = []
    seen = {item.chunk.chunk_id for item in base_items}

    def _maybe_add(ref: str, parent: RetrievedItem) -> None:
        chunk = chunk_by_ref.get(ref)
        if chunk is None or chunk.chunk_id in seen:
            return
        seen.add(chunk.chunk_id)
        expanded.append(
            RetrievedItem(
                chunk=chunk,
                score=parent.score,
                rank=parent.rank,
                expanded=True,
                expanded_from=parent.chunk.chunk_id,
            )
        )

    for item in base_items:
        if item.chunk.doc_type != "session":
            continue
        room_id = item.chunk.metadata.get("room_id")
        if room_id:
            _maybe_add(f"venue_guide.pdf#rooms:{room_id}", item)
        for speaker_id in item.chunk.metadata.get("speaker_ids", []):
            _maybe_add(f"speakers.json#{speaker_id}", item)

    return base_items + expanded


# --- v5: cross-encoder reranking --------------------------------------------


class CrossEncoderReranker:
    """Reranks a candidate pool with fastembed's cross-encoder support
    (fastembed.rerank.cross_encoder.TextCrossEncoder), confirmed available
    in the installed fastembed version via its actual API, not assumed."""

    MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.MODEL_NAME)
        return self._model

    def rerank(self, query: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]:
        if not items:
            return []
        scores = list(self._get_model().rerank(query, [item.chunk.text for item in items]))
        order = sorted(range(len(items)), key=lambda i: -scores[i])[:top_k]
        return [
            RetrievedItem(chunk=items[i].chunk, score=float(scores[i]), rank=new_rank)
            for new_rank, i in enumerate(order, start=1)
        ]

    @staticmethod
    def is_available() -> bool:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401

            return True
        except ImportError:
            return False


# --- pipeline orchestration ---------------------------------------------


class Pipeline:
    """Builds and runs the full retrieval pipeline for a given config.
    search() always returns a uniform list[RetrievedItem], regardless of
    which features the config turns on."""

    def __init__(
        self, config: RetrievalConfig, chunks: list[Chunk], cache_dir: Path | None = None
    ) -> None:
        self.config = config
        self.chunks = chunks
        self.chunk_by_ref = {c.source_ref: c for c in chunks}
        self.rerank_latencies: list[float] = []
        self.query_latencies: list[float] = []

        tokenizer = tokenize_no_stopwords if config.remove_stopwords else tokenize
        self.bm25 = BM25Index(tokenizer=tokenizer)
        self.bm25.add(chunks)

        vector_cache_dir = (cache_dir / config.name) if cache_dir else None
        self.vector = (
            MultiRepVectorIndex(cache_dir=vector_cache_dir)
            if config.title_vectors
            else NumpyVectorIndex(cache_dir=vector_cache_dir)
        )
        self.vector.add(chunks)

        if config.vector_only:
            self.retriever = self.vector
        else:
            fused = (
                WeightedHybridRetriever(self.bm25, self.vector, config)
                if config.weighted_rrf
                else HybridRetriever(self.bm25, self.vector, rrf_k=config.rrf_k)
            )
            self.retriever = (
                IdPinningRetriever(fused, self.chunk_by_ref) if config.id_pinning else fused
            )

        self.reranker = CrossEncoderReranker() if config.rerank else None

    @property
    def last_unresolved_ids(self) -> list[str]:
        return getattr(self.retriever, "last_unresolved_ids", [])

    def search(self, query: str, k: int) -> list[RetrievedItem]:
        t_start = time.perf_counter()

        pool_k = self.config.rerank_pool if self.reranker else k
        raw = self.retriever.search(query, pool_k)
        items = [
            RetrievedItem(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(raw, start=1)
        ]

        if self.reranker is not None:
            t0 = time.perf_counter()
            items = self.reranker.rerank(query, items, top_k=k)
            self.rerank_latencies.append(time.perf_counter() - t0)

        if self.config.fk_expansion:
            items = expand_with_foreign_keys(items, self.chunk_by_ref)

        self.query_latencies.append(time.perf_counter() - t_start)
        return items


# --- end-to-end answer orchestration (LLM layer) ----------------------------
#
# Imports from rag.router / rag.rewrite / rag.sql_agent / rag.answer are done
# lazily, inside answer(), rather than at module level: those modules import
# from this one (ID_PATTERN, Pipeline, RetrievedItem, expand_with_foreign_keys),
# so importing them back at module level here would be circular.


def build_answer_pipelines(
    chunks: list[Chunk], cache_dir: Path | None = None
) -> dict[str, Pipeline]:
    """Builds the two retrieval pipelines answer() needs, once, so repeated
    answer() calls (e.g. across an eval run) don't reload the embedding
    model or re-embed the corpus each time."""
    return {
        "v6_id_pinning": Pipeline(CONFIGS["v6_id_pinning"], chunks, cache_dir=cache_dir),
        "v7_vector_led": Pipeline(CONFIGS["v7_vector_led"], chunks, cache_dir=cache_dir),
    }


@dataclass
class AnswerResult:
    """The full, user-facing-safe result of answer(). Never constructed by
    raising an exception out of answer() - failures are recorded in
    `errors` and surfaced as a plain-language `answer` instead."""

    query: str
    route: str
    fallback_reason: str | None
    rewrites: list[str]
    #: [{"source_ref": ..., "score": ..., "expanded": bool}, ...]
    retrieved: list[dict]
    sql_result: object | None  # rag.sql_agent.SQLResult, typed loosely to avoid the import cycle
    answer: str
    citations: list[str]
    not_in_kb: bool
    low_confidence: bool
    invalid_citations: list[str]
    stage_latencies_s: dict[str, float]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    errors: list[str]


def _safe_fallback_result(
    query: str, message: str, error: str, stage_latencies: dict[str, float],
    input_tokens: int, output_tokens: int, cost_usd: float,
) -> AnswerResult:
    return AnswerResult(
        query=query,
        route="out_of_scope",
        fallback_reason=None,
        rewrites=[],
        retrieved=[],
        sql_result=None,
        answer=message,
        citations=[],
        not_in_kb=False,
        low_confidence=True,
        invalid_citations=[],
        stage_latencies_s=stage_latencies,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        errors=[error],
    )


def answer(
    query: str,
    *,
    con,
    schema: dict[str, list[tuple[str, str]]],
    retrieval_pipelines: dict[str, Pipeline],
    llm,
    router_model: str,
    answer_model: str,
) -> AnswerResult:
    """Routes, retrieves/queries, and answers `query`. Never raises to the
    caller: any unexpected failure becomes a user-safe AnswerResult with the
    error recorded in `errors`."""
    from rag.answer import answer_from_context, build_context_from_chunks, build_context_from_sql
    from rag.rewrite import retrieve_with_rewrites
    from rag.router import route as route_query
    from rag.sql_agent import generate_and_run_sql

    errors: list[str] = []
    stage_latencies: dict[str, float] = {}
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0

    def _track(llm_result) -> None:
        nonlocal input_tokens, output_tokens, cost_usd
        if llm_result is not None:
            input_tokens += llm_result.usage.input_tokens
            output_tokens += llm_result.usage.output_tokens
            cost_usd += llm_result.usage.cost_usd

    try:
        t0 = time.perf_counter()
        route_outcome = route_query(query, llm, router_model, list(schema.keys()))
        stage_latencies["route"] = time.perf_counter() - t0
        _track(route_outcome.llm_result)
        if route_outcome.fallback_reason:
            errors.append(route_outcome.fallback_reason)

        route_name = route_outcome.route
        rewrites: list[str] = []
        retrieved_items: list[RetrievedItem] = []
        sql_result = None
        context = "(No context retrieved: this question appears to be out of scope " \
                  "for this knowledge base.)"
        valid_refs: set[str] = set()

        if route_name == "sql":
            t0 = time.perf_counter()
            sql_result = generate_and_run_sql(query, con, llm, router_model, schema)
            stage_latencies["sql"] = time.perf_counter() - t0
            if sql_result.error:
                errors.append(f"sql error: {sql_result.error}")
            context, valid_refs = build_context_from_sql(sql_result)

        elif route_name == "id_lookup":
            t0 = time.perf_counter()
            retrieved_items = retrieval_pipelines["v6_id_pinning"].search(query, k=5)
            stage_latencies["retrieve"] = time.perf_counter() - t0
            context, valid_refs = build_context_from_chunks(retrieved_items)

        elif route_name in ("retrieve", "rewrite_retrieve"):
            t0 = time.perf_counter()
            retrieved_items, rewrites, rewrite_llm_result, rewrite_fallback = (
                retrieve_with_rewrites(
                    query,
                    retrieval_pipelines["v7_vector_led"],
                    llm,
                    router_model,
                    k=5,
                    use_rewrites=(route_name == "rewrite_retrieve"),
                )
            )
            stage_latencies["retrieve"] = time.perf_counter() - t0
            _track(rewrite_llm_result)
            if rewrite_fallback:
                errors.append(rewrite_fallback)
            context, valid_refs = build_context_from_chunks(retrieved_items)

        # route_name == "out_of_scope": no retrieval, default context stands.

        t0 = time.perf_counter()
        stage_result = answer_from_context(query, context, valid_refs, llm, answer_model)
        stage_latencies["answer"] = time.perf_counter() - t0
        _track(stage_result.llm_result)
        if stage_result.fallback_reason:
            errors.append(stage_result.fallback_reason)

        return AnswerResult(
            query=query,
            route=route_name,
            fallback_reason=route_outcome.fallback_reason,
            rewrites=rewrites,
            retrieved=[
                {
                    "source_ref": item.chunk.source_ref,
                    "score": item.score,
                    "expanded": item.expanded,
                }
                for item in retrieved_items
            ],
            sql_result=sql_result,
            answer=stage_result.answer,
            citations=stage_result.citations,
            not_in_kb=stage_result.not_in_kb,
            low_confidence=stage_result.low_confidence,
            invalid_citations=stage_result.invalid_citations,
            stage_latencies_s=stage_latencies,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            errors=errors,
        )
    except Exception as exc:  # this boundary must never raise to the caller
        errors.append(f"unexpected pipeline error: {exc!r}")
        return _safe_fallback_result(
            query,
            "Sorry, something went wrong while answering this question.",
            errors[-1],
            stage_latencies,
            input_tokens,
            output_tokens,
            cost_usd,
        )

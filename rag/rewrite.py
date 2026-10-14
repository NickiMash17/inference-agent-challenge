"""Query rewriting for the "rewrite_retrieve" route: produces alternative
phrasings in conference-programme vocabulary, retrieves for the original
query plus each rewrite through the v7_vector_led config, and fuses the
per-query result lists with Reciprocal Rank Fusion."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.pipeline import Pipeline, RetrievedItem, expand_with_foreign_keys
from shared.llm import LLMClient, LLMError, LLMResult

REWRITE_SYSTEM_PROMPT = """You rewrite a user's question into alternative phrasings \
using the vocabulary a conference programme would use - the way a session title or \
abstract would phrase the same topic (track terminology, technical vocabulary, \
session-description style), rather than everyday conversational language.

Produce exactly 3 alternative phrasings of the user's question. Each must preserve \
the original question's meaning and intent, phrased differently from each other and \
from the original. Do not answer the question - only rephrase it."""


class RewriteOut(BaseModel):
    rewrites: list[str] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 alternative phrasings of the question.",
    )


def rewrite_query(
    query: str, llm: LLMClient, model: str
) -> tuple[list[str], LLMResult | None, str | None]:
    """Returns (rewrites, llm_result, fallback_reason). On any failure,
    rewrites is [] and fallback_reason explains why - callers should then
    retrieve for the original query alone."""
    try:
        result = llm.generate(
            model=model,
            system=REWRITE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
            output_model=RewriteOut,
            max_tokens=512,
        )
    except LLMError as exc:
        return [], None, f"rewrite LLM error: {exc}"

    parsed = result.parsed
    if not isinstance(parsed, RewriteOut) or len(parsed.rewrites) != 3:
        return [], result, f"rewrite returned invalid output: {parsed!r}"
    return parsed.rewrites, result, None


def _fuse_multi_query_rrf(
    result_lists: list[list[RetrievedItem]], k: int, rrf_k: int = 60
) -> list[RetrievedItem]:
    """RRF across several query variants' result lists. Only base
    (non-expanded) items are fused; expansions are re-applied once on the
    final fused top-k by the caller, since expansion should reflect the
    final ranking, not any single query variant's partial one."""
    scores: dict[str, float] = {}
    item_by_id: dict[str, RetrievedItem] = {}

    for results in result_lists:
        base_items = sorted((r for r in results if not r.expanded), key=lambda r: r.rank)
        for rank, item in enumerate(base_items, start=1):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            item_by_id.setdefault(cid, item)

    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])[:k]
    return [
        RetrievedItem(chunk=item_by_id[cid].chunk, score=scores[cid], rank=rank)
        for rank, cid in enumerate(ranked_ids, start=1)
    ]


def retrieve_with_rewrites(
    query: str,
    pipeline: Pipeline,
    llm: LLMClient,
    model: str,
    k: int = 5,
    use_rewrites: bool = True,
) -> tuple[list[RetrievedItem], list[str], LLMResult | None, str | None]:
    """Retrieves through `pipeline` (expected: built with v7_vector_led) for
    the original query, plus 3 rewrites if `use_rewrites`. Fuses all
    variants' results with RRF and returns the top-k. The "retrieve" route
    calls this with use_rewrites=False (a single-list "fusion" is just the
    original ranking, so no special-casing is needed); "rewrite_retrieve"
    calls it with use_rewrites=True.

    Returns (retrieved_items, rewrites_used, llm_result, fallback_reason).
    """
    rewrites: list[str] = []
    llm_result: LLMResult | None = None
    fallback_reason: str | None = None

    if use_rewrites:
        rewrites, llm_result, fallback_reason = rewrite_query(query, llm, model)

    query_variants = [query, *rewrites]
    pool_k = max(k * 3, 15)
    per_query_results = [pipeline.search(variant, k=pool_k) for variant in query_variants]

    fused = _fuse_multi_query_rrf(per_query_results, k=k)
    if pipeline.config.fk_expansion:
        fused = expand_with_foreign_keys(fused, pipeline.chunk_by_ref)

    return fused, rewrites, llm_result, fallback_reason

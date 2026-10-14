"""Query router: a deterministic ID pre-route first (no LLM call), then an
LLM classification into sql / retrieve / rewrite_retrieve / out_of_scope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from rag.pipeline import ID_PATTERN
from shared.llm import LLMClient, LLMError, LLMResult

LLMRoute = Literal["sql", "retrieve", "rewrite_retrieve", "out_of_scope"]
Route = Literal["id_lookup", "sql", "retrieve", "rewrite_retrieve", "out_of_scope"]

_VALID_LLM_ROUTES = {"sql", "retrieve", "rewrite_retrieve", "out_of_scope"}

#: A query mentioning an ID AND aggregation language (e.g. "how many
#: sessions is SP-001 presenting") is an aggregation question that happens
#: to name an entity, not a simple ID lookup - it needs sessions_active
#: GROUP BY/COUNT, which the deterministic id_lookup path can't do. Send it
#: to the LLM router instead of short-circuiting.
_AGGREGATION_LANGUAGE = re.compile(
    r"\b(how many|count|number of|average|total|most|least|highest|lowest)\b",
    re.IGNORECASE,
)


class RouteDecision(BaseModel):
    route: LLMRoute
    rationale: str = Field(description="One sentence explaining the choice of route.")


ROUTE_SYSTEM_PROMPT = """You are a query router for a conference knowledge-base \
question-answering system. The knowledge base covers a fictional conference: \
sessions, speakers, venue rooms, ticket tiers, and policies.

The database has these tables available for SQL: {table_names}

Classify the user's question into exactly one route:
- "sql": the question requires counting, averaging, or finding a maximum/minimum \
across many rows (e.g. "how many sessions...", "which speaker has given the \
most...", "what is the average capacity..."). Prefer this whenever the answer \
is a computed number or a "most/least X" fact derived from many rows.
- "retrieve": the question can be answered by finding one or a few specific, \
directly-named passages (a specific session, speaker, room, or policy; a \
cross-reference between two specific named entities).
- "rewrite_retrieve": the question describes a topic indirectly - paraphrasing \
a session's subject without naming the session or using its distinctive \
vocabulary - and would benefit from being rephrased in conference-programme \
vocabulary before searching.
- "out_of_scope": the question cannot plausibly be answered from this \
knowledge base at all (unrelated to the conference, e.g. general trivia, \
current weather, or something no conference knowledge base would contain).

Respond with the single best route and a one-sentence rationale."""


@dataclass(frozen=True)
class RouterOutcome:
    route: Route
    rationale: str
    #: Set only when the LLM path was skipped or failed and the router fell
    #: back to a default route.
    fallback_reason: str | None
    #: None for the deterministic id_lookup path, since it never calls the LLM.
    llm_result: LLMResult | None


def route(
    query: str,
    llm: LLMClient,
    model: str,
    table_names: list[str],
) -> RouterOutcome:
    if ID_PATTERN.search(query) and not _AGGREGATION_LANGUAGE.search(query):
        return RouterOutcome(
            route="id_lookup",
            rationale="query contains an exact ID (session/speaker/room)",
            fallback_reason=None,
            llm_result=None,
        )

    system = ROUTE_SYSTEM_PROMPT.format(table_names=", ".join(table_names))
    try:
        result = llm.generate(
            model=model,
            system=system,
            messages=[{"role": "user", "content": query}],
            output_model=RouteDecision,
            max_tokens=256,
        )
    except LLMError as exc:
        return RouterOutcome(
            route="retrieve",
            rationale="",
            fallback_reason=f"router LLM error: {exc}",
            llm_result=None,
        )

    decision = result.parsed
    if not isinstance(decision, RouteDecision) or decision.route not in _VALID_LLM_ROUTES:
        return RouterOutcome(
            route="retrieve",
            rationale="",
            fallback_reason=f"router returned invalid output: {decision!r}",
            llm_result=result,
        )

    return RouterOutcome(
        route=decision.route,
        rationale=decision.rationale,
        fallback_reason=None,
        llm_result=result,
    )

"""LLM client: a thin, swappable wrapper around the Anthropic SDK that
returns parsed Pydantic objects plus usage/latency/cost - never raw text
the caller has to re-parse.

Structured outputs: checked the installed SDK (anthropic==1.13.0) rather
than hand-rolling forced tool use - it has first-class support via
`client.messages.parse(..., output_format=SomeBaseModel)`, which generates
the JSON schema from the Pydantic model itself and returns a
`ParsedMessage` whose `.parsed_output` is already the validated model
instance. That supersedes the forced-tool-use-with-JSON-schema approach for
this use case, so that's what's used here.

Temperature: this SDK version's `messages.create`/`messages.parse` expose no
`temperature` parameter at all (checked `inspect.signature` directly: it is
simply not in the list of accepted kwargs for this model generation). There
is nothing to set to 0 - sampling is not exposed as a tunable here. This is
called out explicitly rather than silently dropped, and nothing is passed
via `extra_body` to fake it, since that risks a 400 if the API has actually
removed the parameter rather than just hidden it from this SDK.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, TypeVar

import anthropic
import pydantic
from pydantic import BaseModel

from shared.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Unified error for any LLM failure (timeout, rate limit after SDK
    retries are exhausted, invalid/unparseable structured output, ...).
    Callers (router, sql_agent, answer, pipeline) only need to catch this
    one type regardless of the underlying cause."""


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


@dataclass(frozen=True)
class LLMResult:
    parsed: BaseModel
    usage: LLMUsage
    model: str


class LLMClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        output_model: type[T],
        max_tokens: int = 1024,
    ) -> LLMResult: ...


class AnthropicClient:
    """Real client. api_key is always taken explicitly from Settings -
    never left for the SDK to discover from an ANTHROPIC_API_KEY env var
    (see CLAUDE.md)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.atlas_anthropic_api_key:
            raise LLMError("ATLAS_ANTHROPIC_API_KEY is not set")
        self._settings = settings
        self._client = anthropic.Anthropic(
            api_key=settings.atlas_anthropic_api_key,
            timeout=settings.request_timeout_s,
            # The SDK retries connection errors and 408/409/429/5xx
            # responses internally, with backoff, up to this many times.
            max_retries=settings.max_retries,
        )

    def generate(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        output_model: type[T],
        max_tokens: int = 1024,
    ) -> LLMResult:
        t0 = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_format=output_model,
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(f"{model} request failed: {exc}") from exc
        except pydantic.ValidationError as exc:
            # messages.parse() validates the model's raw text against
            # `output_model` internally; a truncated or otherwise malformed
            # structured-output response (e.g. max_tokens cut off mid-JSON)
            # surfaces here as a pydantic ValidationError, not an
            # anthropic.AnthropicError. Wrapping it as LLMError lets
            # downstream retry logic (e.g. rag/sql_agent.py) handle it the
            # same way it handles any other generation failure.
            raise LLMError(f"{model} returned unparseable structured output: {exc}") from exc
        latency_s = time.perf_counter() - t0

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"{model} did not return parseable {output_model.__name__} output")

        pricing = self._settings.pricing.get(model)
        cost_usd = (
            pricing.cost_usd(response.usage.input_tokens, response.usage.output_tokens)
            if pricing
            else 0.0
        )
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency_s,
            cost_usd=cost_usd,
        )
        return LLMResult(parsed=parsed, usage=usage, model=model)


class FakeLLMClient:
    """Test double: no network. Scripted responses, popped off a queue in
    order. A queued item may be an LLMResult (returned) or a BaseException
    instance (raised), to simulate failures. Alternatively, pass a
    callable that computes the response/exception per call."""

    def __init__(self, scripted=None, respond=None) -> None:
        self._queue = list(scripted or [])
        self._respond = respond
        self.calls: list[dict] = []

    def generate(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        output_model: type[T],
        max_tokens: int = 1024,
    ) -> LLMResult:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "output_model": output_model,
                "max_tokens": max_tokens,
            }
        )
        if self._respond is not None:
            result = self._respond(
                model=model, system=system, messages=messages, output_model=output_model
            )
        else:
            if not self._queue:
                raise AssertionError("FakeLLMClient: no more scripted responses")
            result = self._queue.pop(0)

        if isinstance(result, BaseException):
            raise result
        return result

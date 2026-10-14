"""Application settings, read from environment variables / .env.

Per CLAUDE.md: the API key variable is ATLAS_ANTHROPIC_API_KEY. Never read
or set ANTHROPIC_API_KEY anywhere in this project.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseModel):
    """USD per 1,000,000 tokens."""

    input_per_million_usd: float
    output_per_million_usd: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million_usd
            + output_tokens * self.output_per_million_usd
        ) / 1_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    #: Never fall back to the SDK's default ANTHROPIC_API_KEY env var - this
    #: is the only key this project reads, and it's passed explicitly to
    #: the Anthropic client rather than left for the SDK to discover.
    atlas_anthropic_api_key: str | None = Field(default=None, alias="ATLAS_ANTHROPIC_API_KEY")

    router_model: str = Field(default="claude-haiku-5-5", alias="ROUTER_MODEL")
    answer_model: str = Field(default="claude-sonnet-5-5", alias="ANSWER_MODEL")

    max_cost_per_eval_run_usd: float = Field(default=2.00, alias="MAX_COST_PER_EVAL_RUN_USD")

    request_timeout_s: float = Field(default=30.0, alias="LLM_REQUEST_TIMEOUT_S")
    #: The Anthropic SDK retries connection errors and 408/409/429/5xx
    #: responses internally with backoff up to this many times.
    max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    @property
    def pricing(self) -> dict[str, ModelPricing]:
        """Prices in USD per million tokens, keyed by model name.

        Verify against the official Anthropic pricing page
        (https://www.anthropic.com/pricing) before relying on these for
        real billing decisions - they are a snapshot, not a live source.
        """
        return {
            # Haiku pricing below is valid for prompts under 100k tokens.
            self.router_model: ModelPricing(
                input_per_million_usd=0.10, output_per_million_usd=0.50
            ),
            self.answer_model: ModelPricing(
                input_per_million_usd=2.00, output_per_million_usd=10.00
            ),
        }


def api_key_status(settings: Settings) -> str:
    """"present" or "absent" - never the key value itself."""
    return "present" if settings.atlas_anthropic_api_key else "absent"

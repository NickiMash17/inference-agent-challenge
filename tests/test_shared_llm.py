"""Tests for shared/config.py and shared/llm.py. No network."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from shared.config import ModelPricing, Settings, api_key_status
from shared.llm import AnthropicClient, FakeLLMClient, LLMError, LLMResult, LLMUsage


class _Foo(BaseModel):
    x: int


def test_api_key_status_present_and_absent():
    assert api_key_status(Settings(_env_file=None, atlas_anthropic_api_key="sk-fake")) == "present"
    assert api_key_status(Settings(_env_file=None, atlas_anthropic_api_key=None)) == "absent"


def test_settings_never_reads_plain_anthropic_api_key_env_var(monkeypatch):
    """Per CLAUDE.md: ATLAS_ANTHROPIC_API_KEY is the only key this project
    reads. A plain ANTHROPIC_API_KEY in the environment must be ignored."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    monkeypatch.delenv("ATLAS_ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.atlas_anthropic_api_key is None


def test_settings_defaults():
    settings = Settings(_env_file=None, atlas_anthropic_api_key="sk-fake")
    assert settings.router_model == "claude-haiku-5-5"
    assert settings.answer_model == "claude-sonnet-5-5"
    assert settings.max_cost_per_eval_run_usd == pytest.approx(2.00)


def test_model_pricing_cost_calculation():
    pricing = ModelPricing(input_per_million_usd=0.10, output_per_million_usd=0.50)
    # 1,000,000 input tokens @ $0.10/M + 1,000,000 output tokens @ $0.50/M
    assert pricing.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.60)
    assert pricing.cost_usd(0, 0) == 0.0


def test_fake_llm_client_returns_scripted_results_in_order():
    r1 = LLMResult(parsed=_Foo(x=1), usage=LLMUsage(1, 1, 0.01, 0.0), model="m")
    r2 = LLMResult(parsed=_Foo(x=2), usage=LLMUsage(1, 1, 0.01, 0.0), model="m")
    fake = FakeLLMClient(scripted=[r1, r2])

    out1 = fake.generate(model="m", system="s", messages=[], output_model=_Foo)
    out2 = fake.generate(model="m", system="s", messages=[], output_model=_Foo)

    assert out1.parsed.x == 1
    assert out2.parsed.x == 2
    assert len(fake.calls) == 2


def test_fake_llm_client_raises_scripted_exceptions():
    fake = FakeLLMClient(scripted=[ValueError("boom")])
    with pytest.raises(ValueError, match="boom"):
        fake.generate(model="m", system="s", messages=[], output_model=_Foo)


def test_fake_llm_client_raises_assertion_error_when_queue_exhausted():
    fake = FakeLLMClient(scripted=[])
    with pytest.raises(AssertionError):
        fake.generate(model="m", system="s", messages=[], output_model=_Foo)


# --- conftest.py safety nets (see tests/conftest.py) --------------------------


def test_dotenv_loading_is_disabled_during_tests():
    """Proves the autouse fixture in conftest.py is active: even a
    Settings() call that forgets to pass _env_file=None must not read the
    real .env while the test suite is running."""
    assert Settings.model_config.get("env_file") is None


def test_constructing_a_real_anthropic_client_is_blocked_in_tests():
    """Proves that a test which accidentally tries to build a real
    AnthropicClient fails fast with a clear error instead of calling the
    API (see tests/conftest.py's _block_real_network)."""
    settings = Settings(_env_file=None, atlas_anthropic_api_key="sk-fake-for-this-test")
    with pytest.raises(RuntimeError, match="network blocked in tests"):
        AnthropicClient(settings)


def test_anthropic_client_wraps_truncated_structured_output_as_llm_error():
    """Regression test for the G-016 bug: when max_tokens truncates the
    model's JSON mid-response, the SDK's messages.parse() raises a
    pydantic.ValidationError while validating the (incomplete) text against
    output_format - not an anthropic.AnthropicError. generate() must wrap
    that as LLMError too, so downstream retry logic can handle it.

    Builds an AnthropicClient without calling anthropic.Anthropic.__init__
    (which tests/conftest.py blocks) - this test exercises generate()'s own
    exception handling in isolation, not SDK construction, so there's
    nothing to fake-network-call in the first place.
    """

    class _StubMessages:
        def parse(self, **kwargs):
            # Mirrors what the real SDK does internally: validates the
            # model's raw response text against output_format and raises a
            # genuine pydantic.ValidationError on truncated/invalid JSON.
            return _Foo.model_validate_json('{"x": ')

    class _StubAnthropicSDKClient:
        messages = _StubMessages()

    settings = Settings(_env_file=None, atlas_anthropic_api_key="sk-fake-for-this-test")
    client = AnthropicClient.__new__(AnthropicClient)
    client._settings = settings
    client._client = _StubAnthropicSDKClient()

    with pytest.raises(LLMError, match="unparseable structured output"):
        client.generate(model="m", system="s", messages=[], output_model=_Foo)

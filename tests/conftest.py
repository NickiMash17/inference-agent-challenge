"""Test-suite-wide safety nets (see CLAUDE.md: never reach the live API
from a test). These are autouse and apply to every test NOT marked
@pytest.mark.live:

1. Settings never reads the real .env during tests, regardless of whether
   an individual Settings(...) call remembers to pass _env_file=None. This
   is what actually failed before: a test deleted the ATLAS_ANTHROPIC_API_KEY
   *process* env var, but Settings still picked the key up from .env
   directly, so a "simulate missing key" test silently had a real key.
2. Constructing a real anthropic.Anthropic client raises immediately. Even
   if some future code path builds one without going through
   shared.llm.AnthropicClient, it still can't reach the network from a test.
"""

from __future__ import annotations

import anthropic
import pytest

from shared.config import Settings


class NetworkBlockedInTests(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_dotenv_in_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if "live" in request.keywords:
        yield
        return
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    yield


@pytest.fixture(autouse=True)
def _block_real_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if "live" in request.keywords:
        yield
        return

    def _blocked_init(self, *args, **kwargs):
        raise NetworkBlockedInTests(
            "network blocked in tests: constructing a real anthropic.Anthropic client is "
            "not allowed outside @pytest.mark.live tests. Use shared.llm.FakeLLMClient instead."
        )

    monkeypatch.setattr(anthropic.Anthropic, "__init__", _blocked_init)
    yield

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from agent.llm.contracts import LLMResponse
from agent.llm.fallback_provider import FallbackLLMProvider
from agent.llm.open_code_provider import OpenCodeProviderError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_llm_provider.py"
spec = importlib.util.spec_from_file_location("check_llm_provider", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class Stub:
    def __init__(self, name, *, answer="pong", error=None):
        self.name, self.answer, self.error, self.calls = name, answer, error, 0

    def complete(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(content=self.answer, provider=self.name)


class Clock:
    """Each call advances by the next step, so durations are deterministic."""

    def __init__(self, *durations):
        self.now, self.steps = 0.0, list(durations)
        self.toggle = False

    def __call__(self):
        self.toggle = not self.toggle
        if not self.toggle:                       # second read of a pair: end of a probe
            self.now += self.steps.pop(0)
        return self.now


def _text(lines):
    return "\n".join(lines)


def test_every_provider_is_probed_on_its_own_with_its_timing():
    first, second = Stub("opencode"), Stub("claude-code")
    ok, lines = tool.check_providers(lambda: FallbackLLMProvider([first, second]), clock=Clock(4.2, 9.0))
    text = _text(lines)
    assert ok and first.calls == 1 and second.calls == 1          # both, not "first success wins"
    assert "#1 opencode: answered in 4.2 s" in text and "#2 claude-code: answered in 9.0 s" in text


def test_a_failing_first_provider_explains_the_wait_and_the_fix():
    first = Stub("opencode", error=OpenCodeProviderError("timed out"))
    second = Stub("claude-code")
    ok, lines = tool.check_providers(lambda: FallbackLLMProvider([first, second]), clock=Clock(300.0, 7.0))
    text = _text(lines)
    assert ok
    assert "[FAIL] #1 opencode: no answer after 300.0 s" in text and "timed out" in text
    assert "waits for its timeout" in text and "HAI_LLM_PROVIDER=claude-code" in text


def test_a_slow_but_working_first_provider_is_flagged():
    ok, lines = tool.check_providers(lambda: FallbackLLMProvider([Stub("opencode"), Stub("claude-code")]),
                                     clock=Clock(45.0, 3.0))
    assert ok and "slow" in _text(lines).splitlines()[0]


def test_nothing_answering_is_reported_as_a_failure():
    both_bad = FallbackLLMProvider([Stub("opencode", error=OpenCodeProviderError("a")),
                                    Stub("claude-code", error=OpenCodeProviderError("b"))])
    ok, lines = tool.check_providers(lambda: both_bad, clock=Clock(1.0, 1.0))
    assert not ok and "No provider answered" in _text(lines)


def test_a_single_provider_setting_is_probed_once():
    only = Stub("claude-code")
    ok, lines = tool.check_providers(lambda: only, clock=Clock(2.0))
    assert ok and only.calls == 1 and "#1 claude-code" in _text(lines)


def test_a_misconfigured_provider_is_reported_not_raised():
    def broken():
        raise RuntimeError("HAI_LLM_PROVIDER must be 'auto', 'opencode' or 'claude-code'")
    ok, lines = tool.check_providers(broken)
    assert not ok and "not configured correctly" in _text(lines) and "HAI_LLM_PROVIDER" in _text(lines)


def test_the_probe_prompt_is_tiny_and_carries_no_data():
    seen = []

    class Spy(Stub):
        def complete(self, request):
            seen.append(request)
            return super().complete(request)
    tool.check_providers(lambda: Spy("opencode"), clock=Clock(1.0))
    (request,) = seen
    assert len(request.messages) == 1 and request.messages[0].content == tool.PROMPT
    assert len(tool.PROMPT) < 60


def test_main_caps_the_timeouts_for_the_check_and_loads_env(tmp_path, monkeypatch, capsys):
    saved = dict(os.environ)
    try:
        env = tmp_path / ".env"
        env.write_text("OPENCODE_TIMEOUT=300\nHAI_LLM_PROVIDER=auto\n", encoding="utf-8")
        for name in ("OPENCODE_TIMEOUT", "CLAUDE_CODE_TIMEOUT", "HAI_LLM_PROVIDER"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr("api.dependencies.create_llm_provider", lambda: Stub("claude-code"))
        assert tool.main(["--env-file", str(env), "--timeout", "30"]) == 0
        out = capsys.readouterr().out
        assert "OPENCODE_TIMEOUT=300" in out                   # shows the real setting ...
        assert os.environ["OPENCODE_TIMEOUT"] == "30.0"        # ... but the check itself is capped
        assert "Result: at least one provider works." in out
    finally:
        os.environ.clear()
        os.environ.update(saved)

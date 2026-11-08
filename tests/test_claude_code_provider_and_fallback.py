from __future__ import annotations

import json
import os
import subprocess

import pytest

from agent.llm.claude_code_provider import (
    ClaudeCodeConfigurationError,
    ClaudeCodeProcessError,
    ClaudeCodeProvider,
    ClaudeCodeResponseParseError,
    ClaudeCodeTimeoutError,
)
from agent.llm.contracts import LLMMessage, LLMRequest, LLMResponse
from agent.llm.fallback_provider import AllProvidersFailedError, FallbackLLMProvider
from agent.llm.open_code_provider import OpenCodeConfigurationError

OK_JSON = json.dumps({
    "type": "result", "is_error": False, "result": "pong", "stop_reason": "end_turn",
    "session_id": "s-1", "total_cost_usd": 0.004,
    "usage": {"input_tokens": 2, "output_tokens": 4},
    "modelUsage": {"claude-sonnet-5-5": {}},
})


def _request(*messages):
    return LLMRequest(messages=tuple(messages or (LLMMessage("user", "ping"),)))


def _runner(stdout="", returncode=0, stderr="", record=None, exc=None):
    def run(command, cwd, timeout, env, stdin_text):
        if record is not None:
            record.update(command=list(command), cwd=cwd, timeout=timeout, stdin=stdin_text)
        if exc:
            raise exc
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    return run


def test_success_maps_result_usage_and_model():
    p = ClaudeCodeProvider(executable="claude", runner=_runner(OK_JSON))
    r = p.complete(_request())
    assert (r.content, r.provider, r.model, r.finish_reason) == (
        "pong", "claude-code", "claude-sonnet-5-5", "end_turn")
    assert r.usage["output_tokens"] == 4 and r.usage["total_cost_usd"] == 0.004


def test_command_disables_tools_and_session_and_sends_prompt_on_stdin():
    seen = {}
    p = ClaudeCodeProvider(executable="claude", model="sonnet", runner=_runner(OK_JSON, record=seen))
    p.complete(_request(LLMMessage("system", "Be terse."), LLMMessage("user", "ping")))
    cmd = seen["command"]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--system-prompt") + 1] == "Be terse."
    assert "ping" in seen["stdin"] and "Be terse." not in seen["stdin"]
    assert "ping" not in cmd  # prompt never on the command line


def test_error_result_raises_process_error():
    bad = json.dumps({"is_error": True, "result": "rate limited"})
    p = ClaudeCodeProvider(executable="claude", runner=_runner(bad))
    with pytest.raises(ClaudeCodeProcessError, match="rate limited"):
        p.complete(_request())


def test_nonzero_exit_timeout_missing_binary_and_garbage_output():
    with pytest.raises(ClaudeCodeProcessError):
        ClaudeCodeProvider(executable="c", runner=_runner("", 1, "boom")).complete(_request())
    with pytest.raises(ClaudeCodeTimeoutError):
        ClaudeCodeProvider(executable="c", runner=_runner(
            exc=subprocess.TimeoutExpired("c", 1))).complete(_request())
    with pytest.raises(ClaudeCodeConfigurationError):
        ClaudeCodeProvider(executable="c", runner=_runner(
            exc=FileNotFoundError())).complete(_request())
    with pytest.raises(ClaudeCodeResponseParseError):
        ClaudeCodeProvider(executable="c", runner=_runner("not json")).complete(_request())


def test_stray_log_lines_before_json_are_tolerated():
    p = ClaudeCodeProvider(executable="claude", runner=_runner("warning: x\n" + OK_JSON))
    assert p.complete(_request()).content == "pong"


def test_missing_cli_without_runner_fails_clearly(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_EXECUTABLE", raising=False)
    monkeypatch.setattr("agent.llm.claude_code_provider.shutil.which", lambda _n: None)
    monkeypatch.setattr("agent.llm.claude_code_provider.glob.glob", lambda _p: [])
    with pytest.raises(ClaudeCodeConfigurationError, match="not found"):
        ClaudeCodeProvider()


class _Stub:
    def __init__(self, name, *, result=None, error=None):
        self.name, self.result, self.error, self.calls = name, result, error, 0

    def complete(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(content=self.result, provider=self.name)


def test_fallback_uses_second_provider_when_first_is_unavailable():
    primary = _Stub("opencode", error=OpenCodeConfigurationError("not installed"))
    backup = _Stub("claude-code", result="from claude")
    fb = FallbackLLMProvider([primary, backup])
    r = fb.complete(_request())
    assert (r.content, r.provider) == ("from claude", "claude-code")
    assert fb.last_used_provider == "claude-code"
    assert [n for n, _ in fb.last_failures] == ["opencode"]


def test_fallback_does_not_call_backup_when_primary_works():
    primary, backup = _Stub("opencode", result="ok"), _Stub("claude-code", result="x")
    FallbackLLMProvider([primary, backup]).complete(_request())
    assert backup.calls == 0


def test_programming_errors_are_not_masked_by_fallback():
    primary = _Stub("opencode", error=TypeError("bug"))
    backup = _Stub("claude-code", result="x")
    with pytest.raises(TypeError):
        FallbackLLMProvider([primary, backup]).complete(_request())
    assert backup.calls == 0


def test_all_failing_raises_with_every_reason():
    fb = FallbackLLMProvider([
        _Stub("opencode", error=OpenCodeConfigurationError("a")),
        _Stub("claude-code", error=ClaudeCodeTimeoutError("b")),
    ])
    with pytest.raises(AllProvidersFailedError) as info:
        fb.complete(_request())
    assert [n for n, _ in info.value.failures] == ["opencode", "claude-code"]


def test_create_llm_provider_modes(monkeypatch):
    from api import dependencies as d
    monkeypatch.setattr(d, "discover_claude_executable", lambda: "claude")
    monkeypatch.setenv("HAI_LLM_PROVIDER", "auto")
    assert isinstance(d.create_llm_provider(), FallbackLLMProvider)
    monkeypatch.setattr(d, "discover_claude_executable", lambda: None)
    assert d.create_llm_provider().name == "opencode"  # no Claude CLI -> OpenCode only
    monkeypatch.setenv("HAI_LLM_PROVIDER", "opencode")
    assert d.create_llm_provider().name == "opencode"
    monkeypatch.setenv("HAI_LLM_PROVIDER", "bogus")
    with pytest.raises(RuntimeError):
        d.create_llm_provider()


@pytest.mark.skipif(os.getenv("HAI_RUN_LIVE_LLM_TESTS") != "1",
                    reason="live Claude call; set HAI_RUN_LIVE_LLM_TESTS=1 to run")
def test_live_claude_code_roundtrip():
    r = ClaudeCodeProvider(timeout=120).complete(
        _request(LLMMessage("user", "Reply with exactly: pong")))
    assert "pong" in r.content.lower()


def test_a_fallback_is_logged_with_how_long_the_failed_provider_took(caplog, monkeypatch):
    """Operators watching the server terminal can see that the first provider is the slow one."""
    import itertools
    # the failing provider "took" 95 seconds; afterwards the clock keeps returning the last value
    clock = itertools.chain([100.0, 195.0], itertools.repeat(195.0))
    monkeypatch.setattr("agent.llm.fallback_provider.time.monotonic", lambda: next(clock))
    primary = _Stub("opencode", error=ClaudeCodeTimeoutError("timed out"))
    backup = _Stub("claude-code", result="ok")
    with caplog.at_level("WARNING", logger="agent.llm.fallback_provider"):
        FallbackLLMProvider([primary, backup]).complete(_request())
    assert "opencode failed after 95 s" in caplog.text and "ClaudeCodeTimeoutError" in caplog.text


# ------------------------------------------------------------------------- cooldown
class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _cooldown_pair(cooldown=300.0):
    clock = _Clock()
    primary = _Stub("opencode", error=OpenCodeConfigurationError("usage limit reached"))
    backup = _Stub("claude-code", result="from claude")
    return FallbackLLMProvider([primary, backup], cooldown_seconds=cooldown, clock=clock), primary, backup, clock


def test_a_provider_that_just_failed_is_skipped_so_only_the_first_request_pays():
    fb, primary, backup, clock = _cooldown_pair()
    assert fb.complete(_request()).content == "from claude"
    assert primary.calls == 1                       # the first request paid for the failure ...
    for _ in range(5):
        clock.now += 10
        assert fb.complete(_request()).content == "from claude"
    assert primary.calls == 1 and backup.calls == 6  # ... the next ones went straight to Claude


def test_the_failed_provider_is_tried_again_after_the_cooldown_and_a_success_clears_it():
    fb, primary, backup, clock = _cooldown_pair(cooldown=60)
    fb.complete(_request())
    clock.now += 61
    primary.error, primary.result = None, "opencode is back"
    assert fb.complete(_request()).content == "opencode is back" and primary.calls == 2
    assert fb.complete(_request()).content == "opencode is back" and primary.calls == 3   # no longer resting


def test_a_failure_after_the_cooldown_starts_a_new_cooldown():
    fb, primary, backup, clock = _cooldown_pair(cooldown=60)
    fb.complete(_request())
    clock.now += 61
    fb.complete(_request())                          # tried again, fails again
    assert primary.calls == 2
    clock.now += 30
    fb.complete(_request())
    assert primary.calls == 2                        # resting again


def test_a_resting_provider_is_still_the_last_resort_when_the_others_fail():
    fb, primary, backup, clock = _cooldown_pair()
    fb.complete(_request())                          # opencode now resting
    backup.error = ClaudeCodeTimeoutError("claude down")
    primary.error, primary.result = None, "opencode recovered"
    assert fb.complete(_request()).content == "opencode recovered"   # tried despite resting
    assert primary.calls == 2


def test_when_every_provider_is_resting_they_are_all_tried_in_order():
    clock = _Clock()
    a = _Stub("opencode", error=OpenCodeConfigurationError("a"))
    b = _Stub("claude-code", error=ClaudeCodeTimeoutError("b"))
    fb = FallbackLLMProvider([a, b], cooldown_seconds=300, clock=clock)
    with pytest.raises(AllProvidersFailedError):
        fb.complete(_request())
    clock.now += 5
    with pytest.raises(AllProvidersFailedError) as info:
        fb.complete(_request())
    assert a.calls == 2 and b.calls == 2             # not refused just because both are resting
    assert [n for n, _ in info.value.failures] == ["opencode", "claude-code"]


def test_cooldown_is_off_by_default_and_changes_nothing():
    primary = _Stub("opencode", error=OpenCodeConfigurationError("x"))
    backup = _Stub("claude-code", result="ok")
    fb = FallbackLLMProvider([primary, backup])
    for _ in range(3):
        fb.complete(_request())
    assert primary.calls == 3


def test_programming_errors_are_still_not_masked_with_a_cooldown():
    fb = FallbackLLMProvider([_Stub("opencode", error=TypeError("bug")), _Stub("claude-code", result="x")],
                             cooldown_seconds=300)
    with pytest.raises(TypeError):
        fb.complete(_request())


def test_the_skip_is_logged_once_per_request(caplog):
    fb, primary, backup, clock = _cooldown_pair()
    fb.complete(_request())
    with caplog.at_level("WARNING", logger="agent.llm.fallback_provider"):
        fb.complete(_request())
    assert "opencode failed recently; skipping it for 300 s" in caplog.text


def test_invalid_cooldown_values_are_rejected():
    for bad in (-1, True):
        with pytest.raises(ValueError):
            FallbackLLMProvider([_Stub("a", result="x")], cooldown_seconds=bad)


def test_auto_mode_enables_the_cooldown_from_the_environment(monkeypatch):
    from api import dependencies as d
    monkeypatch.setattr(d, "discover_claude_executable", lambda: "claude")
    monkeypatch.setenv("HAI_LLM_PROVIDER", "auto")
    monkeypatch.delenv("HAI_LLM_PROVIDER_COOLDOWN_SECONDS", raising=False)
    assert d.create_llm_provider().cooldown_seconds == 300
    monkeypatch.setenv("HAI_LLM_PROVIDER_COOLDOWN_SECONDS", "45")
    assert d.create_llm_provider().cooldown_seconds == 45

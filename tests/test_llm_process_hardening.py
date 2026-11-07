from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from agent.llm.contracts import LLMMessage, LLMRequest
from agent.llm.process_runner import run_cli, sanitized_environment

PY = sys.executable


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ------------------------------------------------------------------ environment
SECRETS = {"HAI_MYSQL_LOCAL_PASSWORD": "s1", "hai_lowercase": "s2", "KC_DB_PASSWORD": "s3",
           "KC_BOOTSTRAP_ADMIN_PASSWORD": "s4", "SOME_APP_PASSWORD": "s5", "OTHER_SECRET": "s6",
           "DB_PASSWD": "s7", "keycloak_client_secret": "s8"}
KEPT = {"PATH": "/bin", "HOME": "/h", "USERPROFILE": "C:/u", "ANTHROPIC_API_KEY": "k",
        "CLAUDE_CODE_OAUTH_TOKEN": "t", "OPENCODE_MODEL": "m", "OPENCODE_TIMEOUT": "300", "LANG": "C"}


def test_application_secrets_are_removed_and_the_clis_own_variables_are_kept():
    env = sanitized_environment({**SECRETS, **KEPT})
    assert set(env) == set(KEPT)
    assert not any(v in env.values() for v in SECRETS.values())


def test_overrides_are_applied_after_filtering_and_the_input_is_not_modified():
    base = {**SECRETS, **KEPT}
    snapshot = dict(base)
    env = sanitized_environment(base, {"HAI_FOR_THE_CHILD": "yes", "PATH": "/custom"})
    assert env["HAI_FOR_THE_CHILD"] == "yes" and env["PATH"] == "/custom"
    assert base == snapshot


def test_the_default_base_is_the_current_environment(monkeypatch):
    monkeypatch.setenv("HAI_TEST_SECRET_VALUE", "x")
    monkeypatch.setenv("TEST_KEEP_ME", "y")
    env = sanitized_environment()
    assert "HAI_TEST_SECRET_VALUE" not in env and env["TEST_KEEP_ME"] == "y"


# ---------------------------------------------------------------------- run_cli
def test_output_exit_code_and_stdin_text_work_like_subprocess_run():
    done = run_cli([PY, "-c", "import sys; data = sys.stdin.read(); print(data.upper()); sys.exit(3)"],
                   cwd=None, env=None, timeout=20, stdin_text="hello")
    assert done.returncode == 3 and done.stdout.strip() == "HELLO" and done.stderr == ""


def test_stderr_is_captured_and_non_ascii_survives():
    # CLIs such as OpenCode and Claude Code (Node) write UTF-8; make the Python child do the same.
    done = run_cli([PY, "-c", "import sys; sys.stderr.write('oops'); print('caf\\u00e9')"],
                   cwd=None, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=20)
    assert done.stderr == "oops" and "caf\u00e9" in done.stdout


def test_the_child_cannot_see_application_secrets(monkeypatch):
    monkeypatch.setenv("HAI_MYSQL_LOCAL_PASSWORD", "super-secret-value")
    monkeypatch.setenv("TEST_VISIBLE", "visible")
    done = run_cli([PY, "-c", "import os; print(os.environ.get('HAI_MYSQL_LOCAL_PASSWORD'), os.environ.get('TEST_VISIBLE'))"],
                   cwd=None, env=sanitized_environment(), timeout=20)
    assert done.stdout.split() == ["None", "visible"]


def test_a_child_that_reads_the_keyboard_gets_end_of_input_instead_of_waiting_forever():
    started = time.monotonic()
    done = run_cli([PY, "-c", "import sys; print(repr(sys.stdin.read()))"], cwd=None, env=None, timeout=20)
    assert done.stdout.strip() == "''" and time.monotonic() - started < 15


def test_a_timeout_kills_the_whole_tree_and_returns_promptly(tmp_path):
    """A grandchild that keeps the pipes open used to block the caller long after the timeout."""
    pid_file = tmp_path / "grandchild.pid"
    grandchild = f"import os,time; open(r'{pid_file}','w').write(str(os.getpid())); time.sleep(120)"
    parent = (f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{grandchild!r}]); time.sleep(120)")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_cli([PY, "-c", parent], cwd=None, env=None, timeout=3)
    elapsed = time.monotonic() - started
    assert elapsed < 20, f"returned after {elapsed:.0f} s"
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if pid_file.exists():                       # the grandchild did start: it must be gone now
        pid = int(pid_file.read_text())
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.2)
        assert not _alive(pid), "the grandchild survived the timeout"


def test_the_working_directory_is_honoured(tmp_path):
    done = run_cli([PY, "-c", "import os; print(os.getcwd())"], cwd=str(tmp_path), env=None, timeout=20)
    assert os.path.samefile(done.stdout.strip(), tmp_path)


# --------------------------------------------------------- wiring into providers
def test_the_claude_provider_runs_through_the_safe_runner_without_secrets(monkeypatch):
    from agent.llm import claude_code_provider as mod
    monkeypatch.setenv("HAI_MYSQL_LOCAL_PASSWORD", "super-secret-value")
    monkeypatch.setenv("TEST_VISIBLE", "visible")
    seen = {}

    def fake(command, *, cwd, env, timeout, stdin_text=None):
        seen.update(command=command, env=env, timeout=timeout, stdin=stdin_text)
        return subprocess.CompletedProcess(command, 0, '{"type":"result","is_error":false,"result":"pong"}', "")
    monkeypatch.setattr(mod, "run_cli", fake)
    provider = mod.ClaudeCodeProvider(executable="claude", timeout=33)
    assert provider.complete(LLMRequest(messages=(LLMMessage("user", "hi"),))).content == "pong"
    assert "HAI_MYSQL_LOCAL_PASSWORD" not in seen["env"] and seen["env"]["TEST_VISIBLE"] == "visible"
    assert seen["timeout"] == 33 and "hi" in seen["stdin"] and "hi" not in seen["command"]


def test_the_opencode_provider_runs_through_the_safe_runner_without_secrets(monkeypatch, tmp_path):
    from agent.llm import open_code_provider as mod
    monkeypatch.setenv("HAI_MYSQL_LOCAL_PASSWORD", "super-secret-value")
    monkeypatch.setenv("TEST_VISIBLE", "visible")
    seen = {}

    def fake(command, *, cwd, env, timeout, stdin_text=None):
        seen.update(command=command, env=env, timeout=timeout)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(mod, "run_cli", fake)
    exe = tmp_path / "opencode-test.exe"
    exe.write_text("x")
    provider = mod.OpenCodeProvider(executable=str(exe), timeout=44, auto_approve=False)
    provider._run_command([str(exe), "run", "hello"])
    assert "HAI_MYSQL_LOCAL_PASSWORD" not in seen["env"] and seen["env"]["TEST_VISIBLE"] == "visible"
    assert seen["timeout"] == 44


def test_explicit_provider_environment_overrides_still_reach_the_child(monkeypatch, tmp_path):
    from agent.llm import open_code_provider as mod
    seen = {}
    monkeypatch.setattr(mod, "run_cli", lambda c, *, cwd, env, timeout, stdin_text=None:
                        seen.update(env=env) or subprocess.CompletedProcess(c, 0, "", ""))
    exe = tmp_path / "oc.exe"
    exe.write_text("x")
    provider = mod.OpenCodeProvider(executable=str(exe), env={"HAI_PASSED_ON_PURPOSE": "1"}, auto_approve=False)
    provider._run_command([str(exe), "run", "x"])
    assert seen["env"]["HAI_PASSED_ON_PURPOSE"] == "1"


# ----------------------------------------------------------- fast failover in auto
@pytest.fixture
def auto_mode(monkeypatch):
    from api import dependencies as d
    monkeypatch.setattr(d, "discover_claude_executable", lambda: "claude")
    monkeypatch.setenv("HAI_LLM_PROVIDER", "auto")
    for name in ("OPENCODE_TIMEOUT", "CLAUDE_CODE_TIMEOUT", "HAI_LLM_FAILOVER_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    return d


def test_a_stuck_first_provider_is_capped_so_the_second_gets_a_turn_quickly(auto_mode, monkeypatch):
    monkeypatch.setenv("OPENCODE_TIMEOUT", "300")           # the configured value that caused the long wait
    first, second = auto_mode.create_llm_provider().providers
    assert first.timeout == 60 and second.timeout == 120


def test_the_cap_is_configurable_and_never_raises_a_smaller_timeout(auto_mode, monkeypatch):
    monkeypatch.setenv("OPENCODE_TIMEOUT", "300")
    monkeypatch.setenv("HAI_LLM_FAILOVER_SECONDS", "25")
    assert auto_mode.create_llm_provider().providers[0].timeout == 25
    monkeypatch.setenv("OPENCODE_TIMEOUT", "15")
    assert auto_mode.create_llm_provider().providers[0].timeout == 15


def test_the_worst_case_total_fits_inside_the_pages_four_minute_wait(auto_mode, monkeypatch):
    monkeypatch.setenv("OPENCODE_TIMEOUT", "300")
    first, second = auto_mode.create_llm_provider().providers
    assert first.timeout + second.timeout < 4 * 60


def test_single_provider_modes_keep_their_configured_timeout(auto_mode, monkeypatch):
    monkeypatch.setenv("OPENCODE_TIMEOUT", "300")
    monkeypatch.setenv("HAI_LLM_PROVIDER", "opencode")
    assert auto_mode.create_llm_provider().timeout == 300

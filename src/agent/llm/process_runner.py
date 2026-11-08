from __future__ import annotations

"""Run an LLM command line tool safely.

Three things ``subprocess.run`` does not do for us:

* **A clean environment.** The server's environment holds this application's secrets (database
  passwords, Keycloak settings). An agent CLI can run shell commands, so those must not be
  inherited. Variables named ``HAI_*``/``KC_*`` and anything ending in ``_PASSWORD``, ``_PASSWD``
  or ``_SECRET`` are removed; everything else (PATH, HOME, the CLI's own login variables) is kept.
* **No inherited keyboard.** Without ``stdin=DEVNULL`` the child shares the server's console, so a
  CLI that asks a question waits forever for an answer nobody can type.
* **A real timeout.** ``subprocess.run(timeout=...)`` only kills the direct child; a grandchild that
  keeps the output pipes open can then block the caller long after the timeout. The whole process
  tree is killed instead, and the wait afterwards is bounded.
"""

import os
import signal
import subprocess
from typing import Mapping, Sequence

_DENY_PREFIXES = ("HAI_", "KC_")
_DENY_SUFFIXES = ("_PASSWORD", "_PASSWD", "_SECRET")
_AFTER_KILL_WAIT = 5.0


def sanitized_environment(base: Mapping[str, str] | None = None,
                          overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """``base`` (default: the current environment) without this application's secrets.

    ``overrides`` is applied afterwards and is never filtered: it is what the caller deliberately
    passes to the child.
    """
    source = os.environ if base is None else base
    kept = {
        name: value for name, value in source.items()
        if not name.upper().startswith(_DENY_PREFIXES) and not name.upper().endswith(_DENY_SUFFIXES)
    }
    if overrides:
        kept.update(overrides)
    return kept


def _kill_tree(process: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           stdin=subprocess.DEVNULL, timeout=10, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except Exception:  # the process may already be gone
        pass
    try:
        process.kill()
    except Exception:
        pass


def run_cli(command: Sequence[str], *, cwd: str | None, env: Mapping[str, str] | None,
            timeout: float | None, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    """Run ``command``; same result shape as ``subprocess.run(..., capture_output=True, text=True)``.

    Raises ``subprocess.TimeoutExpired`` after ``timeout`` seconds, having killed the whole process tree.
    """
    kwargs: dict = dict(
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", cwd=cwd, env=dict(env) if env is not None else None)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True      # lets us kill the whole group
    process = subprocess.Popen(list(command), **kwargs)
    try:
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=_AFTER_KILL_WAIT)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        raise subprocess.TimeoutExpired(list(command), timeout, output=stdout, stderr=stderr) from None
    except BaseException:
        _kill_tree(process)
        raise
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)


__all__ = ["run_cli", "sanitized_environment"]

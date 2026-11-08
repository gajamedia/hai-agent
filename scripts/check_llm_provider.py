"""Check the LLM providers the agent uses to plan reports, one by one, with timings.

    python scripts/check_llm_provider.py            # every provider in your HAI_LLM_PROVIDER setting
    python scripts/check_llm_provider.py --timeout 30

Planning a report asks an LLM for a short JSON plan. With HAI_LLM_PROVIDER=auto the first provider
(OpenCode) is tried and, only when it fails or times out, the second (Claude Code). A stuck first
provider therefore makes the page wait for its whole timeout (OPENCODE_TIMEOUT) before anything
happens. This tool sends one tiny prompt to each provider separately and tells you which answers,
how fast, or why it does not. It loads your .env first, like the server.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.llm.contracts import LLMMessage, LLMRequest  # noqa: E402
from api.dotenv_tools import load_env_file  # noqa: E402

PROMPT = "Reply with exactly the single word: pong"
SLOW_SECONDS = 20


def _provider_list(factory: Callable[[], Any]) -> list[Any]:
    provider = factory()
    return list(getattr(provider, "providers", (provider,)))


def check_providers(factory: Callable[[], Any], *, clock: Callable[[], float] = time.monotonic
                    ) -> tuple[bool, list[str]]:
    """Probe each provider on its own. Returns (at least one works, report lines)."""
    lines: list[str] = []
    working: list[str] = []
    try:
        providers = _provider_list(factory)
    except Exception as exc:  # configuration problems (unknown mode, CLI not found, ...)
        return False, [f"[FAIL] the LLM provider is not configured correctly: {type(exc).__name__}: {exc}"]
    for index, provider in enumerate(providers, 1):
        name = str(getattr(provider, "name", type(provider).__name__))
        started = clock()
        try:
            response = provider.complete(LLMRequest(messages=(LLMMessage("user", PROMPT),)))
        except Exception as exc:
            seconds = clock() - started
            lines.append(f"[FAIL] #{index} {name}: no answer after {seconds:.1f} s: {type(exc).__name__}: {exc}")
            continue
        seconds = clock() - started
        answer = (response.content or "").strip().replace("\n", " ")[:40]
        slow = f"  <- slow: every report plan waits this long first" if (index == 1 and seconds > SLOW_SECONDS) else ""
        lines.append(f"[ok]   #{index} {name}: answered in {seconds:.1f} s ({answer!r}){slow}")
        working.append(name)
    if not working:
        lines.append("\nNo provider answered, so the page cannot create plans.")
    elif len(providers) > 1 and str(getattr(providers[0], "name", "")) not in working:
        lines.append(f"\nThe first provider fails, so every plan first waits for its timeout and then uses "
                     f"{working[0]}. Faster: HAI_LLM_PROVIDER={working[0]} in .env (and restart the backend), "
                     f"or lower the first provider's timeout.")
    return bool(working), lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=["auto", "opencode", "claude-code"],
                        help="test this setting instead of HAI_LLM_PROVIDER")
    parser.add_argument("--timeout", type=float, default=60,
                        help="seconds to wait for each provider in this check (default 60)")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    args = parser.parse_args(argv)
    loaded = load_env_file(Path(args.env_file))
    if loaded:
        print(f"[ok]   loaded {len(loaded)} setting(s) from {args.env_file}")
    if args.provider:
        os.environ["HAI_LLM_PROVIDER"] = args.provider
    configured = {n: os.environ.get(n, "") for n in ("OPENCODE_TIMEOUT", "CLAUDE_CODE_TIMEOUT")}
    print(f"       HAI_LLM_PROVIDER={os.environ.get('HAI_LLM_PROVIDER', 'auto')}, "
          f"OPENCODE_TIMEOUT={configured['OPENCODE_TIMEOUT'] or '120 (default)'}, "
          f"CLAUDE_CODE_TIMEOUT={configured['CLAUDE_CODE_TIMEOUT'] or '120 (default)'}")
    # The check itself should not hang for the full production timeout.
    for name in ("OPENCODE_TIMEOUT", "CLAUDE_CODE_TIMEOUT"):
        os.environ[name] = str(min(float(os.environ.get(name) or 120), args.timeout))
    from api.dependencies import create_llm_provider
    ok, lines = check_providers(create_llm_provider)
    print("\n".join(lines))
    print("\nResult: " + ("at least one provider works." if ok else "fix the [FAIL] lines above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

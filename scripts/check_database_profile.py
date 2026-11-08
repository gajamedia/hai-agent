"""Check a database profile from your own terminal and see the real reason when it fails.

    python scripts/check_database_profile.py local_mysql

It loads your .env first (like ``uvicorn --env-file .env``), so it sees what the server sees.

The web API deliberately hides driver error details from browsers. This tool runs the very
same code path the server uses (same read-only session, same SELECT-only account check,
same schema discovery) but prints the actual cause on *your* console. It never prints the
password. Profiles come from the JSON file named by HAI_DATABASE_PROFILES (or --profiles);
the password comes from the environment variable that the profile names.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.database_engines import EngineError, load_profiles  # noqa: E402
from api.database_profile_check import _hint, check_profile  # noqa: E402,F401
from api.dotenv_tools import load_env_file  # noqa: E402,F401

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", help="profile name from the profiles file")
    parser.add_argument("--profiles", default="",
                        help="path to the profiles JSON (default: HAI_DATABASE_PROFILES)")
    parser.add_argument("--env-file", default=str(ROOT / ".env"),
                        help="settings file to load first, like the server does (default: .env)")
    args = parser.parse_args(argv)
    loaded = load_env_file(Path(args.env_file))
    if loaded:
        print(f"[ok]   loaded {len(loaded)} setting(s) from {args.env_file}")
    args.profiles = args.profiles or os.getenv("HAI_DATABASE_PROFILES", "")
    if not args.profiles:
        print("[FAIL] no profiles file: set HAI_DATABASE_PROFILES or pass --profiles <path>", file=sys.stderr)
        return 1
    try:
        profiles = load_profiles(args.profiles)
    except EngineError as exc:
        print(f"[FAIL] the profiles file is not valid: {exc}", file=sys.stderr)
        return 1
    ok, lines = check_profile(args.profile, profiles)
    print("\n".join(lines))
    print("\nResult: " + ("this profile will work in the web app." if ok else "fix the [FAIL] line above, then run this again."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

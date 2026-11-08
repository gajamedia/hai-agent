from __future__ import annotations

"""Register a database server from a form, without running a setup script.

The form carries only LIMITED accounts: a read-only account, and optionally an account that may create
databases (prefix ``hai_``) and one that may insert/update/delete. An administrator account (sa, root, ...)
is never needed here and is refused by the engines' own safety checks. Creating those limited accounts is
still done once by an administrator with the scripts in ``scripts/``, in a terminal that never stores that
password.

What happens, in order: the fields are validated exactly like a hand-written profile; the connection is
tested with the very code the server uses (read-only session, SELECT-only account check); only if that
works is the profile written to the profiles file and each password to ``.env`` under a variable name
derived from the profile name. The profiles file never holds a password, and no password is ever returned.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from api.database_engines import EngineError, parse_profiles
from api.database_profile_check import check_profile
from api.dotenv_tools import set_env_value

KINDS = ("mysql", "sqlserver", "postgresql")
# Stored when no database was given. It names no real database, so connecting the bare profile fails
# with "unknown database"; databases the agent creates are reached as profile/<hai_name> instead.
NO_DATABASE = "no_base_database"
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_UNSAFE = re.compile(r"[\s#\"'$\\\x00]")
_TOP = {"name", "type", "host", "port", "database", "user", "password", "ssl", "creator", "writer"}


class ServerSetupError(ValueError):
    """The request is not acceptable; the message is safe to show to the user."""


def _text(value: Any, label: str, *, required: bool = True) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ServerSetupError(f"{label} is required")
        return ""
    if not isinstance(value, str) or len(value) > 200:
        raise ServerSetupError(f"{label} is not valid")
    return value.strip()


def _password(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ServerSetupError(f"{label} is required")
    if len(value) > 200 or _UNSAFE.search(value):
        raise ServerSetupError(
            f"{label} contains a character that a .env file cannot hold (space, # \" ' $ or backslash); "
            "give that account a password without them")
    return value


def _account(raw: Any, label: str, env_name: str, *, prefix: bool = False) -> tuple[dict[str, Any], str]:
    if not isinstance(raw, Mapping) or set(raw) - {"user", "password", "prefix"}:
        raise ServerSetupError(f"{label} is not valid")
    entry: dict[str, Any] = {"user": _text(raw.get("user"), f"{label} user"), "password_env": env_name}
    if prefix and raw.get("prefix"):
        entry["prefix"] = _text(raw.get("prefix"), f"{label} prefix")
    return entry, _password(raw.get("password"), f"{label} password")


def build_entry(spec: Any) -> tuple[str, dict[str, Any], dict[str, str]]:
    """(profile name, profiles.json entry, {environment variable: password}); raises ServerSetupError."""
    if not isinstance(spec, Mapping) or set(spec) - _TOP:
        raise ServerSetupError("unsupported fields in the request")
    name = _text(spec.get("name"), "name")
    if not _NAME.match(name):
        raise ServerSetupError("name: lowercase letters, digits or _ (2-32 characters, starts with a letter)")
    kind = spec.get("type")
    if kind not in KINDS:
        raise ServerSetupError(f"type must be one of {', '.join(KINDS)}")
    base = f"HAI_DB_{name.upper()}"
    secrets: dict[str, str] = {}
    database = _text(spec.get("database"), "database", required=False)
    if not database and not (isinstance(spec.get("creator"), Mapping) and spec["creator"].get("user")):
        raise ServerSetupError("database is required unless a creator account is given "
                               "(a server used only to create databases needs no existing database)")
    entry: dict[str, Any] = {"type": kind,
                             "server" if kind == "sqlserver" else "host": _text(spec.get("host"), "host"),
                             "database": database or NO_DATABASE,
                             "user": _text(spec.get("user"), "user"),
                             "password_env": f"{base}_PASSWORD"}
    secrets[entry["password_env"]] = _password(spec.get("password"), "password")
    port = spec.get("port")
    if port not in (None, ""):
        try:
            entry["port"] = int(port)
        except (TypeError, ValueError):
            raise ServerSetupError("port must be a number") from None
    if spec.get("ssl"):
        entry["ssl"] = _text(spec.get("ssl"), "ssl")
    for key, prefix in (("creator", True), ("writer", False)):
        if spec.get(key):
            if kind == "postgresql":
                raise ServerSetupError("PostgreSQL is read-only here: leave out the creator and writer accounts")
            account, password = _account(spec[key], key, f"{base}_{key.upper()}_PASSWORD", prefix=prefix)
            entry[key] = account
            secrets[account["password_env"]] = password
    return name, entry, secrets


def _failure(lines: list[str]) -> str:
    """What to tell the user: the safety-rule or driver message, or the hint; never the driver's raw text."""
    parts = []
    for line in lines:
        if not line.startswith(("[FAIL]", "       hint:")):
            continue
        if "could not connect:" in line:
            continue                       # raw driver text can name hosts and users; the hint line follows
        parts.append(line.replace("[FAIL]", "").replace("hint:", "").strip())
    return " ".join(parts) or "could not connect: check host, port, user and password"


def _check_creator(profile: Any) -> None:
    """No database to read yet: prove the server and the creator account instead (same checks as creating)."""
    from api.database_engines import MySQLEngine, SQLServerEngine
    from api.database_profile_check import _hint
    engine = {"mysql": MySQLEngine, "sqlserver": SQLServerEngine}[profile.type]({profile.name: profile})
    try:
        engine.open_creator(profile.name)
    except ImportError:
        raise ServerSetupError("the database driver is not installed on the server") from None
    except EngineError as exc:
        raise ServerSetupError(f"the creator account was refused: {exc}") from exc
    except Exception as exc:  # the driver's own text can name hosts and users: only the hint is shown
        raise ServerSetupError(_hint(exc) or "could not connect: check host, port, user and password") from None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=".profiles-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
    except ValueError as exc:
        raise ServerSetupError(f"the profiles file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ServerSetupError("the profiles file must be a JSON object")
    return data


def register(spec: Any, *, profiles_path: Path, env_path: Path, taken: set[str]) -> Any:
    """Validate, test and save one server; returns the parsed profile. Nothing is saved unless the test passes."""
    name, entry, secrets = build_entry(spec)
    if name in taken:
        raise ServerSetupError(f"a server named '{name}' already exists")
    try:
        profile = parse_profiles({name: entry})[name]
    except EngineError as exc:
        raise ServerSetupError(str(exc)) from exc
    previous = {key: os.environ.get(key) for key in secrets}
    os.environ.update(secrets)
    try:
        if profile.database == NO_DATABASE:
            _check_creator(profile)
        else:
            ok, lines = check_profile(name, {name: profile})
            if not ok:
                raise ServerSetupError(_failure(lines))
        stored = _read_json(profiles_path)
        if name in stored:
            raise ServerSetupError(f"a server named '{name}' already exists")
        for key, value in secrets.items():
            set_env_value(env_path, key, value)
        stored[name] = entry
        _write_json(profiles_path, stored)
    except BaseException:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        raise
    return profile


def forget(name: str, *, profiles_path: Path) -> None:
    """Remove a profile from the profiles file (its .env variables are left for the administrator)."""
    stored = _read_json(profiles_path)
    if name not in stored:
        raise ServerSetupError(f"'{name}' is not in the profiles file")
    del stored[name]
    _write_json(profiles_path, stored)


__all__ = ["KINDS", "ServerSetupError", "build_entry", "register", "forget"]

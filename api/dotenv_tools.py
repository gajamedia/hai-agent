from __future__ import annotations

"""Small, safe helpers for a ``.env`` file: load it like ``uvicorn --env-file`` and set one value.

Used by the command line tools in ``scripts/``. Nothing here prints a value.
"""

import os
import re
import tempfile
from pathlib import Path

_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Characters that a .env parser treats specially in an unquoted value. Refusing them keeps
# "what we wrote" identical to "what the server later reads".
_UNSAFE = re.compile(r"[\s#\"'$\\\x00]")


def load_env_file(path: Path) -> list[str]:
    """Load KEY=VALUE pairs the way ``uvicorn --env-file`` does; the terminal's own values win.

    Returns the NAMES that were loaded (never the values).
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return []
    if not Path(path).is_file():
        return []
    loaded = []
    for key, value in dotenv_values(path).items():
        if value is not None and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def set_env_value(path: Path, name: str, value: str) -> bool:
    """Set ``NAME=value`` in the file, keeping every other line and the line-ending style.

    Replaces an existing active line, or appends one. Written as UTF-8 without a BOM through a
    temporary file, so an interruption cannot leave a half-written ``.env``. Returns True when
    an existing line was replaced and False when a new one was appended.
    """
    if not _NAME.match(name):
        raise ValueError("invalid variable name")
    if not isinstance(value, str) or _UNSAFE.search(value):
        raise ValueError("value contains characters a .env file cannot hold unquoted "
                         "(whitespace, # \" ' $ \\)")
    target = Path(path)
    raw = target.read_bytes() if target.exists() else b""
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    pattern = re.compile(rf"(?m)^{re.escape(name)}=.*?(?=\r?$)")
    replaced = bool(pattern.search(text))
    if replaced:
        text = pattern.sub(lambda _m: f"{name}={value}", text, count=1)
    else:
        if text and not text.endswith(("\n", "\r")):
            text += newline
        text += f"{name}={value}{newline}"
    fd, temp = tempfile.mkstemp(dir=str(target.parent or Path(".")), prefix=".env-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return replaced


__all__ = ["load_env_file", "set_env_value"]

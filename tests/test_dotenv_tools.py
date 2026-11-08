from __future__ import annotations

import pytest

from api.dotenv_tools import set_env_value


def _write(path, text: str, newline: str = "\r\n"):
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def _read(path) -> str:
    return path.read_bytes().decode("utf-8")


def test_an_existing_empty_line_is_replaced_and_everything_else_is_untouched(tmp_path):
    env = tmp_path / ".env"
    _write(env, "A=1\nHAI_MYSQL_LOCAL_PASSWORD=\nB=2\n")
    assert set_env_value(env, "HAI_MYSQL_LOCAL_PASSWORD", "Secret123") is True
    assert _read(env) == "A=1\r\nHAI_MYSQL_LOCAL_PASSWORD=Secret123\r\nB=2\r\n"


def test_an_existing_value_is_overwritten_not_duplicated(tmp_path):
    env = tmp_path / ".env"
    _write(env, "X=old\nY=keep\n")
    set_env_value(env, "X", "new")
    set_env_value(env, "X", "newer")
    assert _read(env).count("X=") == 1 and "X=newer" in _read(env) and "Y=keep" in _read(env)


def test_a_missing_variable_is_appended_in_the_files_line_ending_style(tmp_path):
    crlf, lf = tmp_path / "crlf.env", tmp_path / "lf.env"
    _write(crlf, "A=1\n", "\r\n")
    _write(lf, "A=1\n", "\n")
    assert set_env_value(crlf, "NEW", "v") is False and set_env_value(lf, "NEW", "v") is False
    assert _read(crlf) == "A=1\r\nNEW=v\r\n"
    assert _read(lf) == "A=1\nNEW=v\n"


def test_a_file_without_a_trailing_newline_is_handled(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"A=1")
    set_env_value(env, "B", "2")
    assert _read(env) == "A=1\nB=2\n"


def test_a_missing_file_is_created(tmp_path):
    env = tmp_path / "fresh.env"
    set_env_value(env, "A", "1")
    assert _read(env) == "A=1\n"


def test_a_commented_out_line_is_left_alone_and_a_real_line_is_added(tmp_path):
    env = tmp_path / ".env"
    _write(env, "#HAI_X=commented\n")
    assert set_env_value(env, "HAI_X", "real") is False
    assert _read(env) == "#HAI_X=commented\r\nHAI_X=real\r\n"


def test_a_similar_name_is_never_touched(tmp_path):
    env = tmp_path / ".env"
    _write(env, "HAI_X_LONG=a\nHAI_X=b\n")
    set_env_value(env, "HAI_X", "c")
    assert _read(env) == "HAI_X_LONG=a\r\nHAI_X=c\r\n"


def test_a_bom_is_dropped_and_no_bom_is_written(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfA=1\r\n")
    set_env_value(env, "A", "2")
    assert not env.read_bytes().startswith(b"\xef\xbb\xbf") and _read(env) == "A=2\r\n"


@pytest.mark.parametrize("value", ["has space", "a#b", 'q"uote', "q'uote", "dollar$x", "back\\slash",
                                   "new\nline", "nul\x00", ""[:0] + " "])
def test_values_a_dotenv_parser_could_mangle_are_refused(tmp_path, value):
    env = tmp_path / ".env"
    _write(env, "A=1\n")
    with pytest.raises(ValueError):
        set_env_value(env, "A", value)
    assert _read(env) == "A=1\r\n"          # nothing was written


@pytest.mark.parametrize("name", ["lower", "1ABC", "A B", "A=B", "", "A\nB"])
def test_invalid_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        set_env_value(tmp_path / ".env", name, "v")


def test_no_temporary_file_is_left_behind(tmp_path):
    env = tmp_path / ".env"
    _write(env, "A=1\n")
    set_env_value(env, "A", "2")
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


def test_what_is_written_is_what_a_dotenv_parser_reads_back(tmp_path):
    dotenv = pytest.importorskip("dotenv")
    env = tmp_path / ".env"
    _write(env, "A=1\nHAI_MYSQL_LOCAL_PASSWORD=\nB=2\n")
    set_env_value(env, "HAI_MYSQL_LOCAL_PASSWORD", "aB3dE6gH9jK2mN5pQ8sT1vW4")
    values = dotenv.dotenv_values(env)
    assert values["HAI_MYSQL_LOCAL_PASSWORD"] == "aB3dE6gH9jK2mN5pQ8sT1vW4"
    assert values["A"] == "1" and values["B"] == "2"

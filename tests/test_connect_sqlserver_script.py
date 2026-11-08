from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "connect_sqlserver.py"
spec = importlib.util.spec_from_file_location("connect_sqlserver", SCRIPT)
tool = importlib.util.module_from_spec(spec)
sys.modules["connect_sqlserver"] = tool
spec.loader.exec_module(tool)

INSTANCE = {"name": "SQLEXPRESS", "folder": "MSSQL11.SQLEXPRESS", "tcp_enabled": False, "tcp_port": None, "login_mode": 2}


class FakeCursor:
    def __init__(self, db):
        self.db, self.row, self.rows = db, None, []

    def execute(self, sql, *params):
        self.db.executed.append((self.db.database, sql))
        if sql.startswith("SELECT CAST(SERVERPROPERTY"):
            self.row = ("11.0.3128.0", "Express Edition", self.db.windows_only, self.db.admin, "PC\me")
        elif sql.startswith("SELECT name FROM sys.databases"):
            self.rows = [("dbsales",)]
        elif sql.startswith("SELECT permission_name"):
            self.rows = [("CONNECT SQL",), ("CREATE ANY DATABASE",)]
        elif sql.startswith("SELECT CAST(IS_SRVROLEMEMBER"):
            self.rows = [(0,) * 8]
        elif sql.startswith("SELECT COUNT(*)"):
            self.row = (3,)
        elif sql.startswith("CREATE TABLE") and self.db.read_only:
            raise RuntimeError("CREATE TABLE permission denied")

    def close(self):
        pass

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, shared, database, read_only=False):
        self.__dict__.update(shared)
        self.database, self.read_only, self.autocommit, self.closed = database, read_only, False, False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


class FakePyodbc:
    def __init__(self, drivers=("ODBC Driver 17 for SQL Server",), fail=False, **shared):
        self._drivers, self.fail = list(drivers), fail
        self.shared = {"executed": [], "windows_only": 0, "admin": 1, **shared}
        self.strings = []

    def drivers(self):
        return self._drivers

    def connect(self, text, timeout=0):
        self.strings.append(text)
        if self.fail:
            raise RuntimeError("login timeout")
        database = text.split("DATABASE=")[1].split(";")[0]
        return FakeConnection(self.shared, database, read_only="UID=" in text)


def run(tmp_path, pyodbc, argv=(), answers=("y",), instances=(INSTANCE,)):
    lines, replies = [], iter(answers)
    code = tool.main(["--env", str(tmp_path / ".env"), "--profiles", str(tmp_path / "profiles.json"), *argv],
                     pyodbc_module=pyodbc, instances_reader=lambda: list(instances),
                     service_state=lambda name: "running", ask=lambda p: next(replies), out=lines.append)
    return code, "\n".join(lines)


def test_diagnose_reports_the_instance_the_login_and_the_databases_and_changes_nothing(tmp_path):
    fake = FakePyodbc()
    code, text = run(tmp_path, fake)
    assert code == 0 and "SQLEXPRESS: service running; TCP/IP off" in text and "Windows or SQL login" in text
    assert "an administrator" in text and "dbsales" in text
    assert "Trusted_Connection=yes" in fake.strings[0] and r"SERVER=(local)\SQLEXPRESS" in fake.strings[0]
    assert all(not sql.startswith(("CREATE", "ALTER")) for _, sql in fake.shared["executed"])
    assert not (tmp_path / ".env").exists() and not (tmp_path / "profiles.json").exists()


def test_a_missing_driver_or_unreachable_server_is_explained(tmp_path):
    assert "No 'ODBC Driver" in run(tmp_path, FakePyodbc(drivers=("SQL Server",)))[1]
    code, text = run(tmp_path, FakePyodbc(fail=True))
    assert code == 1 and "could not connect: login timeout" in text


def test_create_reader_makes_a_read_only_account_and_keeps_the_password_out_of_the_details(tmp_path):
    fake = FakePyodbc()
    code, text = run(tmp_path, fake, ["--create-reader", "--database", "dbsales"])
    assert code == 0 and "a write attempt was refused" in text
    sql = [s for _, s in fake.shared["executed"]]
    assert any("CREATE LOGIN [hai_sql_reader]" in s for s in sql)
    assert any("db_datareader" in s for s in sql) and any("db_denydatawriter" in s for s in sql)
    assert any(db == "dbsales" and "CREATE USER" in s for db, s in fake.shared["executed"])
    env = (tmp_path / ".env").read_text("utf-8")
    password = env.split("HAI_LOCAL_SQLSERVER_PASSWORD=")[1].strip()
    assert len(password) == 24
    details = (tmp_path / "profiles.json").read_text("utf-8")
    assert password not in details and json.loads(details)["local_sqlserver"]["user"] == "hai_sql_reader"
    assert password not in text
    # the app accepts exactly what the script wrote
    from api.database_engines import parse_profiles
    assert parse_profiles(json.loads(details))["local_sqlserver"].host == r"(local)\SQLEXPRESS"


def test_declining_changes_nothing(tmp_path):
    fake = FakePyodbc()
    code, text = run(tmp_path, fake, ["--create-reader", "--database", "dbsales"], answers=("n",))
    assert code == 1 and "Nothing changed" in text
    assert not any(s.startswith(("CREATE", "ALTER", "IF")) for _, s in fake.shared["executed"])


@pytest.mark.parametrize("shared,argv,needle", [
    ({"windows_only": 1}, ["--database", "dbsales"], "Windows logins only"),
    ({"admin": 0}, ["--database", "dbsales"], "not an administrator"),
    ({}, ["--database", "other"], "not found"),
    ({}, ["--database", "dbsales; DROP"], "not found"),
    ({}, ["--database", "dbsales", "--user", "x]; DROP LOGIN sa"], "letters, digits"),
    ({}, ["--database", "dbsales", "--profile", "Bad Name"], "profile"),
])
def test_refusals_say_why_and_run_no_changes(tmp_path, shared, argv, needle):
    fake = FakePyodbc(**shared)
    code, text = run(tmp_path, fake, ["--create-reader", *argv])
    assert code != 0 and needle in text
    assert not any(s.startswith(("CREATE", "ALTER", "IF")) for _, s in fake.shared["executed"])


def test_the_statements_quote_every_name_and_use_a_password_that_cannot_break_out():
    plan = tool.plan_statements("hai_sql_reader", "dbsales", tool.generate_password())
    assert all(s.count("[") == s.count("]") for sqls in plan.values() for s in sqls)
    with pytest.raises(tool.SetupError):
        tool.plan_statements("a b", "dbsales", "x")
    assert tool.pick_driver(["ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server"]).endswith("18 for SQL Server")


def test_the_registry_is_read_into_instances():
    class Key:
        def __init__(self, path): self.path = path
        def __enter__(self): return self
        def __exit__(self, *a): return False

    values = {
        r"Instance Names\SQL": {"SQLEXPRESS": "MSSQL11.SQLEXPRESS"},
        r"MSSQL11.SQLEXPRESS\MSSQLServer": {"LoginMode": 2},
        r"MSSQL11.SQLEXPRESS\MSSQLServer\SuperSocketNetLib\Tcp": {"Enabled": 1},
        r"MSSQL11.SQLEXPRESS\MSSQLServer\SuperSocketNetLib\Tcp\IPAll": {"TcpPort": "1433", "TcpDynamicPorts": ""},
    }

    class W:
        HKEY_LOCAL_MACHINE = 0
        @staticmethod
        def OpenKey(_, path):
            rel = path.split("Microsoft SQL Server\\", 1)[1]
            if rel not in values:
                raise OSError
            return Key(rel)
        @staticmethod
        def QueryValueEx(key, name):
            if name not in values[key.path]:
                raise OSError
            return values[key.path][name], 1
        @staticmethod
        def EnumValue(key, index):
            items = list(values[key.path].items())
            if index >= len(items):
                raise OSError
            return items[index][0], items[index][1], 1

    got = tool.read_instances(W)
    assert got == [{"name": "SQLEXPRESS", "folder": "MSSQL11.SQLEXPRESS", "tcp_enabled": True,
                    "tcp_port": "1433", "login_mode": 2}]


def _profile_file(tmp_path, **extra):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"local_sqlserver": {
        "type": "sqlserver", "server": r"(local)\SQLEXPRESS", "database": "dbsales", "user": "hai_sql_reader",
        "password_env": "HAI_LOCAL_SQLSERVER_PASSWORD", "ssl": "require", **extra}}), encoding="utf-8")
    return path


def test_create_creator_grants_one_permission_and_adds_a_creator_block(tmp_path):
    path = _profile_file(tmp_path)
    fake = FakePyodbc()
    code, text = run(tmp_path, fake, ["--create-creator"])
    assert code == 0 and "holds nothing else" in text
    sql = [s for _, s in fake.shared["executed"]]
    assert any("CREATE LOGIN [hai_sql_creator]" in s for s in sql)
    assert any(s == "GRANT CREATE ANY DATABASE TO [hai_sql_creator]" for s in sql)
    assert not any(("dbcreator" in s or "sysadmin" in s) for s in sql if not s.startswith("SELECT"))   # no role is granted
    block = json.loads(path.read_text("utf-8"))["local_sqlserver"]["creator"]
    assert block == {"user": "hai_sql_creator", "password_env": "HAI_LOCAL_SQLSERVER_CREATOR_PASSWORD",
                     "prefix": "hai_"}
    env = (tmp_path / ".env").read_text("utf-8")
    password = env.split("HAI_LOCAL_SQLSERVER_CREATOR_PASSWORD=")[1].strip()
    assert password not in path.read_text("utf-8") and password not in text
    from api.database_engines import parse_profiles
    assert parse_profiles(json.loads(path.read_text("utf-8")))["local_sqlserver"].creator_user == "hai_sql_creator"


@pytest.mark.parametrize("shared,answers,extra_argv,needle", [
    ({"windows_only": 1}, ("y",), [], "Windows logins only"),
    ({"admin": 0}, ("y",), [], "not an administrator"),
    ({}, ("n",), [], "Nothing changed"),
    ({}, ("y",), ["--creator-user", "hai_sql_reader"], "different login"),
    ({}, ("y",), ["--creator-user", "x]; DROP"], "letters, digits"),
    ({}, ("y",), ["--profile", "nope_such"], "not a SQL Server profile"),
])
def test_create_creator_refuses_or_stops_without_changes(tmp_path, shared, answers, extra_argv, needle):
    _profile_file(tmp_path)
    fake = FakePyodbc(**shared)
    code, text = run(tmp_path, fake, ["--create-creator", *extra_argv], answers=answers)
    assert code != 0 and needle in text
    assert not any(s.startswith(("CREATE", "GRANT", "IF", "ALTER")) for _, s in fake.shared["executed"])
    assert not (tmp_path / ".env").exists()


def test_the_reader_plan_names_only_roles_that_exist_in_sql_server():
    """db_denyddladmin does not exist; a fake connection let that through once."""
    import re
    fixed = {"db_accessadmin", "db_backupoperator", "db_datareader", "db_datawriter", "db_ddladmin",
             "db_denydatareader", "db_denydatawriter", "db_owner", "db_securityadmin"}
    plan = tool.plan_statements("hai_sql_reader", "dbsales", "x")
    used = {m for sqls in plan.values() for s in sqls for m in re.findall(r"\[(db_[a-z]+)\]", s)}
    assert used == {"db_datareader", "db_denydatawriter"} and used <= fixed


# ------------------------------------------------------------- a server on another computer
def run_remote(tmp_path, pyodbc, argv, answers=("y",), secret="Adm}in;pw=1"):
    lines, replies = [], iter(answers)
    code = tool.main(["--env", str(tmp_path / ".env"), "--profiles", str(tmp_path / "profiles.json"), *argv],
                     pyodbc_module=pyodbc, instances_reader=lambda: [], service_state=lambda n: "running",
                     ask=lambda p: next(replies), ask_secret=lambda p: secret, out=lines.append)
    return code, chr(10).join(lines)


def test_a_remote_server_is_reached_with_an_admin_login_whose_password_is_asked_and_kept_out_of_everything(tmp_path):
    fake = FakePyodbc()
    code, text = run_remote(tmp_path, fake, ["--server", "10.62.10.34,1433", "--admin-user", "sa"])
    assert code == 0 and "Connecting to 10.62.10.34,1433 as sa" in text and "dbsales" in text
    first = fake.strings[0]
    assert "SERVER=10.62.10.34,1433;" in first and "UID=sa;" in first and "Trusted_Connection" not in first
    assert "PWD={Adm}}in;pw=1}" in first                           # braces: the password cannot add keywords
    assert "Adm}in" not in text and not (tmp_path / ".env").exists()


def test_create_reader_on_a_remote_server_writes_host_and_port_apart_and_the_app_accepts_it(tmp_path):
    fake = FakePyodbc()
    code, text = run_remote(tmp_path, fake, ["--server", "10.62.10.34,1433", "--admin-user", "sa",
                                             "--create-reader", "--database", "dbsales"])
    assert code == 0
    saved = json.loads((tmp_path / "profiles.json").read_text("utf-8"))["local_sqlserver"]
    assert (saved["server"], saved["port"], saved["ssl"]) == ("10.62.10.34", 1433, "require")
    from api.database_engines import parse_profiles
    profile = parse_profiles({"x1": saved})["x1"]
    assert (profile.host, profile.port) == ("10.62.10.34", 1433)
    assert "Adm}in" not in (tmp_path / ".env").read_text("utf-8") + text


@pytest.mark.parametrize("server,expected", [
    ("10.62.10.34,1433", ("10.62.10.34", 1433)), ("db.example.com", ("db.example.com", 0)),
    (r"(local)\SQLEXPRESS", (r"(local)\SQLEXPRESS", 0)), (r"10.1.1.1\SQL2019,50123", (r"10.1.1.1\SQL2019", 50123)),
])
def test_a_server_string_is_split_into_host_and_port(server, expected):
    assert tool.split_server(server) == expected


@pytest.mark.parametrize("server", ["10.0.0.1,abc", "10.0.0.1,0", "10.0.0.1,70000", "10.0.0.1,"])
def test_a_bad_port_is_refused(server):
    with pytest.raises(tool.SetupError):
        tool.split_server(server)


def test_a_server_name_that_cannot_be_a_profile_is_refused_before_anything_is_written(tmp_path):
    fake = FakePyodbc()
    code, text = run_remote(tmp_path, fake, ["--server", "db;Trusted_Connection=yes", "--admin-user", "sa",
                                             "--create-reader", "--database", "dbsales"])
    assert code != 0 and not (tmp_path / "profiles.json").exists()

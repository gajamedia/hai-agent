"""Tests for passive database discovery."""

from agent.database import DatabaseDetector


def test_driver_detection_reports_expected_modules(monkeypatch):
    availability = {
        "sqlite3": True,
        "psycopg": False,
        "psycopg2": False,
        "pymysql": False,
        "mysql.connector": False,
        "pyodbc": False,
        "oracledb": False,
        "cx_Oracle": False,
        "pymongo": False,
        "duckdb": False,
        "sqlalchemy": False,
        "aiosqlite": False,
    }
    monkeypatch.setattr(
        "agent.database.database_detector.importlib.util.find_spec",
        lambda name: object() if availability.get(name, False) else None,
    )

    report = DatabaseDetector().detect(
        include_services=False,
        include_docker_containers=False,
    )

    by_module = {item.module: item.available for item in report.drivers}
    assert by_module == availability
    assert report.services == ()


def test_windows_service_parser_finds_mysql_and_postgresql():
    output = """
SERVICE_NAME: MySQL80
        TYPE               : 10  WIN32_OWN_PROCESS
        STATE              : 4  RUNNING
SERVICE_NAME: postgresql-x64-17
        TYPE               : 10  WIN32_OWN_PROCESS
        STATE              : 1  STOPPED
SERVICE_NAME: Spooler
        STATE              : 4  RUNNING
"""
    services = DatabaseDetector._parse_windows_services(output)

    assert len(services) == 2
    assert services[0].database == "mysql"
    assert services[0].running is True
    assert services[1].database == "postgresql"
    assert services[1].running is False


def test_docker_parser_classifies_running_database_containers(monkeypatch):
    from subprocess import CompletedProcess

    monkeypatch.setattr(
        "agent.database.database_detector.shutil.which",
        lambda name: "docker.exe" if name == "docker" else None,
    )
    monkeypatch.setattr(
        "agent.database.database_detector.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(
            args[0],
            0,
            stdout=(
                "nsmbakery_postgres\tpostgres:16\tUp 2 hours\n"
                "hai_agent_keycloak_db\tpostgres:17\tUp 2 hours (healthy)\n"
                "redis_cache\tredis:7\tUp 2 hours\n"
            ),
            stderr="",
        ),
    )

    warnings = []
    services = DatabaseDetector._detect_docker_services(warnings)

    assert len(services) == 2
    assert all(item.database == "postgresql" for item in services)
    assert all(item.running for item in services)
    assert warnings == []


def test_detection_report_is_json_serializable(monkeypatch):
    import json

    monkeypatch.setattr(
        "agent.database.database_detector.shutil.which",
        lambda _name: None,
    )
    report = DatabaseDetector().detect()

    decoded = json.loads(json.dumps(report.to_dict()))
    assert decoded["detected_at"]
    assert isinstance(decoded["drivers"], list)
    assert isinstance(decoded["services"], list)
    assert decoded["docker_available"] is False


def test_database_name_classification():
    classify = DatabaseDetector._database_from_text

    assert classify("postgresql-x64-17") == "postgresql"
    assert classify("MySQL80") == "mysql"
    assert classify("mariadb") == "mariadb"
    assert classify("duckdb_service") == "duckdb"
    assert classify("MongoDB") == "mongodb"
    assert classify("MSSQLSERVER") == "sqlserver"
    assert classify("OracleServiceXE") == "oracle"
    assert classify("unknown_service") == "unknown"

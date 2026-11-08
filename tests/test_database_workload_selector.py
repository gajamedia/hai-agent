"""Tests for workload classification and conservative database recommendations."""

from agent.database import (
    DatabaseDetectionReport,
    DatabaseDriverStatus,
    DatabaseSelector,
    DatabaseServiceStatus,
    WorkloadAnalyzer,
    WorkloadRequest,
    WorkloadType,
)


def make_detection(drivers, services):
    return DatabaseDetectionReport(
        detected_at="2026-01-01T00:00:00+00:00",
        drivers=tuple(
            DatabaseDriverStatus(database=db, module=module, available=available)
            for db, module, available in drivers
        ),
        services=tuple(services),
        docker_available=True,
        warnings=(),
    )


def test_explicit_workload_type_takes_precedence():
    request = WorkloadRequest(
        description="monthly sales report",
        workload_type=WorkloadType.INTERNAL_STATE,
    )
    analysis = WorkloadAnalyzer().analyze(request)

    assert analysis.workload_type == WorkloadType.INTERNAL_STATE
    assert analysis.confidence == 1.0


def test_analytics_workload_is_classified():
    analysis = WorkloadAnalyzer().analyze(
        WorkloadRequest(description="Build analytics dashboard from historical data")
    )

    assert analysis.workload_type == WorkloadType.ANALYTICS
    assert analysis.confidence > 0


def test_ambiguous_task_is_not_guessed():
    analysis = WorkloadAnalyzer().analyze(
        WorkloadRequest(description="Please help me with my project")
    )

    assert analysis.workload_type == WorkloadType.UNKNOWN
    assert analysis.confidence == 0.0


def test_internal_state_recommends_sqlite_when_driver_available():
    detection = make_detection(
        drivers=[("sqlite", "sqlite3", True)],
        services=[],
    )
    request = WorkloadRequest(
        description="save agent memory",
        workload_type=WorkloadType.INTERNAL_STATE,
    )
    analysis = WorkloadAnalyzer().analyze(request)
    result = DatabaseSelector().recommend(request, analysis, detection)

    assert result.recommended_database == "sqlite"
    assert result.eligible is True
    assert result.requires_user_approval is True


def test_transactional_database_blocked_when_driver_missing():
    detection = make_detection(
        drivers=[
            ("mysql", "pymysql", False),
            ("postgresql", "psycopg", False),
        ],
        services=[
            DatabaseServiceStatus("mysql", "windows_service", "MySQL80", True),
            DatabaseServiceStatus("postgresql", "docker_container", "nsmbakery_postgres", True),
        ],
    )
    request = WorkloadRequest(
        description="read customer orders",
        workload_type=WorkloadType.TRANSACTIONAL_READ,
    )
    analysis = WorkloadAnalyzer().analyze(request)
    result = DatabaseSelector().recommend(request, analysis, detection)

    assert result.recommended_database is None
    assert result.eligible is False
    assert any("driver" in item.lower() for item in result.blockers)


def test_ready_database_still_requires_approval_and_warns_about_authorization():
    detection = make_detection(
        drivers=[("mysql", "pymysql", True)],
        services=[
            DatabaseServiceStatus("mysql", "windows_service", "MySQL80", True),
        ],
    )
    request = WorkloadRequest(
        description="read sales transactions",
        workload_type=WorkloadType.TRANSACTIONAL_READ,
    )
    analysis = WorkloadAnalyzer().analyze(request)
    result = DatabaseSelector().recommend(request, analysis, detection)

    assert result.recommended_database == "mysql"
    assert result.eligible is True
    assert result.requires_user_approval is True
    assert any("authorize" in item.lower() for item in result.blockers)


def test_unknown_workload_never_selects_database():
    detection = make_detection(
        drivers=[("sqlite", "sqlite3", True)],
        services=[],
    )
    request = WorkloadRequest(description="do something")
    analysis = WorkloadAnalyzer().analyze(request)
    result = DatabaseSelector().recommend(request, analysis, detection)

    assert result.recommended_database is None
    assert result.eligible is False
    assert result.requires_user_approval is True


def test_sqlserver_or_oracle_are_candidates_when_ready():
    detection = make_detection(
        drivers=[("sqlserver", "pyodbc", True), ("oracle", "oracledb", True)],
        services=[DatabaseServiceStatus("sqlserver", "windows_service", "MSSQLSERVER", True)],
    )
    request = WorkloadRequest(
        description="read invoices", workload_type=WorkloadType.TRANSACTIONAL_READ,
    )
    result = DatabaseSelector().recommend(request, WorkloadAnalyzer().analyze(request), detection)

    assert result.recommended_database == "sqlserver"
    assert {item["database"] for item in result.detected_options} >= {"sqlserver", "oracle", "mongodb"}

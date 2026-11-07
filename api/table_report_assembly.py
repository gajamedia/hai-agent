from __future__ import annotations

"""Opt-in production wiring for the question -> table report endpoint.

Nothing happens on import. Reporting is enabled only when ``HAI_TABLE_REPORT_CONFIG``
names a JSON file (or a caller passes a config explicitly). The file selects the
approved SQLite database, the allowlisted tables/columns, and where results go.
It contains no passwords: SQLite needs none, and ``authorization_ref`` is only
the audit reference of the user's approval.

Example::

    {
      "database_path": "data/business.sqlite3",
      "profile_name": "sales",
      "authorization_ref": "approval:2026-11-07-sales",
      "workflow_db": "data/reports/workflow.sqlite3",
      "artifacts_dir": "data/reports/artifacts",
      "trace_path": "data/reports/trace.jsonl",
      "tables": [{
        "table": "sales", "description": "Sales orders", "key_column": "id",
        "columns": {"id": "integer", "region": "string", "amount": "number"},
        "filterable_columns": ["region"],
        "required_non_null_columns": ["amount"], "max_limit": 500
      }]
    }
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from agent.llm.business_database_adapter import ReadOnlyTableSpec
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.service import LLMService
from agent.llm.trusted_capability_invocation_port import CapabilityInvocationSpec
from agent.workflow import SQLiteWorkflowStore
from agent.workflow.analytics import WorkflowAnalyticalTaskExecutor
from agent.workflow.invocation import WorkflowCapabilityTaskExecutor
from agent.workflow.reports import LocalArtifactStore, WorkflowReportTaskExecutor
from agent.workflow.service import WorkflowApplicationService
from agent.workflow.table_report_job import READ_ACTION, TableReportJobRunner
from agent.workflow.table_report_planner import TableCatalogEntry, TableReportPlanner
from agent.workflow.table_report_service import TableReportService
from api.sqlite_database_gateway_factory import create_sqlite_readonly_gateway_factory

CONFIG_ENV = "HAI_TABLE_REPORT_CONFIG"


class TableReportConfigError(ValueError):
    """The table report configuration is missing or invalid."""


@dataclass(frozen=True)
class TableReportConfig:
    database_path: Path
    profile_name: str
    authorization_ref: str
    workflow_db: Path
    artifacts_dir: Path
    trace_path: Path
    entries: tuple[TableCatalogEntry, ...]
    specs: Mapping[str, ReadOnlyTableSpec]


def _text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TableReportConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def parse_config(data: Any, *, base_dir: Path | None = None) -> TableReportConfig:
    """Validate a config mapping; relative paths resolve against ``base_dir``."""
    if not isinstance(data, Mapping):
        raise TableReportConfigError("config must be a JSON object")
    base = base_dir or Path.cwd()

    def path(key: str) -> Path:
        p = Path(_text(data, key)).expanduser()
        return p if p.is_absolute() else base / p

    database_path = path("database_path")
    if not database_path.is_file():
        raise TableReportConfigError(f"database_path does not exist: {database_path}")
    tables = data.get("tables")
    if not isinstance(tables, list) or not tables:
        raise TableReportConfigError("'tables' must be a non-empty list")
    entries, specs = [], {}
    for item in tables:
        if not isinstance(item, Mapping):
            raise TableReportConfigError("each table must be an object")
        try:
            columns = dict(item["columns"])
            max_limit = int(item.get("max_limit", 500))
            spec = ReadOnlyTableSpec(
                table=_text(item, "table"), columns=tuple(columns),
                filterable_columns=tuple(item.get("filterable_columns", ())),
                default_limit=min(100, max_limit), max_limit=max_limit)
            entry = TableCatalogEntry.from_table_spec(
                spec, columns, description=str(item.get("description", "")),
                key_column=item.get("key_column"),
                required_non_null_columns=tuple(item.get("required_non_null_columns", ())))
        except TableReportConfigError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise TableReportConfigError(f"invalid table definition: {exc}") from exc
        if spec.table in specs:
            raise TableReportConfigError(f"duplicate table: {spec.table}")
        specs[spec.table] = spec
        entries.append(entry)
    return TableReportConfig(
        database_path=database_path, profile_name=_text(data, "profile_name"),
        authorization_ref=_text(data, "authorization_ref"),
        workflow_db=path("workflow_db"), artifacts_dir=path("artifacts_dir"),
        trace_path=path("trace_path"), entries=tuple(entries), specs=specs)


def load_config(path: str | Path) -> TableReportConfig:
    file = Path(path).expanduser()
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TableReportConfigError(f"cannot read {file}: {exc}") from exc
    return parse_config(data, base_dir=file.parent)


def build_create_app_kwargs(config: TableReportConfig, *,
                            llm_provider_factory: Callable[[], Any]) -> dict[str, Callable]:
    """Factories to pass to ``create_app`` (gateway, invocation specs, service)."""
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.workflow_db.parent.mkdir(parents=True, exist_ok=True)
    config.trace_path.parent.mkdir(parents=True, exist_ok=True)

    gateway_factory = create_sqlite_readonly_gateway_factory(
        database_path=config.database_path, profile_name=config.profile_name,
        authorization_ref=config.authorization_ref, table_specs=config.specs,
        trace_path=config.trace_path)

    def specs_factory(_app: Any) -> dict[str, CapabilityInvocationSpec]:
        return {READ_ACTION: CapabilityInvocationSpec(
            action_name=READ_ACTION, target="business_database",
            capability_id="business_database.select")}

    def service_factory(app: Any) -> TableReportService:
        port = getattr(app.state, "trusted_capability_invocation_port", None)
        if port is None:
            raise RuntimeError("Table reporting requires the trusted invocation port.")
        # Dedicated, minimal agent stand-in: the report job can only ever run the
        # one read action, under its own low-risk security policy.
        reader = SimpleNamespace(
            actions={}, unified_executor=object(),
            execution_security_policy=ExecutionSecurityPolicy(action_risks={READ_ACTION: 0.1}))
        store = SQLiteWorkflowStore(config.workflow_db)
        runner = TableReportJobRunner(
            service=WorkflowApplicationService(store=store), store=store,
            capability_executor=WorkflowCapabilityTaskExecutor(
                store=store, agent=reader, trusted_capability_invocation_port=port),
            analytical_executor=WorkflowAnalyticalTaskExecutor(store=store),
            report_executor=WorkflowReportTaskExecutor(
                store=store, artifact_store=LocalArtifactStore(config.artifacts_dir)))
        planner = TableReportPlanner(LLMService(llm_provider_factory()), config.entries)
        return TableReportService(planner=planner, runner=runner)

    return {"capability_execution_gateway_factory": gateway_factory,
            "trusted_capability_invocation_specs_factory": specs_factory,
            "table_report_service_factory": service_factory}


__all__ = ["CONFIG_ENV", "TableReportConfig", "TableReportConfigError",
           "parse_config", "load_config", "build_create_app_kwargs"]

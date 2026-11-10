"""The Database Agent's database-creation operation (``database.create``) on the shared platform.

Creating a database (or adding tables to one this application made) keeps every protection it had:
the server profile's creator account can only CREATE under a fixed name prefix, nothing is ever
dropped or overwritten, the exact phrase and a fresh login are needed. The platform now carries what
is not specific to databases: the durable task and its audit, replay protection, and the history.

Added by the platform: after the statements ran, the server is asked again whether the database and
every requested table exist. If it cannot be confirmed the task ends as "unverified", not "completed".

What stays in the Database Agent: the designer (LLM proposes a closed, structured design), the
statement builder and the JSON-lines audit that ``DatabaseCreationService`` has always written.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from agent.platform import ApprovalPolicy, Effect, Operation, PlanUnsupported, Verdict, Verification
from agent.workflow.database_creation import (
    ColumnDef, CreationRefused, DatabaseDesign, TableDef,
)
from api.database_runtime_registry import DatabaseRuntimeRegistry

FEATURE = "database"
CREATE_OPERATION = "database.create"
PERMISSION = "database:create:manage"


# --------------------------------------------------------------------------- plan <-> JSON

def design_to_json(design: DatabaseDesign) -> dict[str, Any]:
    return {"database": design.database, "title": design.title,
            "tables": [{"name": t.name,
                        "columns": [{"name": c.name, "type": c.type, "nullable": c.nullable}
                                    for c in t.columns]} for t in design.tables]}


def design_from_json(data: Mapping[str, Any]) -> DatabaseDesign:
    return DatabaseDesign(
        database=data["database"], title=data.get("title", ""),
        tables=tuple(TableDef(name=t["name"], columns=tuple(
            ColumnDef(name=c["name"], type=c["type"], nullable=bool(c["nullable"]))
            for c in t["columns"])) for t in data["tables"]))


# --------------------------------------------------------------------------- the feature

def describe_create(plan: Optional[Mapping[str, Any]], result: Optional[Mapping[str, Any]]) -> str:
    """One line for the task list: from the result once it ran, from the plan before."""
    if result and result.get("database"):
        verb = "Tambah tabel ke" if result.get("mode") == "add_tables" else "Buat database"
        return f"{verb} {result['database']}"
    summary = (plan or {}).get("summary") or {}
    if summary.get("database"):
        verb = "Tambah tabel ke" if (plan or {}).get("mode") == "add_tables" else "Buat database"
        return f"{verb} {summary['database']}"
    return "Pembuatan database"


def build_create_operation(get_registry: Callable[[], DatabaseRuntimeRegistry]) -> Operation:
    """``get_registry`` returns the live database registry (raising if it is not configured)."""

    def plan(request: Mapping[str, Any], ctx) -> dict[str, Any]:
        service = get_registry().database_creator(request.get("profile"))
        result = service.designer.plan(request["question"])
        if result.design is None:
            raise PlanUnsupported(result.reason or "The request cannot be done.", {"kind": "unsupported"})
        try:
            mode, statements = service.preview(result.design)
        except CreationRefused as exc:
            raise PlanUnsupported(str(exc), {"kind": "refused"}) from exc
        return {
            "question": result.question, "profile": service.profile,
            "design": design_to_json(result.design), "summary": service.summarize(result.design),
            "mode": mode, "statements": statements,
            "confirm_phrase": service.confirm_phrase(result.design, mode),
            "expected_tables": [t.name for t in result.design.tables],
            "provider": result.provider,
        }

    def preview(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        return {"confirm_phrase": plan_doc["confirm_phrase"], "summary": plan_doc["summary"],
                "mode": plan_doc["mode"], "statements": plan_doc["statements"]}

    def execute(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        service = get_registry().database_creator(plan_doc["profile"])
        return service.execute(design_from_json(plan_doc["design"]), actor=ctx.actor_id,
                               plan_id=ctx.task_id, question=plan_doc["question"])

    def verify(plan_doc: Mapping[str, Any], result: Mapping[str, Any], ctx) -> Verification:
        service = get_registry().database_creator(plan_doc["profile"])
        seen = service.verify(design_from_json(plan_doc["design"]))
        if not seen["confirmed"]:
            return Verification(Verdict.UNVERIFIED, seen)
        return Verification(Verdict.VERIFIED, seen)

    return Operation(
        feature=FEATURE, name="create", effect=Effect.WRITE, permission=PERMISSION, risk=0.6,
        approval=ApprovalPolicy(confirm_phrase=True, fresh_login=True),
        description="Create a database, or add tables to one this application created",
        plan=plan, preview=preview, execute=execute, verify=verify, max_active=50,
        label="Buat database", describe=describe_create)


__all__ = ["CREATE_OPERATION", "build_create_operation", "describe_create",
           "design_from_json", "design_to_json"]

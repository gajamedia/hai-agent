from __future__ import annotations

"""Joined listings: relations come from the database's foreign keys or are declared by the user, and the
adapter only ever joins along a relation that was approved."""

import json
import sqlite3

import pytest

from agent.llm.business_database_adapter import BusinessDatabaseAdapter, ReadOnlyTableSpec
from agent.llm.capability_execution_contracts import CapabilityRequestValidationError
from agent.llm.contracts import LLMResponse
from agent.llm.service import LLMService
from agent.workflow.table_report_planner import TableCatalogEntry, TableRelation, TableReportPlanner
from api.database_runtime_registry import DatabaseRuntimeRegistry, RegistryError
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import _context

ALL = ["id", "kode", "nama", "id_jenis"]
BAHAN = ["kode", "nama", "id_jenis", "id"]
JENIS = ["id", "nama"]


class _Scripted:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, request):
        self.prompts.append(request)
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


JOIN_PLAN = json.dumps({
    "supported": True, "kind": "list", "table": "tb_bahan",
    "joins": [{"table": "tb_jenisbahan", "type": "left"}],
    "columns": ["tb_bahan.kode", "tb_bahan.nama", "tb_jenisbahan.nama"],
    "order_by": [{"column": "tb_bahan.kode", "direction": "asc"}], "limit": 100})


def _make_db(folder):
    folder.mkdir()
    with sqlite3.connect(folder / "sales.sqlite3") as db:
        db.execute("CREATE TABLE tb_jenisbahan (id INTEGER PRIMARY KEY, nama TEXT)")
        db.execute("CREATE TABLE tb_bahan (id INTEGER PRIMARY KEY, kode TEXT, nama TEXT, "
                   "id_jenis INTEGER REFERENCES tb_jenisbahan(id), id_gudang INTEGER, stok INTEGER)")
        db.execute("CREATE TABLE tb_gudang (id INTEGER PRIMARY KEY, nama TEXT)")  # no foreign key points here
        db.executemany("INSERT INTO tb_jenisbahan VALUES (?,?)", [(1, "Kain"), (2, "Benang")])
        db.executemany("INSERT INTO tb_bahan VALUES (?,?,?,?,?,?)",
                       [(1, "B01", "Katun", 1, 1, 10), (2, "B02", "Wol", 2, 1, 5),
                        (3, "B03", "Tanpa jenis", None, 2, 7), (4, "B04", "Katun halus", 1, 2, 3)])
        db.execute("INSERT INTO tb_gudang VALUES (1, 'Pusat'), (2, 'Cabang')")


SELECTION = {
    "tb_bahan": {"columns": ["id", "kode", "nama", "id_jenis", "id_gudang", "stok"],
                 "filterable_columns": ["kode", "id_jenis"], "key_column": "id"},
    "tb_jenisbahan": {"columns": ["id", "nama"], "filterable_columns": ["nama"], "key_column": "id"},
    "tb_gudang": {"columns": ["id", "nama"], "filterable_columns": ["nama"], "key_column": "id"},
}
FK = {"table": "tb_bahan", "column": "id_jenis", "ref_table": "tb_jenisbahan", "ref_column": "id"}
MANUAL = {"table": "tb_bahan", "column": "id_gudang", "ref_table": "tb_gudang", "ref_column": "id"}


def _registry(tmp_path, *answers):
    data = tmp_path / "data"
    if not data.exists():
        _make_db(data)
    provider = _Scripted(*answers)
    reg = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: provider)
    return reg, provider


def test_foreign_keys_are_discovered(tmp_path):
    reg, _ = _registry(tmp_path)
    try:
        info = reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        assert info["schema"]["foreign_keys"] == [FK]
    finally:
        reg.shutdown()


def test_a_listing_joins_along_an_approved_relation_and_left_join_keeps_unmatched_rows(tmp_path):
    reg, provider = _registry(tmp_path, JOIN_PLAN)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        described = reg.activate("sales", SELECTION, actor_id="alice", relations=[FK, MANUAL])
        assert {r["source"] for r in described["relations"]} == {"foreign_key", "manual"}
        outcome = reg.service("sales").ask("daftar bahan dan jenisnya", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        assert outcome.plan["joins"] == [{"table": "tb_jenisbahan", "type": "left",
                                          "on": ["tb_jenisbahan.id", "tb_bahan.id_jenis"]}]
        # "nama" exists in both tables, so those two result columns are named after their table
        assert list(outcome.rows) == [
            {"kode": "B01", "tb_bahan_nama": "Katun", "tb_jenisbahan_nama": "Kain"},
            {"kode": "B02", "tb_bahan_nama": "Wol", "tb_jenisbahan_nama": "Benang"},
            {"kode": "B03", "tb_bahan_nama": "Tanpa jenis", "tb_jenisbahan_nama": None},
            {"kode": "B04", "tb_bahan_nama": "Katun halus", "tb_jenisbahan_nama": "Kain"},
        ]
        assert "RELATIONS" in provider.prompts[0].messages[0].content
    finally:
        reg.shutdown()


def test_relations_survive_a_restart_and_refresh(tmp_path):
    reg, _ = _registry(tmp_path)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        reg.activate("sales", SELECTION, actor_id="alice", relations=[FK, MANUAL])
        assert len(reg.refresh("sales", actor_id="alice")["relations"]) == 2
    finally:
        reg.shutdown()
    again, _ = _registry(tmp_path)
    try:
        listed = {c["name"]: c for c in again.list()}["sales"]
        assert listed["status"] == "active"
        assert {r["source"] for r in listed["relations"]} == {"foreign_key", "manual"}
        assert again.schema("sales")["approved_relations"] == listed["relations"]
    finally:
        again.shutdown()


@pytest.mark.parametrize("relation, needle", [
    ({**FK, "column": "no_such"}, "not an approved column"),
    ({**FK, "ref_table": "tb_bahan"}, "two different tables"),
    ({**FK, "ref_table": "tb_missing"}, "not an approved column"),
    ({**FK, "extra": 1}, "must be"),
])
def test_bad_relations_are_refused(tmp_path, relation, needle):
    reg, _ = _registry(tmp_path)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        with pytest.raises(RegistryError, match=needle):
            reg.activate("sales", SELECTION, actor_id="alice", relations=[relation])
    finally:
        reg.shutdown()


def test_the_source_of_a_relation_is_decided_by_the_server_not_the_request(tmp_path):
    reg, _ = _registry(tmp_path)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        described = reg.activate("sales", SELECTION, actor_id="alice",
                                 relations=[{**MANUAL, "source": "foreign_key"}])
        assert described["relations"][0]["source"] == "manual"
    finally:
        reg.shutdown()


# ---------------------------------------------------------------------- adapter
def _adapter(relations=((("tb_bahan"), "id_jenis", "tb_jenisbahan", "id"),), dialect="limit"):
    specs = {
        "tb_bahan": ReadOnlyTableSpec("tb_bahan", tuple(BAHAN), ("kode",), tuple(BAHAN)),
        "tb_jenisbahan": ReadOnlyTableSpec("tb_jenisbahan", tuple(JENIS), ("nama",), tuple(JENIS)),
        "tb_gudang": ReadOnlyTableSpec("tb_gudang", ("id", "nama")),
    }
    return BusinessDatabaseAdapter(connection_factory=lambda: None, table_specs=specs,
                                   identifier_quote='"', dialect=dialect, relations=relations)


def _join(on=("tb_bahan.id_jenis", "tb_jenisbahan.id"), **extra):
    return {"table": "tb_bahan", "columns": ["tb_bahan.kode", "tb_jenisbahan.nama"],
            "joins": [{"table": "tb_jenisbahan", "type": "inner", "on": list(on)}], "limit": 10, **extra}


def test_join_sql_is_built_from_the_allowlist_with_bound_values():
    sql, params = _adapter()._build_select(_join(
        filters={"tb_jenisbahan.nama": "Kain"},
        conditions=[{"column": "tb_bahan.kode", "operator": "starts_with", "value": "B"}],
        order_by=["-tb_bahan.kode"]))
    assert sql == ('SELECT "tb_bahan"."kode" AS "kode", "tb_jenisbahan"."nama" AS "nama" FROM "tb_bahan" '
                   'INNER JOIN "tb_jenisbahan" ON "tb_jenisbahan"."id" = "tb_bahan"."id_jenis" '
                   'WHERE "tb_jenisbahan"."nama" = ? AND "tb_bahan"."kode" LIKE ? ESCAPE \'!\' '
                   'ORDER BY "tb_bahan"."kode" DESC LIMIT ?')
    assert params == ("Kain", "B%", 10)


def test_top_dialect_binds_the_limit_first():
    sql, params = _adapter(dialect="top")._build_select(_join())
    assert sql.startswith('SELECT TOP (?) ') and params == (10,)


@pytest.mark.parametrize("arguments, needle", [
    (_join(on=("tb_bahan.id", "tb_jenisbahan.id")), "not an approved relation"),
    ({**_join(), "joins": [{"table": "tb_gudang", "on": ["tb_bahan.id", "tb_gudang.id"]}]},
     "not an approved relation"),
    ({**_join(), "joins": [{"table": "tb_missing", "on": ["tb_bahan.id", "tb_missing.id"]}]}, "allowlisted"),
    ({**_join(), "joins": [{"table": "tb_jenisbahan", "type": "cross", "on": ["a.b", "c.d"]}]}, "join type"),
    ({**_join(), "columns": ["kode"]}, "table.column"),
    ({**_join(), "columns": ["tb_gudang.nama"]}, "not allowlisted"),
    ({**_join(), "filters": {"tb_jenisbahan.id": 1}}, "filter column"),
    ({**_join(), "order_by": ["tb_gudang.id"]}, "not allowlisted"),
    ({**_join(), "columns": ["tb_bahan.kode; DROP TABLE x"]}, "table.column|Invalid"),
    ({**_join(), "joins": [{"table": "tb_jenisbahan", "on": ["tb_bahan.id_jenis", "tb_jenisbahan.id"]}] * 2},
     "only once"),
])
def test_joins_outside_the_approved_relations_are_refused(arguments, needle):
    with pytest.raises(CapabilityRequestValidationError, match=needle):
        _adapter()._build_select(arguments)


def test_an_adapter_without_relations_cannot_join():
    with pytest.raises(CapabilityRequestValidationError, match="not an approved relation"):
        _adapter(relations=())._build_select(_join())


def test_a_relation_on_unapproved_columns_cannot_even_be_configured():
    with pytest.raises(ValueError):
        _adapter(relations=(("tb_bahan", "nope", "tb_jenisbahan", "id"),))


# ---------------------------------------------------------------------- planner
CATALOG = [
    TableCatalogEntry("tb_bahan", {"id": "integer", "kode": "string", "nama": "string", "id_jenis": "integer",
                                   "stok": "integer"},
                      key_column="id", filterable_columns=("kode",)),
    TableCatalogEntry("tb_jenisbahan", {"id": "integer", "nama": "string"}, key_column="id",
                      filterable_columns=("nama",)),
]
REL = TableRelation("tb_bahan", "id_jenis", "tb_jenisbahan", "id")


def _planner(answer, relations=(REL,)):
    return TableReportPlanner(LLMService(_Scripted(answer, answer)), CATALOG, relations=relations)


def test_planner_finds_the_relation_itself_and_names_result_columns():
    result = _planner(JOIN_PLAN).plan("daftar bahan dan jenis")
    assert result.ok, result.reason
    req = result.request
    assert req.joins == (("tb_jenisbahan", "left", "tb_jenisbahan.id", "tb_bahan.id_jenis"),)
    assert req.select_columns == ("tb_bahan.kode", "tb_bahan.nama", "tb_jenisbahan.nama")
    assert req.columns["tb_jenisbahan.nama"] == "string"


@pytest.mark.parametrize("change, needle", [
    ({"kind": "chart"}, "kind must be"),
    ({"columns": "all"}, "distinct table.column"),
    ({"columns": ["kode"]}, "table.column"),
    ({"columns": ["tb_bahan.kode", "tb_bahan.bogus"]}, "Unknown column"),
    ({"joins": [{"table": "tb_jenisbahan", "on": ["tb_bahan.id", "tb_jenisbahan.id"]}]}, "not an approved relation"),
    ({"joins": [{"table": "tb_bahan"}]}, "already in the query"),
    ({"filters": {"tb_jenisbahan.id": 1}}, "Cannot filter"),
    ({"conditions": [{"column": "tb_bahan.nama", "operator": "eq", "value": "x"}]}, "Cannot filter"),
])
def test_bad_join_plans_are_rejected_with_a_reason(change, needle):
    answer = json.dumps({**json.loads(JOIN_PLAN), **change})
    result = _planner(answer).plan("q")
    assert not result.ok and needle in result.errors[0]


def test_without_relations_joins_stay_unsupported_and_the_prompt_says_so():
    planner = _planner(JOIN_PLAN, relations=())
    assert not planner.plan("q").ok
    prompt = planner._system_prompt()
    assert "RELATIONS" not in prompt and "joins, several tables" in prompt


def test_a_relation_outside_the_catalog_is_refused():
    with pytest.raises(ValueError, match="not in the catalog"):
        _planner(JOIN_PLAN, relations=(TableRelation("tb_bahan", "zzz", "tb_jenisbahan", "id"),))


# ------------------------------------------------------------- aggregates (stage 2)
AGG_PLAN = {
    "supported": True, "table": "tb_bahan", "joins": [{"table": "tb_jenisbahan", "type": "left"}],
    "group_by": ["tb_jenisbahan.nama"],
    "metrics": [{"output": "total_stok", "operation": "sum", "column": "tb_bahan.stok"},
                {"output": "jumlah", "operation": "count"}]}


def _by_group(rows):
    return {r["nama"]: (r["total_stok"], r["jumlah"]) for r in rows}


def test_totals_over_joined_tables_end_to_end(tmp_path):
    reg, _ = _registry(tmp_path, json.dumps(AGG_PLAN), json.dumps(
        {**AGG_PLAN, "joins": [{"table": "tb_jenisbahan", "type": "inner"}],
         "filters": {"tb_jenisbahan.nama": "Kain"}}))
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        reg.activate("sales", SELECTION, actor_id="alice", relations=[FK])
        left = reg.service("sales").ask("total stok per jenis", execution_context=_context())
        assert left.status == "succeeded", (left.error_code, left.reason)
        assert left.kind == "aggregate"
        # the left join keeps the item without a type as its own (empty) group
        assert _by_group(left.rows) == {"Kain": (13, 2), "Benang": (5, 1), None: (7, 1)}
        assert left.plan["joins"][0]["type"] == "left"

        inner = reg.service("sales").ask("total stok jenis Kain", execution_context=_context())
        assert inner.status == "succeeded", (inner.error_code, inner.reason)
        assert _by_group(inner.rows) == {"Kain": (13, 2)}
        assert inner.plan["filters"] == {"tb_jenisbahan.nama": "Kain"}
    finally:
        reg.shutdown()


def test_the_aggregate_request_carries_qualified_columns_and_catalog_types():
    req = _planner(json.dumps(AGG_PLAN)).plan("q").request
    assert req.kind == "aggregate" and req.group_by == ("tb_jenisbahan.nama",)
    assert req.metrics == (("total_stok", "sum", "tb_bahan.stok"), ("jumlah", "count", "tb_bahan.id"))
    assert req.columns == {"tb_jenisbahan.nama": "string", "tb_bahan.stok": "integer", "tb_bahan.id": "integer"}
    assert req.numeric_columns == ("tb_bahan.stok", "tb_bahan.id")
    assert req.categorical_columns == ("tb_jenisbahan.nama",)


@pytest.mark.parametrize("change, needle", [
    ({"group_by": ["nama"]}, "table.column"),
    ({"group_by": ["tb_jenisbahan.bogus"]}, "Unknown group_by column"),
    ({"metrics": [{"output": "x", "operation": "sum", "column": "tb_bahan.kode"}]}, "needs a numeric column"),
    ({"metrics": [{"output": "x", "operation": "sum", "column": "stok"}]}, "table.column"),
    ({"metrics": [{"output": "nama", "operation": "count"}]}, "clash"),
    ({"metrics": [{"output": "n", "operation": "median", "column": "tb_bahan.stok"}]}, "not allowed"),
    ({"joins": [{"table": "tb_jenisbahan", "on": ["tb_bahan.id", "tb_jenisbahan.id"]}]}, "not an approved relation"),
    ({"filters": {"tb_jenisbahan.id": 1}}, "Cannot filter"),
])
def test_bad_join_aggregates_are_rejected_with_a_reason(change, needle):
    result = _planner(json.dumps({**AGG_PLAN, **change})).plan("q")
    assert not result.ok and needle in result.errors[0]


def test_the_prompt_describes_joined_totals_only_when_relations_exist():
    with_relations = _planner(json.dumps(AGG_PLAN))._system_prompt()
    assert "over joined tables" in with_relations
    assert "over joined tables" not in _planner(json.dumps(AGG_PLAN), relations=())._system_prompt()


# ------------------------------------------------------ suggestions and the "needs a join" hint
from api.relation_suggestions import suggest_relations  # noqa: E402
from agent.database.sqlite_schema import ForeignKey, SQLiteColumn, SQLiteSchema, SQLiteTable  # noqa: E402


def _table(name, *columns):
    return SQLiteTable(name, tuple(SQLiteColumn(c, t, True, pk) for c, t, pk in columns))


def _schema(*tables, fks=()):
    return SQLiteSchema("x", tuple(tables), tuple(fks))


_INT = lambda declared: "integer" if "int" in declared.lower() else "string"  # noqa: E731


def test_a_column_name_suggests_the_table_it_resembles():
    schema = _schema(
        _table("tb_bahan", ("id", "int", True), ("id_jenis", "int", False), ("id_gudang", "int", False)),
        _table("tb_jenisbahan", ("id", "int", True), ("nama", "text", False)),
        _table("tb_gudang", ("id", "int", True)))
    pairs = [(r["table"], r["column"], r["ref_table"], r["ref_column"]) for r in suggest_relations(schema, _INT)]
    assert pairs == [("tb_bahan", "id_gudang", "tb_gudang", "id"),
                     ("tb_bahan", "id_jenis", "tb_jenisbahan", "id")]


def test_declared_foreign_keys_are_not_suggested_again():
    schema = _schema(
        _table("tb_bahan", ("id", "int", True), ("id_jenis", "int", False)),
        _table("tb_jenisbahan", ("id", "int", True)),
        fks=[ForeignKey("tb_bahan", "id_jenis", "tb_jenisbahan", "id")])
    assert suggest_relations(schema, _INT) == []


def test_ambiguous_or_mismatched_guesses_are_dropped():
    ambiguous = _schema(
        _table("tb_bahan", ("id", "int", True), ("id_jenis", "int", False)),
        _table("tb_jenisbahan", ("id", "int", True)), _table("tb_jenispegawai", ("id", "int", True)))
    assert suggest_relations(ambiguous, _INT) == []          # two equally good tables: do not guess
    wrong_type = _schema(
        _table("tb_bahan", ("id", "int", True), ("id_jenis", "text", False)),
        _table("tb_jenis", ("id", "int", True)))
    assert suggest_relations(wrong_type, _INT) == []
    unrelated = _schema(_table("tb_bahan", ("id", "int", True), ("ukuran", "int", False)),
                        _table("tb_jenis", ("id", "int", True)))
    assert suggest_relations(unrelated, _INT) == []


def test_exact_names_win_over_partial_ones_and_a_shared_key_name_matches():
    schema = _schema(
        _table("tb_bahan", ("id", "int", True), ("id_jenis", "int", False), ("kode_unit", "int", False)),
        _table("tb_jenis", ("id", "int", True)), _table("tb_jenisbahan", ("id", "int", True)),
        _table("tb_unit", ("kode_unit", "int", True)))
    got = {(r["column"], r["ref_table"], r["ref_column"]) for r in suggest_relations(schema, _INT)}
    assert ("id_jenis", "tb_jenis", "id") in got
    assert ("kode_unit", "tb_unit", "kode_unit") in got


def test_the_schema_offers_suggestions_for_what_the_database_does_not_declare(tmp_path):
    reg, _ = _registry(tmp_path)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        suggested = reg.schema("sales")["suggested_relations"]
        assert [(r["table"], r["column"], r["ref_table"]) for r in suggested] == [
            ("tb_bahan", "id_gudang", "tb_gudang")]          # id_jenis is a real foreign key: not repeated
        # accepting a suggestion makes it an ordinary manual relation
        described = reg.activate("sales", SELECTION, actor_id="alice", relations=[
            {k: v for k, v in suggested[0].items() if k != "reason"}])
        assert described["relations"][0]["source"] == "manual"
    finally:
        reg.shutdown()


def test_unsupported_because_a_join_is_missing_is_flagged():
    answer = json.dumps({"supported": False, "reason": "perlu join", "needs_join": True})
    result = _planner(answer, relations=()).plan("q")
    assert not result.ok and result.needs_join and result.reason == "perlu join"
    other = _planner(json.dumps({"supported": False, "reason": "bulan lalu"}), relations=()).plan("q")
    assert not other.ok and not other.needs_join
    assert '"needs_join": true' in _planner(answer)._system_prompt()


# ------------------------------------------- relations found by the AI, approved by running the plan
NEEDS_JOIN = json.dumps({"supported": False, "reason": "perlu join", "needs_join": True})
PROPOSAL = json.dumps({"relations": [
    {"table": "tb_bahan", "column": "id_jenis", "ref_table": "tb_jenisbahan", "ref_column": "id"}]})
QUESTION = "daftar bahan dan jenisnya"


def _ai_relation():
    return TableRelation("tb_bahan", "id_jenis", "tb_jenisbahan", "id", "ai")


def test_the_ai_proposes_a_relation_when_none_is_approved_and_the_plan_carries_it(tmp_path):
    reg, provider = _registry(tmp_path, NEEDS_JOIN, PROPOSAL, JOIN_PLAN)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        reg.activate("sales", SELECTION, actor_id="alice")          # no relation approved
        plan = reg.service("sales").plan(QUESTION)
        assert plan.ok and plan.proposed_relations == (_ai_relation(),)
        assert plan.request.joins[0][0] == "tb_jenisbahan"
        summary = reg.service("sales").summarize(plan.request, plan.proposed_relations)
        assert summary["proposed_relations"] == [
            {"table": "tb_bahan", "column": "id_jenis", "ref_table": "tb_jenisbahan", "ref_column": "id"}]
        # nothing is approved by merely planning
        assert reg.list()[0]["relations"] == []
        # the AI was shown the name-based guesses
        assert "tb_gudang" in provider.prompts[1].messages[0].content
    finally:
        reg.shutdown()


GUDANG_PLAN = json.dumps({
    "supported": True, "kind": "list", "table": "tb_bahan", "joins": [{"table": "tb_gudang", "type": "left"}],
    "columns": ["tb_bahan.kode", "tb_gudang.nama"], "order_by": [{"column": "tb_bahan.kode"}]})
GUDANG_PROPOSAL = json.dumps({"relations": [
    {"table": "tb_bahan", "column": "id_gudang", "ref_table": "tb_gudang", "ref_column": "id"}]})


def test_running_the_plan_approves_the_relation_and_it_is_remembered(tmp_path):
    # the database declares no foreign key for id_gudang, so the relation is the AI's until a person runs it
    reg, _ = _registry(tmp_path, NEEDS_JOIN, GUDANG_PROPOSAL, GUDANG_PLAN)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        reg.activate("sales", SELECTION, actor_id="alice")
        service = reg.service("sales")
        plan = service.plan("bahan dan gudangnya")
        assert [r.source for r in plan.proposed_relations] == ["ai"]
        outcome = service.run_plan("bahan dan gudangnya", plan, execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        assert [(r["kode"], r["nama"]) for r in outcome.rows] == [
            ("B01", "Pusat"), ("B02", "Pusat"), ("B03", "Cabang"), ("B04", "Cabang")]
        assert [(r["source"], r["ref_table"]) for r in reg.list()[0]["relations"]] == [("ai", "tb_gudang")]
        assert reg.service("sales") is service                      # same service: plans in flight stay valid
    finally:
        reg.shutdown()
    again, provider = _registry(tmp_path, GUDANG_PLAN)               # after a restart, and no proposal needed
    try:
        assert [r["source"] for r in again.list()[0]["relations"]] == ["ai"]
        assert again.schema("sales")["approved_relations"][0]["source"] == "ai"
        plan = again.service("sales").plan("bahan dan gudangnya")
        assert plan.ok and plan.proposed_relations == () and len(provider.prompts) == 1
    finally:
        again.shutdown()


def test_the_one_shot_endpoint_never_uses_a_relation_nobody_looked_at(tmp_path):
    reg, _ = _registry(tmp_path, NEEDS_JOIN, PROPOSAL, JOIN_PLAN)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        reg.activate("sales", SELECTION, actor_id="alice")
        outcome = reg.service("sales").ask(QUESTION, execution_context=_context())
        assert outcome.status == "unsupported" and "approve the relation" in outcome.reason
        assert reg.list()[0]["relations"] == []
    finally:
        reg.shutdown()


def test_a_service_that_cannot_approve_refuses_to_run_a_proposed_relation():
    from agent.workflow.table_report_service import TableReportService
    planner = _planner(JOIN_PLAN, relations=())
    result = _scripted_plan(NEEDS_JOIN, PROPOSAL, JOIN_PLAN)
    assert result.ok and result.proposed_relations
    outcome = TableReportService(planner=planner, runner=None).run_plan(
        QUESTION, result, execution_context=_context())
    assert outcome.status == "failed" and outcome.error_code == "relation_not_approved"


def _scripted_plan(*answers, hints=()):
    provider = _Scripted(*answers)
    planner = TableReportPlanner(LLMService(provider), CATALOG, relation_hints=hints)
    result = planner.plan(QUESTION)
    result_provider[0] = provider
    return result


result_provider = [None]


@pytest.mark.parametrize("proposal", [
    {"table": "tb_bahan", "column": "kode", "ref_table": "tb_jenisbahan", "ref_column": "id"},      # types differ
    {"table": "tb_bahan", "column": "stok", "ref_table": "tb_jenisbahan", "ref_column": "nama"},    # types differ
    {"table": "tb_bahan", "column": "nama", "ref_table": "tb_jenisbahan", "ref_column": "nama"},    # no key side
    {"table": "tb_bahan", "column": "id_jenis", "ref_table": "tb_missing", "ref_column": "id"},     # unknown
    {"table": "tb_bahan", "column": "bogus", "ref_table": "tb_jenisbahan", "ref_column": "id"},     # unknown
    {"table": "tb_bahan", "column": "id", "ref_table": "tb_bahan", "ref_column": "id"},            # same table
    {"table": "tb_bahan", "column": "id_jenis", "ref_table": "tb_jenisbahan", "ref_column": "id", "x": 1},
])
def test_proposals_that_cannot_be_verified_are_dropped(proposal):
    result = _scripted_plan(NEEDS_JOIN, json.dumps({"relations": [proposal]}))
    assert not result.ok and result.needs_join and result.proposed_relations == ()


def test_a_proposal_written_the_wrong_way_round_is_turned_around():
    reverse = json.dumps({"relations": [
        {"table": "tb_jenisbahan", "column": "id", "ref_table": "tb_bahan", "ref_column": "id_jenis"}]})
    result = _scripted_plan(NEEDS_JOIN, reverse, JOIN_PLAN)
    assert result.ok and result.proposed_relations == (_ai_relation(),)


@pytest.mark.parametrize("second", ["not json at all", json.dumps({"relations": "x"}),
                                    json.dumps({"relations": []})])
def test_when_the_ai_finds_nothing_the_answer_stays_unsupported(second):
    result = _scripted_plan(NEEDS_JOIN, second)
    assert not result.ok and result.needs_join and result.reason == "perlu join"


def test_a_failing_provider_during_the_proposal_does_not_break_the_answer():
    class Flaky(_Scripted):
        def complete(self, request):
            if not self.answers:
                raise RuntimeError("provider down")
            return super().complete(request)
    planner = TableReportPlanner(LLMService(Flaky(NEEDS_JOIN)), CATALOG)
    result = planner.plan(QUESTION)
    assert not result.ok and result.needs_join


def test_a_plan_that_does_not_use_the_proposed_relation_carries_none():
    single = json.dumps({"supported": True, "kind": "list", "table": "tb_bahan", "columns": ["id"]})
    result = _scripted_plan(NEEDS_JOIN, PROPOSAL, single)
    assert result.ok and result.proposed_relations == ()


def test_only_the_planner_and_the_service_can_mark_a_relation_as_found_by_the_ai(tmp_path):
    reg, _ = _registry(tmp_path)
    try:
        reg.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        described = reg.activate("sales", SELECTION, actor_id="alice", relations=[{**MANUAL, "source": "ai"}])
        assert described["relations"][0]["source"] == "manual"      # a request cannot claim "ai"
    finally:
        reg.shutdown()

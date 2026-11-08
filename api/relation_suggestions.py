from __future__ import annotations

"""Suggest relations a database does not declare, from column and table names.

Many databases (MyISAM, or schemas without constraints) hold ``tb_bahan.id_jenis`` that points at
``tb_jenisbahan.id`` without saying so. This reads only the discovered schema (names and types, never
data) and proposes pairs; a suggestion is never used until a person ticks it in the approval form, and the
server then treats it like any other manual relation.
"""

import re
from typing import Any, Callable, Mapping

_TABLE_PREFIX = re.compile(r"^(tb|tbl|t|table|m|ms|mst)_", re.IGNORECASE)
_REF_COLUMN = (re.compile(r"^id_(?P<x>.+)$", re.IGNORECASE), re.compile(r"^(?P<x>.+)_id$", re.IGNORECASE),
               re.compile(r"^fk_(?P<x>.+)$", re.IGNORECASE), re.compile(r"^(?P<x>.+)_fk$", re.IGNORECASE))
MAX_SUGGESTIONS = 40


def _stem(table: str) -> str:
    return _TABLE_PREFIX.sub("", table).lower()


def _target_column(table: Any) -> Any:
    """The column another table would point at: the single primary key, else a column called id."""
    keys = [c for c in table.columns if c.primary_key]
    if len(keys) == 1:
        return keys[0]
    return next((c for c in table.columns if c.name.lower() == "id"), None)


def _score(stem: str, name: str) -> int:
    """How well a table's name (without prefix) fits the word in the column name; 0 means no fit."""
    if stem == name:
        return 4
    if stem in {name + "s", name + "es"} or name in {stem + "s", stem + "es"}:
        return 3
    if stem.startswith(name) or stem.endswith(name):
        return 2
    return 0


def suggest_relations(schema: Any, map_type: Callable[[str], str]) -> list[dict[str, str]]:
    """Pairs the schema does not declare as foreign keys but whose names suggest a relation.

    A column ``id_jenis`` / ``jenis_id`` suggests the table whose name fits ``jenis`` best
    (``tb_jenis``, ``tb_jenisbahan``) and its key column. A column that has exactly the name of another
    table's single primary key (``id_jenis`` -> ``tb_jenis.id_jenis``) is suggested too. When several tables
    fit equally well nothing is suggested: guessing between them would be wrong half the time."""
    declared = {frozenset({(k.table, k.column), (k.ref_table, k.ref_column)})
                for k in getattr(schema, "foreign_keys", ())}
    tables = list(schema.tables)
    found: dict[frozenset, dict[str, str]] = {}
    for table in tables:
        for column in table.columns:
            if column.primary_key and len([c for c in table.columns if c.primary_key]) == 1:
                continue  # a table's own key is the target of relations, not a pointer
            candidates: list[tuple[int, Any, Any, str]] = []
            for other in tables:
                if other.name == table.name:
                    continue
                target = _target_column(other)
                if target is None or map_type(target.declared_type) != map_type(column.declared_type):
                    continue
                if target.primary_key and target.name == column.name and column.name.lower() != "id":
                    candidates.append((5, other, target, f"kolom {column.name} sama dengan kunci {other.name}"))
                    continue
                for pattern in _REF_COLUMN:
                    match = pattern.match(column.name)
                    if match:
                        score = _score(_stem(other.name), match.group("x").lower())
                        if score:
                            candidates.append((score, other, target,
                                               f"nama kolom {column.name} mirip nama tabel {other.name}"))
                        break
            if not candidates:
                continue
            best = max(c[0] for c in candidates)
            top = [c for c in candidates if c[0] == best]
            if len(top) != 1:
                continue
            _, other, target, reason = top[0]
            key = frozenset({(table.name, column.name), (other.name, target.name)})
            if key in declared or key in found:
                continue
            found[key] = {"table": table.name, "column": column.name, "ref_table": other.name,
                          "ref_column": target.name, "reason": reason}
    return sorted(found.values(), key=lambda r: (r["table"], r["column"]))[:MAX_SUGGESTIONS]


__all__ = ["suggest_relations"]

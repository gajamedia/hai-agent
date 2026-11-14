from __future__ import annotations

"""What a person typed next to an uploaded file: instructions about the file, turned into fill rules.

Typical text: "isi kolom jabatan dengan Staff, departemen acak dari HR/IT/Finance, gaji acak 4 sampai 9 juta".
The model is shown **names only** (the table's columns and types, which of them the file already fills, and the
typed text); it never sees a value from the file. It answers with *rules*, not values; our own code draws the
values, checks each one against its column's type, and the person sees every resulting row before saving.
It may also return complete extra rows the text spells out ("catat juga gula 3").
"""

import logging
import random
import re
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from agent.llm.contracts import LLMMessage
from agent.workflow.table_report_planner import _extract_json
from agent.workflow.uploaded_table import UploadRefused, coerce

logger = logging.getLogger(__name__)

MAX_CHOICES = 50
MAX_EXTRA_ROWS = 100
# Random values must never stand in for identity or secrets; a person can still type an exact value.
SENSITIVE = re.compile(r"nik|ktp|npwp|rekening|rek_|passport|paspor|password|sandi|pin$|token|secret|kartu|card|cvv", re.I)


def propose_fill(llm: Any, instruction: str, columns: Mapping[str, str], writable: Sequence[str],
                 filled_by_file: Sequence[str], table: str,
                 mode: str = "insert") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(rules, extra_rows). Raises :class:`UploadRefused` with a reason the person can act on."""
    catalog = {c: columns[c] for c in writable}
    free = [c for c in writable if c not in filled_by_file]
    if mode == "update":
        purpose = (f"to UPDATE existing rows of the table {table!r} (each spreadsheet row names one existing row; "
                   "fill sets the new value of a column on EVERY row listed in the file; rows must always be [])")
    else:
        purpose = f"to be recorded as new rows into the table {table!r}"
    system = (
        f"A person uploaded a spreadsheet {purpose} and typed an instruction. You see "
        "column names only, never the spreadsheet's data. Reply with ONLY one JSON object:\n"
        '{"supported": true|false, "reason": "<why not, in the person\'s language>", '
        '"fill": {"<column>": RULE}, "rows": [{"<column>": value}]}\n'
        "RULE is exactly one of: {\"value\": x} (same for every row) | {\"choice\": [x, y, ...]} (picked at random per "
        "row) | {\"random_int\": [min, max]} | {\"random_decimal\": [min, max, decimals]} | "
        "{\"random_date\": [\"YYYY-MM-DD\", \"YYYY-MM-DD\"]} | {\"sequence\": [\"<prefix>\", <start>, <digits>]} "
        "(unique running numbers such as EMP-2026-001, EMP-2026-002: use it for an employee, product or code "
        "number column the table needs, e.g. nip, kode, no_pegawai; such a column must not stay empty).\n"
        "fill may only name columns the file does not already fill (FREE COLUMNS). When the person asks for all or "
        "the other columns to be filled randomly, give a plausible rule for EVERY free column you can judge from its "
        "name and type (e.g. jenis_kelamin -> choice of two values, agama -> choice, departemen/jabatan -> choice of "
        "common values, gaji -> random_int in a realistic range, tanggal -> random_date, status -> choice). Skip "
        "columns holding identity numbers, account or tax numbers, passwords or other secrets: they are left empty "
        "by our code, so do NOT refuse because of them. Skip columns you cannot judge (they stay empty too). "
        "rows are complete extra rows the text spells out literally; leave it [] otherwise. Never invent a column "
        "name. If the text only says to use the file's data as it is (for example 'update the employees with the "
        "data in the file', 'add the data in the file'), set supported to true and leave fill {} and rows []: the "
        "file is used as it is. Set supported to false only if the text asks for something that is not recording "
        "this file into this table (delete data, read the file, another table).\n"
        f"TABLE COLUMNS (name: type): {catalog}\nFILE ALREADY FILLS: {list(filled_by_file)}\nFREE COLUMNS: {free}")
    try:
        response = llm.complete([LLMMessage("system", system), LLMMessage("user", f"Instruction:\n{instruction}")],
                                temperature=0)
        data = _extract_json(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fill rules unavailable: %s", type(exc).__name__)
        raise UploadRefused("Teks tambahan tidak bisa diproses sekarang (model tidak menjawab). "
                            "Kosongkan teks untuk mencatat file apa adanya, atau coba lagi nanti.") from exc
    if not isinstance(data, Mapping):
        raise UploadRefused("Teks tambahan tidak dapat dipahami. Tulis kolom mana yang diisi dan nilainya.")
    if data.get("supported") is False:
        raise UploadRefused(str(data.get("reason") or "Teks tambahan tidak dapat dikerjakan.").strip())
    rules = data.get("fill") or {}
    extra = data.get("rows") or []
    if not isinstance(rules, Mapping) or not isinstance(extra, list) or len(extra) > MAX_EXTRA_ROWS \
            or any(not isinstance(r, Mapping) or not r for r in extra):
        raise UploadRefused("Teks tambahan tidak dapat dipahami. Tulis kolom mana yang diisi dan nilainya.")
    for column, rule in rules.items():
        if column not in free:
            raise UploadRefused(f"Kolom {column!r} tidak bisa diisi dari teks (sudah diisi file atau tidak boleh diisi).")
        if not isinstance(rule, Mapping) or len(rule) != 1:
            raise UploadRefused(f"Aturan untuk kolom {column!r} tidak jelas.")
    # Random values never stand in for identity or secrets: such a column stays empty unless an exact value is typed.
    rules = {c: r for c, r in rules.items() if not (SENSITIVE.search(c) and next(iter(r)) != "value")}
    if mode == "update" and extra:
        raise UploadRefused("Di Ubah data, teks hanya bisa mengisi kolom untuk baris yang ada di file; "
                            "menambah baris baru dilakukan di halaman Catat data.")
    # No rules and no extra rows is fine: the text only restated that the file is to be used as it is.
    return dict(rules), [dict(r) for r in extra]


def _draw(rule: Mapping[str, Any], rng: random.Random, index: int = 0) -> Any:
    (kind, arg), = rule.items()
    try:
        if kind == "value":
            return arg
        if kind == "sequence":
            prefix, start, width = str(arg[0])[:30], int(arg[1]), max(0, min(int(arg[2]), 12))
            return f"{prefix}{start + index:0{width}d}"
        if kind == "choice":
            if not isinstance(arg, list) or not 1 <= len(arg) <= MAX_CHOICES:
                raise ValueError
            return rng.choice(arg)
        if kind == "random_int":
            low, high = int(arg[0]), int(arg[1])
            return rng.randint(min(low, high), max(low, high))
        if kind == "random_decimal":
            low, high, places = float(arg[0]), float(arg[1]), max(0, min(int(arg[2]), 6))
            return round(rng.uniform(min(low, high), max(low, high)), places)
        if kind == "random_date":
            start, end = date.fromisoformat(arg[0]), date.fromisoformat(arg[1])
            if end < start:
                start, end = end, start
            return (start + timedelta(days=rng.randint(0, (end - start).days))).isoformat()
    except (ValueError, TypeError, IndexError) as exc:
        raise UploadRefused(f"Aturan isian {rule!r} tidak valid.") from exc
    raise UploadRefused(f"Jenis aturan isian {kind!r} tidak dikenal.")


def apply_fill(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any], columns: Mapping[str, str],
               *, seed: int | None = None) -> list[dict[str, Any]]:
    """The file's rows plus one drawn value per rule and row, each converted to its column's type."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=2):
        filled = dict(row)
        for column, rule in rules.items():
            try:
                filled[column] = coerce(columns[column], _draw(rule, rng, number - 2))
            except ValueError as exc:
                raise UploadRefused(f"Isian untuk kolom {column!r} tidak cocok dengan tipenya: {exc}") from exc
        out.append(filled)
    return out


__all__ = ["apply_fill", "propose_fill"]

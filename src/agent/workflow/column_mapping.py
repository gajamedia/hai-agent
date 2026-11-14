from __future__ import annotations

"""Proposing which table column each column of an uploaded file belongs to.

The model is shown **names only**: the file's column headers and the table's column names and types. It never
sees a single value from the file. Its answer is only a proposal: it is checked by our own code, shown to the
person who uploaded the file, and used only after they accept (or correct) it. If no model answers, the person
simply maps the columns by hand.
"""

import logging
from typing import Any, Mapping, Sequence

from agent.llm.contracts import LLMMessage
from agent.workflow.table_report_planner import _extract_json

logger = logging.getLogger(__name__)


def propose_mapping(llm: Any, headers: Sequence[str], columns: Mapping[str, str],
                    writable: Sequence[str]) -> dict[str, str | None]:
    """{header: table column or None}. Anything the model gets wrong is dropped, never guessed at; an empty
    result means 'no proposal'."""
    catalog = {c: columns[c] for c in writable}
    system = (
        "You match the column headers of an uploaded spreadsheet to the columns of a database table. You see "
        "names only, never data. Reply with ONLY one JSON object: "
        '{"mapping": {"<file header>": "<table column or null>"}}. Use null when no table column clearly '
        "matches. Use each table column at most once. Never invent a column name.\n"
        f"TABLE COLUMNS (name: type): {catalog}")
    user = f"File headers: {list(headers)}"
    try:
        response = llm.complete([LLMMessage("system", system), LLMMessage("user", user)], temperature=0)
        data = _extract_json(response.content)
    except Exception as exc:  # noqa: BLE001 - no provider, or an unreadable answer: the person maps by hand
        logger.warning("Column mapping proposal unavailable: %s", type(exc).__name__)
        return {}
    raw = data.get("mapping") if isinstance(data, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    proposal: dict[str, str | None] = {}
    taken: set[str] = set()
    for header in headers:
        column = raw.get(header)
        if isinstance(column, str) and column in catalog and column not in taken:
            proposal[header] = column
            taken.add(column)
        else:
            proposal[header] = None
    return proposal

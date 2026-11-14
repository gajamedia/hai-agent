from __future__ import annotations

"""Deciding which page's pipeline a chat message (and perhaps a file) belongs to.

The model is shown the typed message, the file's *name* and its column *headers*, and the names of the
databases and tables the person may record into. It never sees a value from the file. Its answer is only a
suggestion: it is checked here against a fixed list, and nothing is planned, let alone run, by this step. When
the answer is missing, unreadable or unsure, the intent is ``unclear`` and the chat asks the person.
"""

import logging
import re
from typing import Any, Mapping, Sequence

from agent.llm.contracts import LLMMessage
from agent.workflow.table_report_planner import _extract_json

logger = logging.getLogger(__name__)

CREATE_DATABASE, RECORD_DATA, CHANGE_DATA, VIEW_DATA, UNCLEAR = (
    "create_database", "record_data", "change_data", "view_data", "unclear")
INTENTS = (CREATE_DATABASE, RECORD_DATA, CHANGE_DATA, VIEW_DATA)
REPORT_FORMATS = ("xlsx", "pdf", "csv", "html")
MIN_CONFIDENCE = 0.6
_NAME = re.compile(r"[a-z][a-z0-9_]{1,63}")

_SYSTEM = (
    "You route a request to one of four database tools. You see the person's message, and for an uploaded file "
    "only its name and column headers, never its data. Reply with ONLY one JSON object: "
    '{"intent": "create_database|record_data|change_data|view_data|unclear", "confidence": 0.0-1.0, '
    '"database": "<database name or null>", "table": "<table name or null>", "format": "<xlsx|pdf|csv|html or null>", "question": "<the request, '
    'restated briefly, or null>"}.\n'
    "- create_database: make a NEW database or table (also: a file whose columns should become a new table).\n"
    "- record_data: ADD new rows/records to an existing table (a file of new rows to add).\n"
    "- change_data: UPDATE or DELETE existing rows (correct, edit, remove; a file of corrections).\n"
    "- view_data: read, list, count, summarise or report on existing data.\n"
    "- unclear: you cannot tell, or the message is not about databases. Prefer unclear over guessing, "
    "especially between record_data and change_data.\n"
    "- format: only for view_data, and only if the person asked for the result as a file in that format "
    "(Excel = xlsx); otherwise null.\n"
    "Use database/table names only if the person wrote them or they appear in the list below; otherwise null.")


def _name(value: Any) -> str | None:
    return value if isinstance(value, str) and _NAME.fullmatch(value) else None


def classify_intent(llm: Any, message: str, *, filename: str | None = None,
                    headers: Sequence[str] = (), known: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """{intent, confidence, database, table, question}. Never raises; on any trouble the intent is 'unclear'."""
    unclear = {"intent": UNCLEAR, "confidence": 0.0, "database": None, "table": None, "format": None, "question": None}
    parts = [f"Message: {message.strip()[:2000]!r}"]
    if filename:
        parts.append(f"Uploaded file: {filename[:200]!r} with columns {[str(h)[:80] for h in headers][:60]}")
    else:
        parts.append("No file uploaded.")
    catalog = {d.get("name"): list(d.get("tables", []))[:30] for d in list(known)[:20]}
    system = _SYSTEM + f"\nDATABASES AND TABLES THE PERSON CAN RECORD INTO: {catalog}"
    try:
        response = llm.complete([LLMMessage("system", system), LLMMessage("user", "\n".join(parts))], temperature=0)
        data = _extract_json(response.content)
    except Exception as exc:  # noqa: BLE001 - no provider or an unreadable answer: ask the person instead
        logger.warning("Intent routing unavailable: %s", type(exc).__name__)
        return unclear
    intent = data.get("intent")
    confidence = data.get("confidence")
    if intent not in INTENTS or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return unclear
    if confidence < MIN_CONFIDENCE:
        return {**unclear, "confidence": float(confidence)}
    question = data.get("question")
    return {"intent": intent, "confidence": float(min(confidence, 1.0)),
            "database": _name(data.get("database")), "table": _name(data.get("table")),
            "format": data.get("format") if intent == VIEW_DATA and data.get("format") in REPORT_FORMATS else None,
            "question": question.strip()[:2000] if isinstance(question, str) and question.strip() else None}

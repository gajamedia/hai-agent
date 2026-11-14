from __future__ import annotations

"""What the upload routes share: reading the request body safely, and answering "these columns need matching".

Kept free of any other route module so that every route that takes a file can import it.
"""

import json
from typing import Any, Mapping, Sequence

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from agent.workflow.column_mapping import propose_mapping
from agent.workflow.uploaded_table import MAX_UPLOAD_BYTES, MappingNeeded, UploadedTable


def _too_large() -> HTTPException:
    return HTTPException(413, detail={"error": "file_too_large",
                                      "message": f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."})


async def read_upload(request: Request) -> bytes:
    """The raw request body, refused as soon as it passes the size limit (the rest is never read)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise _too_large()
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_UPLOAD_BYTES:
            raise _too_large()
    return bytes(body)


def parse_mapping(text: str | None) -> Mapping[str, Any] | None:
    """The ``mapping`` query parameter: a JSON object (file column -> table column or null), or None."""
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise HTTPException(422, detail={"error": "invalid_mapping",
                                         "message": "mapping must be a JSON object: file column -> table column."})
    return value


async def needs_mapping(exc: MappingNeeded, parsed: UploadedTable, llm: Any, columns: Mapping[str, str],
                        usable: Sequence[str], **extra: Any) -> dict[str, Any]:
    """The answer when some headers do not match: a proposal made from column NAMES only (a model never sees a
    value), for the person to correct. ``usable`` are the table columns the file may be mapped to."""
    free = [c for c in usable if c not in exc.auto.values()]
    proposal = (await run_in_threadpool(propose_mapping, llm, list(exc.unknown), columns, free)) if free else {}
    merged = {h: exc.auto.get(h) or proposal.get(h) for h in parsed.headers}
    filled = [v for v in merged.values() if v]
    if len(set(filled)) != len(filled):
        merged = {h: exc.auto.get(h) for h in parsed.headers}       # a clash: keep only the certain matches
    return {"status": "needs_mapping", "plan_id": None, "reason": str(exc), "headers": list(parsed.headers),
            "columns": [{"name": c, "type": columns[c]} for c in usable], "mapping": merged,
            "proposed_by_model": any(merged.get(h) != exc.auto.get(h) for h in merged), **extra}

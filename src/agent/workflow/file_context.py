from __future__ import annotations

"""An uploaded file written out as text, to be read by a model together with what the person typed.

Only for a page whose person has accepted that the file's content goes to the model provider (the database
design page: the file there is a description of the table wanted, not business data). It is capped, so a big
file sends its first rows and says how many were left out.
"""

import csv
import io

from agent.workflow.uploaded_table import UploadedTable

MAX_ROWS_SENT = 100
MAX_CHARS_SENT = 20000
MAX_CELL_CHARS = 200


def file_as_text(table: UploadedTable, *, max_rows: int = MAX_ROWS_SENT, max_chars: int = MAX_CHARS_SENT) -> tuple[str, int]:
    """(the file as CSV text, how many data rows it holds). Cells are shortened, the rows capped."""
    def line(cells) -> str:
        out = io.StringIO()
        csv.writer(out, lineterminator="").writerow(
            "" if c is None else str(c).replace("\n", " ")[:MAX_CELL_CHARS] for c in cells)
        return out.getvalue()

    lines = [line(table.headers)]
    sent = 0
    size = len(lines[0])
    for row in table.rows[:max_rows]:
        text = line(row)
        if size + len(text) > max_chars:
            break
        lines.append(text)
        size += len(text) + 1
        sent += 1
    return "\n".join(lines), sent


def describe_request(*, instruction: str, table: UploadedTable, database: str, table_name: str | None,
                     rows_text: str, rows_sent: int) -> str:
    """The one request a model reads: the instruction, the names that must be used, and the file."""
    total = len(table.rows)
    left_out = f" ({total - rows_sent} more rows were left out)" if rows_sent < total else ""
    parts = [f"Database name to use: {database}"]
    if table_name:
        parts.append(f"Table name to use for the table made from the file: {table_name}")
    parts.append(f"Instruction: {instruction}")
    parts.append(
        f"The person also uploaded the file {table.filename!r}"
        f"{f' (sheet {table.sheet})' if table.sheet else ''}. It is the description of what to design: use its "
        f"column names, and choose each column's type from its values. The first {rows_sent} of {total} rows "
        f"follow{left_out}:\n{rows_text}")
    return "\n\n".join(parts)


__all__ = ["describe_request", "file_as_text"]

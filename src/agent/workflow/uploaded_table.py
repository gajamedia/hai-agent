from __future__ import annotations

"""Reading a table from an uploaded file (CSV, TSV, TXT or Excel .xlsx) without an LLM.

The file is parsed and checked by our own code only: its rows never go to a language model. The result is
either a clean table (headers, rows) or a plain-language refusal that names the row and column at fault.
Nothing here touches a database; mapping the table onto a real table is :func:`rows_for_table`.
"""

import csv
import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMPORT_ROWS = 1000
MAX_COLUMNS = 200
MAX_ERRORS_SHOWN = 10
SUPPORTED = (".csv", ".tsv", ".txt", ".xlsx")
_TRUE = {"true", "ya", "y", "yes", "1", "benar"}
_FALSE = {"false", "tidak", "t", "n", "no", "0", "salah"}
_NUMBER = re.compile(r"^[+-]?\d+(\.\d+)?$")
_DMY = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")


class UploadRefused(ValueError):
    """The file cannot be used; the message says why, for the person who uploaded it."""


@dataclass(frozen=True)
class UploadedTable:
    filename: str
    sha256: str
    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]   # the cell values, as read (text for CSV, typed for Excel)
    sheet: str | None = None
    lines: tuple[int, ...] = ()         # the spreadsheet row number of each data row, for messages


def normalize_name(text: Any) -> str:
    """'Nama Produk ' -> 'nama_produk'; accents and punctuation are dropped, so headers match column names."""
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UploadRefused("The file's text encoding is not recognised; save it as UTF-8 and upload it again.")


def _read_csv(data: bytes) -> list[list[Any]]:
    text = _decode(data)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return [list(row) for row in csv.reader(io.StringIO(text, newline=""), dialect)]


def _read_xlsx(data: bytes) -> tuple[list[list[Any]], str]:
    try:
        from openpyxl import load_workbook
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # a corrupt or password-protected file
        raise UploadRefused("The Excel file could not be opened. Is it a valid .xlsx without a password?") from exc
    try:
        sheet = book.worksheets[0] if book.worksheets else None
        if sheet is None:
            raise UploadRefused("The Excel file has no sheet.")
        rows: list[list[Any]] = []
        for row in sheet.iter_rows(values_only=True):
            rows.append(list(row))
            if len(rows) > MAX_IMPORT_ROWS + 50:  # blank trailing rows are trimmed below; stop reading far past the cap
                break
        return rows, sheet.title
    finally:
        book.close()


def _filled(row: Sequence[Any]) -> int:
    return sum(1 for cell in row if cell not in (None, "") and str(cell).strip() != "")


def read_table(filename: str, data: bytes, header_row: int = 1) -> UploadedTable:
    """Parse an uploaded file into headers and rows, or raise :class:`UploadRefused`. ``header_row`` is the
    spreadsheet row (1 = the first) that holds the column names; rows above it (a title, say) are ignored."""
    if not isinstance(header_row, int) or not 1 <= header_row <= 100:
        raise UploadRefused("The header row must be a row number between 1 and 100.")
    name = (filename or "").strip()
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in SUPPORTED:
        if suffix in {".xls", ".xlsm", ".xlsb"}:
            raise UploadRefused("Old or macro Excel formats are not accepted; save the file as .xlsx or .csv.")
        raise UploadRefused(f"Only {', '.join(SUPPORTED)} files are accepted.")
    if not data:
        raise UploadRefused("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRefused(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    sheet = None
    if suffix == ".xlsx":
        raw, sheet = _read_xlsx(data)
    else:
        raw = _read_csv(data)
    numbered = list(enumerate(raw, start=1))[header_row - 1:]      # (spreadsheet row number, cells)
    if header_row == 1:
        # A lone title above a wider table ("Data pegawai") is not the column names: skip it.
        rows_after = [_filled(r) for _n, r in numbered[1:6] if _filled(r)]
        if len(numbered) > 2 and _filled(numbered[0][1]) == 1 and rows_after and max(rows_after) > 1:
            numbered = numbered[1:]
    # Drop empty rows (all cells blank) and empty trailing columns.
    numbered = [(n, row) for n, row in numbered if _filled(row)]
    raw = [row for _n, row in numbered]
    line_numbers = [n for n, _row in numbered]
    if len(raw) < 2:
        raise UploadRefused("The file needs a header row and at least one row of data.")
    headers = [str(cell).strip() if cell is not None else "" for cell in raw[0]]
    while headers and not headers[-1] and not any(
            len(row) >= len(headers) and row[len(headers) - 1] not in (None, "") and str(row[len(headers) - 1]).strip()
            for row in raw[1:]):
        headers.pop()  # a trailing column with no name and no data is just spreadsheet padding
    if not headers or any(not h for h in headers):
        raise UploadRefused("Every column in the first row needs a name.")
    if len(headers) > MAX_COLUMNS:
        raise UploadRefused(f"The file has more than {MAX_COLUMNS} columns.")
    normalized = [normalize_name(h) for h in headers]
    if len(set(normalized)) != len(normalized):
        raise UploadRefused("Two columns have the same name; rename one and upload again.")
    body = raw[1:]
    if len(body) > MAX_IMPORT_ROWS:
        raise UploadRefused(
            f"The file has {len(body)} rows; at most {MAX_IMPORT_ROWS} can be recorded at once. Split it and upload the parts.")
    rows = tuple(tuple((row + [None] * len(headers))[:len(headers)]) for row in body)
    return UploadedTable(name, hashlib.sha256(data).hexdigest(), tuple(headers), rows, sheet,
                         tuple(line_numbers[1:]))


# --------------------------------------------------------------------------- values


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce(kind: str, value: Any) -> Any:
    """One cell as the column's type needs it, or ValueError saying what was expected. Empty means NULL."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if kind == "integer":
        if isinstance(value, bool):
            raise ValueError("expected a whole number")
        if isinstance(value, (int, float, Decimal)):
            if value != int(value):
                raise ValueError("expected a whole number")
            return int(value)
        text = _text(value)
        if not _NUMBER.match(text) or "." in text and Decimal(text) != int(Decimal(text)):
            raise ValueError("expected a whole number")
        return int(Decimal(text))
    if kind == "number":
        if isinstance(value, bool):
            raise ValueError("expected a number")
        if isinstance(value, (int, float, Decimal)):
            return float(value) if not isinstance(value, int) else value
        text = _text(value)
        if not _NUMBER.match(text):
            # '1.234,50' or '1,5' would be guesses between decimal and thousands separators: do not guess.
            raise ValueError("expected a plain number written with a dot as the decimal mark, such as 25000 or 12.5")
        try:
            return float(Decimal(text)) if "." in text else int(text)
        except InvalidOperation:
            raise ValueError("expected a number") from None
    if kind in {"date", "datetime"}:
        if isinstance(value, datetime):
            return (value.date() if kind == "date" else value).isoformat()
        if isinstance(value, date):
            return value.isoformat() if kind == "date" else datetime.combine(value, time()).isoformat()
        text = _text(value)
        match = _DMY.match(text)
        if match:
            day, month, year = (int(g) for g in match.groups())
            try:
                parsed = date(year, month, day)
            except ValueError:
                raise ValueError("not a real date") from None
            return parsed.isoformat() if kind == "date" else datetime.combine(parsed, time()).isoformat()
        try:
            return (date if kind == "date" else datetime).fromisoformat(text).isoformat()
        except ValueError:
            raise ValueError("expected a date such as 2026-03-31 or 31/03/2026") from None
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        text = _text(value).lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError("expected ya/tidak (or true/false)")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return _text(value)


class MappingNeeded(UploadRefused):
    """Some file columns do not match a column of the table by name. ``auto`` holds the ones that do."""

    def __init__(self, unknown: Sequence[str], auto: Mapping[str, str], writable: Sequence[str]) -> None:
        super().__init__(f"These columns are not in the table: {', '.join(unknown)}. "
                         f"Columns that can be filled: {', '.join(writable)}.")
        self.unknown, self.auto = tuple(unknown), dict(auto)


def auto_mapping(table: UploadedTable, writable: Sequence[str]) -> dict[str, str | None]:
    """Each header matched to the column of the same name (after :func:`normalize_name`), else None."""
    by_name = {normalize_name(c): c for c in writable}
    return {h: by_name.get(normalize_name(h)) for h in table.headers}


def check_mapping(table: UploadedTable, writable: Sequence[str], mapping: Mapping[str, Any]) -> dict[str, str | None]:
    """A mapping chosen by a person (or proposed by a model) is only used if it is plainly valid."""
    if not isinstance(mapping, Mapping) or set(mapping) != set(table.headers):
        raise UploadRefused("The column mapping must name every column of the file.")
    chosen = [v for v in mapping.values() if v is not None]
    if any(not isinstance(v, str) or v not in writable for v in chosen):
        raise UploadRefused(f"A file column was mapped to a column that cannot be filled; allowed: {', '.join(writable)}.")
    if len(chosen) != len(set(chosen)):
        raise UploadRefused("Two file columns were mapped to the same table column.")
    if not chosen:
        raise UploadRefused("Map at least one file column to a column of the table.")
    return dict(mapping)


def rows_for_table(table: UploadedTable, columns: Mapping[str, str], writable: Sequence[str],
                   mapping: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Map the file onto a table and convert every value to its column's type. Without a ``mapping`` the headers
    must match column names (:func:`normalize_name`); otherwise :class:`MappingNeeded` says which do not. With
    one, a file column mapped to None is left out on purpose. Anything that does not fit refuses the whole
    file (nothing is skipped or guessed), naming the first problems."""
    if mapping is None:
        found = auto_mapping(table, writable)
        unknown = [h for h, c in found.items() if c is None]
        if unknown:
            raise MappingNeeded(unknown, {h: c for h, c in found.items() if c}, writable)
        mapping = found
    else:
        mapping = check_mapping(table, writable, mapping)
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    used = [(i, h, mapping[h]) for i, h in enumerate(table.headers) if mapping[h] is not None]
    for position, cells in enumerate(table.rows):
        # The number as the spreadsheet shows it (falls back to counting from the header as line 1).
        number = table.lines[position] if position < len(table.lines) else position + 2
        row: dict[str, Any] = {}
        for index, header, column in used:
            try:
                row[column] = coerce(columns[column], cells[index])
            except ValueError as exc:
                problems.append(f"Row {number}, column {header}: {exc} (found {str(cells[index])[:40]!r}).")
        rows.append(row)
    if problems:
        more = len(problems) - MAX_ERRORS_SHOWN
        shown = " ".join(problems[:MAX_ERRORS_SHOWN]) + (f" And {more} more." if more > 0 else "")
        raise UploadRefused(f"The file was not recorded. {shown}")
    return rows


def design_from_table(table: UploadedTable) -> list[tuple[str, str]]:
    """Column names and types for a new table that would hold this file: a type is chosen only when every
    non-empty value of the column fits it, otherwise the column is text."""
    columns: list[tuple[str, str]] = []
    for index, header in enumerate(table.headers):
        values = [row[index] for row in table.rows if row[index] not in (None, "")]
        texts = [str(v).strip() for v in values]
        # Codes such as phone numbers, ID numbers or '007' look like numbers but are not: keep them as text.
        code_like = any((len(t) > 1 and t.startswith("0") and t.isdigit()) or (t.isdigit() and len(t) > 9)
                        for t in texts)
        kind = "text"
        if values and not code_like:
            for candidate, probe in (("integer", "integer"), ("decimal", "number"), ("date", "date"),
                                     ("datetime", "datetime"), ("boolean", "boolean")):
                try:
                    for value in values:
                        coerce(probe, value)
                except ValueError:
                    continue
                kind = candidate
                break
        if kind == "text" and any(len(t) > 255 for t in texts):
            kind = "long_text"
        columns.append((normalize_name(header), kind))
    return columns


__all__ = ["MappingNeeded", "auto_mapping", "check_mapping", "MAX_IMPORT_ROWS", "MAX_UPLOAD_BYTES", "SUPPORTED", "UploadRefused", "UploadedTable", "coerce",
           "design_from_table", "normalize_name", "read_table", "rows_for_table"]

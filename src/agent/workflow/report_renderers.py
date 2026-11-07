from __future__ import annotations

"""Dependency-free renderers for tabular reports: Excel (.xlsx) and PDF.

Both take rows that were already validated by the caller (same columns in every
row) and never execute anything: no formulas in Excel, no scripts or links in PDF.
"""

import io
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

# ---------------------------------------------------------------------- shared
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def cell_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return "" if value is None else str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ------------------------------------------------------------------------ XLSX
_SHEET_NAME_BAD = re.compile(r"[\[\]:*?/\\']")


def sheet_name(title: str) -> str:
    """Excel forbids []:*?/\\ and apostrophes in sheet names and caps them at 31 characters."""
    name = _SHEET_NAME_BAD.sub(" ", title)
    name = re.sub(r"\s+", " ", name).strip()[:31].strip()
    return name or "Report"


def _column_letters(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
    '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
    '<fills count="2"><fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill></fills>'
    '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
    '</styleSheet>'
)


def _xlsx_cell(value: Any, column: int, row: int, *, header: bool = False) -> str:
    ref = f"{_column_letters(column)}{row}"
    style = ' s="1"' if header else ""
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{ref}"{style} t="b"><v>{int(value)}</v></c>'
    if _is_number(value):
        # Real numeric cells: Excel can sum, sort and format them. A number is
        # never a formula, so this stays injection-safe.
        return f'<c r="{ref}"{style}><v>{value!r}</v></c>'
    text = xml_escape(_CONTROL.sub(" ", cell_text(value)))
    # Inline strings are text, never formulas, even when they start with "=".
    return f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def render_xlsx(*, title: str, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    table = [list(fieldnames)] + [[row[name] for name in fieldnames] for row in rows]
    widths = [
        min(60, max(8, max(len(cell_text(line[i])) for line in table) + 2))
        for i in range(len(fieldnames))
    ]
    cols = "".join(
        f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths, 1)
    )
    sheet_rows = "".join(
        f'<row r="{r}">' + "".join(
            _xlsx_cell(value, c, r, header=(r == 1)) for c, value in enumerate(line, 1)
        ) + "</row>"
        for r, line in enumerate(table, 1)
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<cols>{cols}</cols><sheetData>{sheet_rows}</sheetData></worksheet>"
    )
    name = xml_escape(sheet_name(title), {'"': "&quot;"})
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{name}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _STYLES)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    return output.getvalue()


# ------------------------------------------------------------------------- PDF
_COURIER = 0.6  # Courier advances 0.6 em per character, so layout is exact
_MIN_COLUMN = 4


def _pdf_string(text: str) -> bytes:
    """A PDF literal string: Latin-1 (WinAnsi) only, control characters removed, escaped."""
    cleaned = _CONTROL.sub(" ", text.replace("\r", " ").replace("\n", " "))
    raw = cleaned.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _fit(text: str, width: int) -> str:
    text = _CONTROL.sub(" ", text.replace("\r", " ").replace("\n", " "))
    return text if len(text) <= width else text[: max(0, width - 1)] + "~"


def render_pdf(*, title: str, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]],
               generated_at: datetime | None = None) -> bytes:
    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    names = list(fieldnames)
    cells = [[cell_text(row[n]) for n in names] for row in rows]
    numeric = [all(row[n] is None or _is_number(row[n]) for row in rows) for n in names]

    landscape = len(names) > 4
    page_w, page_h = (842, 595) if landscape else (595, 842)
    margin = 40
    usable = page_w - 2 * margin
    gap = 2
    widths = [max([len(n)] + [len(c[i]) for c in cells]) for i, n in enumerate(names)]

    def total() -> int:
        return sum(widths) + gap * (len(widths) - 1)

    font = 9.0
    while font > 6 and total() * font * _COURIER > usable:
        font -= 0.5
    max_chars = int(usable / (font * _COURIER))
    while total() > max_chars and max(widths) > _MIN_COLUMN:
        widths[widths.index(max(widths))] -= 1  # trim the widest column first

    def line(values: Sequence[str]) -> str:
        parts = []
        for i, value in enumerate(values):
            text = _fit(value, widths[i])
            parts.append(text.rjust(widths[i]) if numeric[i] else text.ljust(widths[i]))
        return (" " * gap).join(parts).rstrip()

    line_h = font * 1.45
    head_block = 62
    foot_block = 34
    per_page = max(1, int((page_h - 2 * margin - head_block - foot_block - line_h) // line_h))
    chunks = [cells[i:i + per_page] for i in range(0, len(cells), per_page)] or [[]]
    title_chars = int(usable / (14 * _COURIER))

    streams: list[bytes] = []
    for number, chunk in enumerate(chunks, 1):
        ops: list[bytes] = []

        def text_op(font_name: str, size: float, x: float, y: float, text: str) -> None:
            ops.append(b"BT /%s %s Tf %s %s Td (%s) Tj ET" % (
                font_name.encode(), repr(size).encode(), repr(x).encode(), repr(y).encode(),
                _pdf_string(text)))

        top = page_h - margin
        text_op("F2", 14, margin, top - 14, _fit(title, title_chars))
        text_op("F1", 8, margin, top - 30, f"Dibuat: {stamp}   |   {len(rows)} baris")
        y = top - head_block
        text_op("F2", font, margin, y, line(names))
        ops.append(b"%s %s m %s %s l 0.5 w S" % (
            repr(margin).encode(), repr(y - 3).encode(),
            repr(margin + min(usable, total() * font * _COURIER)).encode(), repr(y - 3).encode()))
        for cell_row in chunk:
            y -= line_h
            text_op("F1", font, margin, y, line(cell_row))
        text_op("F1", 8, margin, margin - 12, f"Halaman {number}/{len(chunks)}")
        streams.append(b"\n".join(ops))

    objects: list[bytes] = []   # index 0 is object 1
    kids = [5 + 2 * i for i in range(len(chunks))]  # page objects are 5, 7, 9, ...
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % k for k in kids)
                   + b"] /Count %d >>" % len(chunks))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>")
    for i, stream in enumerate(streams):
        page_obj, content_obj = 5 + 2 * i, 6 + 2 * i
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>"
            % (page_w, page_h, content_obj))
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    info_id = len(objects) + 1
    objects.append(b"<< /Title (" + _pdf_string(title[:200]) + b") /Producer (hAI Agent) >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
              % (len(objects) + 1, info_id, xref))
    return out.getvalue()


__all__ = ["cell_text", "render_pdf", "render_xlsx", "sheet_name"]

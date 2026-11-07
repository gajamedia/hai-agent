from __future__ import annotations

import csv
import io
import re
import xml.dom.minidom
from datetime import datetime, timezone
from zipfile import ZipFile

import pytest

from agent.workflow.report_renderers import render_pdf, render_xlsx, sheet_name
from agent.workflow.reports import _csv_bytes

ROWS = [{"region": "west", "revenue": 175.5, "orders": 7, "active": True, "note": None},
        {"region": "east", "revenue": 50, "orders": 2, "active": False, "note": "ok"}]
NAMES = ["region", "revenue", "orders", "active", "note"]
WHEN = datetime(2026, 11, 7, 10, 30, tzinfo=timezone.utc)


# ------------------------------------------------------------------- XLSX
def _xlsx(title="Sales", rows=ROWS, names=NAMES):
    return render_xlsx(title=title, fieldnames=names, rows=rows)


def test_every_part_of_the_workbook_is_well_formed_xml():
    with ZipFile(io.BytesIO(_xlsx())) as z:
        names = z.namelist()
        assert {"[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/styles.xml",
                "xl/worksheets/sheet1.xml", "xl/_rels/workbook.xml.rels"} <= set(names)
        for name in names:
            xml.dom.minidom.parseString(z.read(name))  # raises on malformed XML


def test_numbers_are_numeric_cells_and_text_is_inline_string():
    with ZipFile(io.BytesIO(_xlsx())) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
    assert '<c r="B2"><v>175.5</v></c>' in sheet            # a real number, not text
    assert '<c r="C3"><v>2</v></c>' in sheet
    assert '<c r="D2" t="b"><v>1</v></c>' in sheet            # boolean
    assert 'r="E2"' not in sheet                              # None -> empty cell
    assert '<c r="A2" t="inlineStr">' in sheet and ">west<" in sheet


def test_formula_text_stays_text_and_is_escaped():
    rows = [{"a": "=HYPERLINK(\"http://evil\",\"x\")", "b": "<b>&"}]
    with ZipFile(io.BytesIO(render_xlsx(title="t", fieldnames=["a", "b"], rows=rows))) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
        xml.dom.minidom.parseString(sheet)
    assert "<f>" not in sheet and 't="inlineStr"' in sheet and "&lt;b&gt;&amp;" in sheet


@pytest.mark.parametrize("title,expected", [
    ("Sales by region [filter: region=west]", "Sales by region filter region=w"),
    ("a/b\\c:d*e?f", "a b c d e f"), ("'quoted'", "quoted"), ("   ", "Report"), ("[]", "Report"),
    ("x" * 80, "x" * 31)])
def test_sheet_names_follow_excel_rules(title, expected):
    name = sheet_name(title)
    assert name == expected and len(name) <= 31 and not re.search(r"[\[\]:*?/\\']", name)


def test_workbook_opens_in_a_real_reader_with_typed_cells_and_header():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(_xlsx("Sales [filter: x=y]")))
    ws = wb.active
    assert ws.title == "Sales filter x=y"
    assert [c.value for c in ws[1]] == NAMES and ws["A1"].font.b is True
    assert ws["B2"].value == 175.5 and isinstance(ws["C2"].value, int)
    assert ws["D2"].value is True and ws["E2"].value is None and ws["E3"].value == "ok"
    assert ws.freeze_panes == "A2"
    assert ws["B2"].data_type == "n"                          # numeric, so Excel can SUM it
    assert ws.column_dimensions["A"].width >= 8


def test_a_numeric_column_can_be_summed_by_a_reader():
    openpyxl = pytest.importorskip("openpyxl")
    ws = openpyxl.load_workbook(io.BytesIO(_xlsx())).active
    assert sum(ws.cell(row=r, column=2).value for r in (2, 3)) == pytest.approx(225.5)


# --------------------------------------------------------------------- CSV
def test_csv_neutralizes_formula_prefixes_but_keeps_numbers():
    aggregation = {"rows": [
        {"name": "=SUM(A1)", "n": -5}, {"name": "+1", "n": 1}, {"name": "-x", "n": 2},
        {"name": "@cmd", "n": 3}, {"name": "\tTAB", "n": 4}, {"name": "plain", "n": -1.5}]}
    parsed = list(csv.DictReader(_csv_bytes(aggregation).decode("utf-8-sig").splitlines()))
    assert [r["name"] for r in parsed] == ["'=SUM(A1)", "'+1", "'-x", "'@cmd", "'\tTAB", "plain"]
    assert [r["n"] for r in parsed] == ["-5", "1", "2", "3", "4", "-1.5"]  # real negatives untouched


# --------------------------------------------------------------------- PDF
def _pdf(title="Laporan penjualan", rows=ROWS, names=NAMES):
    return render_pdf(title=title, fieldnames=names, rows=rows, generated_at=WHEN)


def test_pdf_has_a_valid_cross_reference_table():
    data = _pdf()
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    start = int(re.search(rb"startxref\n(\d+)\n", data).group(1))
    assert data[start:start + 4] == b"xref"
    count = int(re.search(rb"xref\n0 (\d+)\n", data).group(1))
    entries = re.findall(rb"(\d{10}) 00000 n ", data)
    assert len(entries) == count - 1
    for number, offset in enumerate(entries, 1):
        assert data[int(offset):].startswith(b"%d 0 obj" % number)


def test_pdf_text_is_extractable_by_a_real_reader():
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(_pdf()))
    assert len(reader.pages) == 1 and reader.metadata.title == "Laporan penjualan"
    text = reader.pages[0].extract_text()
    assert "Laporan penjualan" in text and "west" in text and "175.5" in text
    assert "2026-11-07 10:30 UTC" in text and "2 baris" in text and "Halaman 1/1" in text


def test_pdf_paginates_and_repeats_the_header():
    pypdf = pytest.importorskip("pypdf")
    rows = [{"id": i, "name": f"item-{i}"} for i in range(200)]
    reader = pypdf.PdfReader(io.BytesIO(_pdf(rows=rows, names=["id", "name"])))
    assert len(reader.pages) > 2
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        assert "id" in text and f"Halaman {number}/{len(reader.pages)}" in text
    assert "item-199" in reader.pages[-1].extract_text()


def test_pdf_wide_tables_use_landscape_and_still_fit():
    pypdf = pytest.importorskip("pypdf")
    names = [f"column_{i}" for i in range(8)]
    rows = [{n: "x" * 30 for n in names}]
    page = pypdf.PdfReader(io.BytesIO(_pdf(names=names, rows=rows))).pages[0]
    assert float(page.mediabox.width) > float(page.mediabox.height)
    assert "~" in page.extract_text()  # long cells were trimmed instead of running off the page


def test_pdf_cannot_be_broken_by_special_characters_and_has_no_active_content():
    pypdf = pytest.importorskip("pypdf")
    nasty = "evil ) Tj ET /JS (app.alert(1)) \\ ( \x00\x07\n\r café 中文"
    data = _pdf(title=nasty, rows=[{"a": nasty, "b": 1}], names=["a", "b"])
    reader = pypdf.PdfReader(io.BytesIO(data))
    text = reader.pages[0].extract_text()
    assert "café" in text and "?" in text                  # Latin-1 kept, CJK replaced
    # The attack text is only ever shown as text (it survived intact, so it was escaped) ...
    assert "Tj ET" in text and "app.alert(1)" in text
    # ... and no object in the file carries an action or annotation.
    active = {"/JS", "/JavaScript", "/OpenAction", "/AA", "/URI", "/Launch", "/Annots", "/A",
              "/Names", "/EmbeddedFiles"}
    count = int(re.search(rb"xref\n0 (\d+)\n", data).group(1))
    for number in range(1, count):
        obj = reader.get_object(number)
        if hasattr(obj, "keys"):
            assert not (active & set(obj.keys())), (number, set(obj.keys()))


def test_pdf_right_aligns_numbers():
    pytest.importorskip("pypdf")
    data = _pdf(rows=[{"n": 5}, {"n": 12345}], names=["n"])
    assert b"(    5)" in data or b"(5)" in data
    assert re.search(rb"\(\s+5\)", data)


def test_empty_row_pages_do_not_crash_for_a_single_row():
    assert render_pdf(title="t", fieldnames=["a"], rows=[{"a": 1}], generated_at=WHEN).startswith(b"%PDF")


def test_opening_the_workbook_raises_no_reader_warnings():
    openpyxl = pytest.importorskip("openpyxl")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. "Workbook contains no default style"
        openpyxl.load_workbook(io.BytesIO(_xlsx()))

"""Reading a table from an uploaded file: our own parser, strict, with plain-language refusals."""
from __future__ import annotations

import io
from datetime import date, datetime

import pytest

from agent.workflow.uploaded_table import (
    MAX_IMPORT_ROWS, UploadRefused, coerce, design_from_table, normalize_name, read_table, rows_for_table,
)

COLUMNS = {"product": "text", "amount": "number", "qty": "integer", "sold_on": "date", "paid": "boolean"}
WRITABLE = tuple(COLUMNS)


def csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def xlsx_bytes(rows) -> bytes:
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    for row in rows:
        sheet.append(row)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_a_csv_is_read_with_its_delimiter_guessed():
    for delimiter in (",", ";", "\t"):
        text = delimiter.join(["Product", "Amount"]) + "\n" + delimiter.join(["kopi", "25"]) + "\n"
        table = read_table("sales.csv", csv_bytes(text))
        assert table.headers == ("Product", "Amount") and table.rows == (("kopi", "25"),)


def test_an_excel_file_keeps_its_types_and_names_its_sheet():
    table = read_table("sales.xlsx", xlsx_bytes([["product", "amount", "sold_on"],
                                                  ["kopi", 25.5, datetime(2026, 3, 31)]]))
    assert table.sheet and table.rows[0][1] == 25.5
    rows = rows_for_table(table, COLUMNS, WRITABLE)
    assert rows == [{"product": "kopi", "amount": 25.5, "sold_on": "2026-03-31"}]


def test_headers_match_column_names_however_they_are_written():
    assert normalize_name(" Nama Produk ") == "nama_produk"
    assert normalize_name("Tanggal-Jual") == "tanggal_jual"
    table = read_table("a.csv", csv_bytes("PRODUCT,Qty\nkopi,3\n"))
    assert rows_for_table(table, COLUMNS, WRITABLE) == [{"product": "kopi", "qty": 3}]


def test_blank_rows_are_ignored_and_empty_cells_become_null():
    table = read_table("a.csv", csv_bytes("product,amount\nkopi,\n,\n\nteh,15\n"))
    assert rows_for_table(table, COLUMNS, WRITABLE) == [{"product": "kopi", "amount": None},
                                                         {"product": "teh", "amount": 15}]


@pytest.mark.parametrize("name,data,message", [
    ("a.pdf", b"x", "Only"),
    ("a.xls", b"x", "Old or macro"),
    ("a.csv", b"", "empty"),
    ("a.csv", b"product\n", "header row and at least one row"),
    ("a.csv", b"product,Product\nx,y\n", "same name"),
    ("a.csv", b"product,\nx,y\n", "needs a name"),
    ("a.xlsx", b"not a zip", "could not be opened"),
])
def test_a_file_that_cannot_be_used_says_why(name, data, message):
    with pytest.raises(UploadRefused, match=message):
        read_table(name, data)


def test_too_many_rows_are_refused_not_cut():
    text = "product\n" + "x\n" * (MAX_IMPORT_ROWS + 1)
    with pytest.raises(UploadRefused, match="at most"):
        read_table("a.csv", csv_bytes(text))


def test_a_column_that_is_not_in_the_table_refuses_the_whole_file():
    table = read_table("a.csv", csv_bytes("product,secret\nkopi,1\n"))
    with pytest.raises(UploadRefused, match="not in the table: secret"):
        rows_for_table(table, COLUMNS, WRITABLE)


def test_a_bad_value_names_its_row_and_column_and_nothing_is_skipped():
    table = read_table("a.csv", csv_bytes("product,amount,qty\nkopi,25,1\nteh,abc,2\nroti,3,1.5\n"))
    with pytest.raises(UploadRefused) as caught:
        rows_for_table(table, COLUMNS, WRITABLE)
    message = str(caught.value)
    assert "Row 3, column amount" in message and "Row 4, column qty" in message and "not recorded" in message


def test_no_more_than_ten_problems_are_listed():
    table = read_table("a.csv", csv_bytes("qty\n" + "x\n" * 30))
    with pytest.raises(UploadRefused, match="And 20 more"):
        rows_for_table(table, COLUMNS, WRITABLE)


@pytest.mark.parametrize("kind,value,expected", [
    ("integer", "12", 12), ("integer", 12.0, 12), ("number", "12.5", 12.5), ("number", "1200", 1200),
    ("date", "31/03/2026", "2026-03-31"), ("date", "2026-03-31", "2026-03-31"),
    ("date", date(2026, 3, 31), "2026-03-31"), ("datetime", "2026-03-31 10:00", "2026-03-31T10:00:00"),
    ("boolean", "Ya", True), ("boolean", "tidak", False), ("text", "  kopi ", "kopi"), ("text", 7, "7"),
    ("integer", "", None),
])
def test_values_are_converted_to_the_column_type(kind, value, expected):
    assert coerce(kind, value) == expected


@pytest.mark.parametrize("kind,value", [
    ("integer", "1.5"), ("integer", "abc"), ("number", "1.234,50"), ("number", "1,5"),
    ("date", "31/02/2026"), ("date", "kemarin"), ("boolean", "mungkin"),
])
def test_ambiguous_or_wrong_values_are_refused_rather_than_guessed(kind, value):
    with pytest.raises(ValueError):
        coerce(kind, value)


def test_a_new_table_is_designed_from_the_file_and_codes_stay_text():
    table = read_table("a.csv", csv_bytes(
        "Nama,Umur,Gaji,Masuk,Aktif,Telepon,NIK\n"
        "Ani,30,5000000.50,01/02/2020,ya,08123456789,3201010101010001\n"
        "Budi,41,6000000,15/06/2019,tidak,0811111111,3201010101010002\n"))
    assert design_from_table(table) == [
        ("nama", "text"), ("umur", "integer"), ("gaji", "decimal"), ("masuk", "date"),
        ("aktif", "boolean"), ("telepon", "text"), ("nik", "text")]

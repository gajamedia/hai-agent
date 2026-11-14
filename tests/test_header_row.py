"""Which spreadsheet row holds the column names: a title above them must not become data."""
from __future__ import annotations

import io

from openpyxl import Workbook

from agent.workflow.uploaded_table import read_table


def _xlsx(rows):
    book = Workbook()
    for row in rows:
        book.active.append(row)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_the_header_row_can_be_chosen_when_a_title_sits_above_it():
    data = _xlsx([["Data pegawai tambahan"], ["nama lengkap"], ["Ani"], ["Budi"]])
    wrong = read_table("a.xlsx", data)
    assert wrong.headers == ("Data pegawai tambahan",) and len(wrong.rows) == 3
    right = read_table("a.xlsx", data, header_row=2)
    assert right.headers == ("nama lengkap",) and [r[0] for r in right.rows] == ["Ani", "Budi"]


def test_a_lone_title_above_a_wider_table_is_skipped_automatically():
    data = _xlsx([["Data pegawai"], ["nama", "jabatan"], ["Ani", "Staf"], ["Budi", "Manajer"]])
    table = read_table("a.xlsx", data)
    assert table.headers == ("nama", "jabatan") and len(table.rows) == 2


def test_a_plain_table_is_untouched():
    table = read_table("a.csv", b"nama,jabatan\nAni,Staf\n")
    assert table.headers == ("nama", "jabatan") and len(table.rows) == 1


def test_an_error_names_the_row_as_the_spreadsheet_shows_it():
    import pytest
    from agent.workflow.uploaded_table import UploadRefused, rows_for_table
    data = _xlsx([["Judul"], ["product", "amount"], ["kopi", 5], [None, None], ["teh", "banyak"]])
    table = read_table("a.xlsx", data, header_row=2)
    assert table.lines == (3, 5)
    with pytest.raises(UploadRefused, match=r"Row 5, column amount"):
        rows_for_table(table, {"product": "string", "amount": "integer"}, ["product", "amount"])

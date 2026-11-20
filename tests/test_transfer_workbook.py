"""Workbook reader tests: the stdlib .xlsx parser returns anchor
positions and signal-quality flags, never guesses."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.transfer.workbook import (
    read_anchor_positions,
    read_signal_quality_flags,
    read_workbook_columns,
)

_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    """Build a tiny single-sheet .xlsx with the std library (no
    openpyxl) — inline strings only, no shared strings."""
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            col = ""
            n = c_idx
            while n:
                n, rem = divmod(n - 1, 26)
                col = chr(65 + rem) + col
            cells.append(
                f'<c r="{col}{r_idx}" t="inlineStr">'
                f"<is><t>{escape(str(value))}</t></is></c>")
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    sheet_xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<worksheet xmlns="{_SHEET_NS}"><sheetData>'
        + "".join(sheet_rows) + "</sheetData></worksheet>")
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<workbook xmlns="{_SHEET_NS}"><sheets>'
        '<sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>')
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-'
        'package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.spreadsheetml.'
        'worksheet+xml"/></Types>')
    root_rel = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rel)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_reader_returns_anchor_positions_and_flags(tmp_path: Path) -> None:
    rows = [
        ["SUBJECT", "Age", "M1_Contact_Img", "Signal quality"],
        ["SUB01", "61", "3", "1"],
        ["SUB02", "63", "6", "0"],
        ["SUB03", "78", "No Postop CT", "1"],  # non-numeric anchor
        ["", "", "", ""],                       # empty row skipped
    ]
    xlsx = tmp_path / "wb.xlsx"
    _minimal_xlsx(xlsx, rows)
    anchors = read_anchor_positions(xlsx)
    assert anchors == {"SUB01": 3.0, "SUB02": 6.0}
    flags = read_signal_quality_flags(xlsx)
    assert flags == {"SUB01": 1.0, "SUB02": 0.0, "SUB03": 1.0}


def test_reader_rejects_missing_columns(tmp_path: Path) -> None:
    xlsx = tmp_path / "wb.xlsx"
    _minimal_xlsx(xlsx, [["SUBJECT", "Age"], ["SUB01", "61"]])
    with pytest.raises(DataValidationError):
        read_anchor_positions(xlsx)


def test_reader_reports_requested_columns(tmp_path: Path) -> None:
    xlsx = tmp_path / "wb.xlsx"
    _minimal_xlsx(xlsx, [["SUBJECT", "M1_Contact_Img"], ["SUB01", "5"]])
    table = read_workbook_columns(xlsx, ("SUBJECT", "M1_Contact_Img"))
    assert table["SUB01"]["M1_Contact_Img"] == "5"


def test_reader_rejects_non_xlsx(tmp_path: Path) -> None:
    junk = tmp_path / "wb.xlsx"
    junk.write_bytes(b"not a zip")
    with pytest.raises(DataValidationError):
        read_anchor_positions(junk)

"""Minimal stdlib .xlsx reader for the localization workbook (v2).

The v2 driver needs exactly one per-recording field from the
localization workbook: the motor-anchor contact position
(``M1_Contact_Img``, 1–8). Adding a spreadsheet-parsing dependency for
one column is not warranted (dependency changes require review per the
owner brief), so this module reads the .xlsx with the standard library
only (an .xlsx is a zip of XML). It parses the first worksheet's rows,
locates the ``SUBJECT`` and ``M1_Contact_Img`` columns by header text,
and returns ``{subject_code: anchor_position}`` — codes stay local,
never committed.

Numbers come back as floats (``openpyxl``-free); header matching is
exact after whitespace stripping.
"""

from __future__ import annotations

import re
import zipfile
from xml.etree import ElementTree

from ..errors import DataValidationError

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")


def _column_letters(ref: str) -> str:
    match = _CELL_REF.match(ref)
    if not match:
        return ""
    return match.group(1)


def _column_index(letters: str) -> int:
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("m:si", _NS):
        parts = [
            t.text or "" for t in item.iter(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        ]
        strings.append("".join(parts))
    return strings


def _cell_value(cell, shared: list[str]) -> str:
    kind = cell.get("t")
    value_node = cell.find("m:v", _NS)
    if kind == "s" and value_node is not None:
        idx = int(value_node.text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    if kind == "inlineStr":
        node = cell.find("m:is/m:t", _NS)
        return node.text if node is not None and node.text else ""
    return value_node.text if value_node is not None and value_node.text else ""


def read_workbook_columns(
    xlsx_path, wanted: tuple[str, ...],
    sheet_path: str = "xl/worksheets/sheet1.xml",
) -> dict[str, dict[str, str]]:
    """Return ``{subject_code: {column_name: raw_value}}`` for the
    requested header columns of the first worksheet (header row located
    by exact text). Missing columns raise; missing values are simply
    absent from the inner dict (never guessed)."""
    try:
        with zipfile.ZipFile(xlsx_path) as archive:
            shared = _shared_strings(archive)
            root = ElementTree.fromstring(archive.read(sheet_path))
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DataValidationError(
            f"localization workbook unreadable: {type(exc).__name__}"
        ) from exc
    rows: list[dict[int, str]] = []
    for row in root.findall("m:sheetData/m:row", _NS):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", _NS):
            letters = _column_letters(cell.get("r", ""))
            if letters:
                cells[_column_index(letters)] = _cell_value(cell, shared)
        rows.append(cells)
    if not rows:
        raise DataValidationError("localization workbook has no rows")
    header = rows[0]
    subject_col = next((i for i, v in header.items() if v.strip() == "SUBJECT"), None)
    if subject_col is None:
        raise DataValidationError("localization workbook lacks a 'SUBJECT' column")
    column_index = {}
    for name in wanted:
        if name == "SUBJECT":
            continue
        found = next((i for i, v in header.items() if v.strip() == name), None)
        if found is None:
            raise DataValidationError(f"localization workbook lacks column {name!r}")
        column_index[name] = found
    table: dict[str, dict[str, str]] = {}
    for cells in rows[1:]:
        subject = cells.get(subject_col, "").strip()
        if not subject:
            continue
        table[subject] = {
            name: cells.get(idx, "").strip()
            for name, idx in column_index.items()
        }
    return table


def read_anchor_positions(xlsx_path) -> dict[str, float]:
    """``{subject_code: motor-anchor position}`` (M1_Contact_Img)."""
    table = read_workbook_columns(xlsx_path, ("SUBJECT", "M1_Contact_Img"))
    positions: dict[str, float] = {}
    for subject, columns in table.items():
        raw = columns.get("M1_Contact_Img", "")
        try:
            positions[subject] = float(raw)
        except ValueError:
            continue  # e.g. 'No Postop CT' — recorded absent, never guessed
    return positions


def read_signal_quality_flags(xlsx_path) -> dict[str, float | None]:
    """``{subject_code: signal-quality flag}`` (retained + reported; the
    frozen rule never excludes on it)."""
    table = read_workbook_columns(xlsx_path, ("SUBJECT", "Signal quality"))
    flags: dict[str, float | None] = {}
    for subject, columns in table.items():
        raw = columns.get("Signal quality", "")
        try:
            flags[subject] = float(raw)
        except ValueError:
            flags[subject] = None
    return flags

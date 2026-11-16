#!/usr/bin/env python3
"""Read-only structural inspection of a raw-layout recording directory.

Prints the MATLAB variable inventory (names, shapes, types) for ``.mat``
files, and — for an Excel workbook — the sheet names, columns, and a
two-row preview. Pass paths as arguments; nothing is modified, no
signal values are printed, only structure metadata (this keeps subject
data out of any log).

Examples:

    python scripts/inspect_raw_layout.py \
        /local/data/DBS_bG01/filtered_downsampled_Data_FT_format.mat
    python scripts/inspect_raw_layout.py /local/data/ECoG_localization.xlsx
    python scripts/inspect_raw_layout.py /local/data/            # survey dir
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_PREVIEW_CHARS = 600


def inspect_mat(path: Path) -> None:
    from scipy.io import whosmat

    print(f"== {path} ==")
    try:
        for name, shape, kind in whosmat(str(path)):
            print(f"  {name}: {shape} ({kind})")
    except Exception as exc:  # noqa: BLE001 — report and continue surveying
        print(f"  ERROR: {exc.__class__.__name__}: {exc}")


def inspect_workbook(path: Path) -> None:
    import pandas as pd

    print(f"== {path} ==")
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc.__class__.__name__}: {exc}")
        return
    for sheet in xl.sheet_names:
        frame = xl.parse(sheet, nrows=3)
        print(f"  sheet {sheet!r}: {len(frame.columns)} columns")
        print(f"    columns: {list(frame.columns)[:24]}")
        preview = frame.head(2).to_string()[:MAX_PREVIEW_CHARS]
        for line in preview.splitlines():
            print(f"    {line}")


def survey_dir(path: Path) -> None:
    print(f"== survey {path} ==")
    entries = sorted(path.iterdir())
    print(f"  {len(entries)} entries; first 10:")
    for entry in entries[:10]:
        kind = "dir " if entry.is_dir() else "file"
        print(f"    [{kind}] {entry.name}")
    for child in entries:
        if child.is_dir():
            inner = sorted(child.iterdir())
            names = ", ".join(item.name for item in inner[:6])
            print(f"  {child.name}/: {len(inner)} items ({names}...)")


def main(argv: list[str] | None = None) -> int:
    targets = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not targets:
        print(__doc__)
        return 2
    for target in targets:
        if target.is_dir():
            survey_dir(target)
        elif target.suffix.lower() == ".mat":
            inspect_mat(target)
        elif target.suffix.lower() in {".xlsx", ".xls"}:
            inspect_workbook(target)
        else:
            print(f"== {target} == (not a directory/.mat/.xlsx — skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

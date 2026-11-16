#!/usr/bin/env python3
"""Convert the raw DABI pilot layout into the kinematic .npz contract.

Input layout (as shipped in the authorized local copy):

* ``--raw-dir`` — one folder per subject, each containing
  ``filtered_downsampled_Data_FT_format.mat`` with three variables:
  ``data`` (FieldTrial struct: ``label``, ``fsample``, ``trial`` — the
  17-channel ECoG matrix: RDBS/LDBS contacts, the P0–P7 ECoG strip,
  aux), ``data_glove`` (FieldTrip struct: 5 glove sensors on the same
  sample grid), and ``restmove_segs`` (n_blocks × 4; columns 1–2 are
  0-based half-open sample bounds, column 4 is the block kind
  0 = REST / 1 = MOVE).
* ``--workbook`` — the localization workbook; ``Sheet1`` holds one row
  per subject with the prespecified covariates ``Age``, ``FREQ STIM``,
  ``VOLT STIM`` (first row per subject is used).

Output: one ``<subject>.npz`` per recording in the documented
``build_kinematic_table`` contract (subject, fs_ecog, fs_glove,
restmove_segs, ecog [P0–P7 strip only], glove, age, dbs_volt,
dbs_freq), written into the given output directory (gitignored
``outputs/`` by default).

Exclusions are printed explicitly with reasons and never silent: a
subject folder without the standard .mat, a subject missing from the
workbook, missing/NaN covariates, or no ECoG strip channels. The
original data files are only ever read.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from curve_icon_dbso.errors import CurveIconError

STANDARD_MAT = "filtered_downsampled_Data_FT_format.mat"
ECoG_LABEL_RE = re.compile(r"^P\d+$")
COVARIATES = ("age", "dbs_volt", "dbs_freq")
WORKBOOK_COLUMNS = {"age": "Age", "dbs_volt": "VOLT STIM", "dbs_freq": "FREQ STIM"}

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): keep raw data, subject codes, and "
    "per-subject outputs out of version control (gitignored data/, "
    "outputs/, artifacts/ are the designated local locations); findings "
    "are published as findings, never as data. See docs/data_access.md. "
    "This reminder never prevents a run."
)


def _labels(label_field) -> list[str]:
    labels: list[str] = []
    for item in np.asarray(label_field).ravel():
        if isinstance(item, np.ndarray):
            labels.append(str(item.ravel()[0]))
        else:
            labels.append(str(item))
    return labels


def _trial_matrix(trial_field, n_channels: int) -> np.ndarray:
    arr = np.asarray(trial_field)
    if arr.dtype == object:
        # FieldTrip stores trials as a MATLAB cell array: each item is a
        # (channels x samples) or (samples,) numeric array.
        rows = [np.asarray(item, dtype=float).squeeze() for item in arr.ravel()]
        matrix = np.vstack([np.atleast_2d(row) for row in rows])
    else:
        matrix = arr.astype(float).squeeze()
        if matrix.ndim == 0:  # a (1,1)-wrapped single row
            matrix = matrix.reshape(1, -1)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] != n_channels:
        # Channel-major vs sample-major safety: labels decide the axis.
        if matrix.shape[1] == n_channels:
            matrix = matrix.T
        else:
            raise CurveIconError(
                f"trial matrix {matrix.shape} does not match {n_channels} labels"
            )
    return matrix


def load_recording(mat_path: Path) -> tuple[list[str], np.ndarray, float, np.ndarray, list[str], np.ndarray, float]:
    """Return (ecog_labels, ecog, fs_ecog, restmove_segs, glove_labels, glove, fs_glove)."""
    mat = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    for key in ("data", "data_glove", "restmove_segs"):
        if key not in mat:
            raise CurveIconError(f"{mat_path.name}: missing variable '{key}'")

    data = mat["data"][0, 0]
    labels = _labels(data.label)
    fs_ecog = float(np.asarray(data.fsample).squeeze())
    ecog_all = _trial_matrix(data.trial, len(labels))

    ecog_idx = [i for i, label in enumerate(labels) if ECoG_LABEL_RE.match(label)]
    if not ecog_idx:
        raise CurveIconError(f"{mat_path.name}: no ECoG strip channels (P0–P7) found")
    ecog = ecog_all[ecog_idx]
    ecog_labels = [labels[i] for i in ecog_idx]

    glove = mat["data_glove"][0, 0]
    glove_labels = _labels(glove.label)
    fs_glove = float(np.asarray(glove.fsample).squeeze())
    glove_matrix = _trial_matrix(glove.trial, len(glove_labels))

    segs = np.asarray(mat["restmove_segs"], dtype=float)
    if segs.ndim != 2 or segs.shape[1] < 4:
        raise CurveIconError(
            f"{mat_path.name}: restmove_segs must be (n_blocks, >=4); got {segs.shape}"
        )
    return ecog_labels, ecog, fs_ecog, segs, glove_labels, glove_matrix, fs_glove


def load_covariates(workbook: Path, subject: str) -> dict[str, float]:
    sheet1 = pd.ExcelFile(workbook).parse("Sheet1")
    rows = sheet1.loc[sheet1["SUBJECT"].astype(str) == subject]
    if rows.empty:
        raise CurveIconError(f"{subject}: no row in the localization workbook")
    first = rows.iloc[0]
    values: dict[str, float] = {}
    for name, column in WORKBOOK_COLUMNS.items():
        if column not in sheet1.columns:
            raise CurveIconError(f"{subject}: workbook column '{column}' missing")
        value = pd.to_numeric(first[column], errors="coerce")
        if not np.isfinite(value):
            raise CurveIconError(f"{subject}: covariate '{column}' is missing/NaN")
        values[name] = float(value)
    return values


def convert(raw_dir: Path, workbook: Path, output_dir: Path) -> pd.DataFrame:
    """Convert every standard subject folder; return a summary table."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise CurveIconError(
            f"Raw directory not found: {raw_dir}. Point --raw-dir at your "
            "local authorized copy (read-only; the data never moves)."
        )
    if not Path(workbook).is_file():
        raise CurveIconError(f"Localization workbook not found: {workbook}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for subject_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        subject = subject_dir.name
        mat_path = subject_dir / STANDARD_MAT
        try:
            if not mat_path.is_file():
                raise CurveIconError(f"no '{STANDARD_MAT}' in folder")
            ecog_labels, ecog, fs_ecog, segs, glove_labels, glove, fs_glove = (
                load_recording(mat_path)
            )
            covariates = load_covariates(Path(workbook), subject)
        except CurveIconError as exc:
            print(f"EXCLUDED: {subject} — {exc}")
            continue

        npz_path = output_dir / f"{subject}.npz"
        np.savez(
            npz_path,
            subject=subject,
            fs_ecog=fs_ecog,
            fs_glove=fs_glove,
            restmove_segs=segs,
            ecog=ecog,
            glove=glove,
            **covariates,
        )
        n_blocks = int(segs.shape[0])
        n_move = int((segs[:, 3] == 1).sum())
        print(
            f"wrote {npz_path}: ecog={ecog.shape} {ecog_labels}, "
            f"glove={glove.shape}, segs={n_blocks} blocks ({n_move} MOVE), "
            f"fs={fs_ecog:g}/{fs_glove:g}"
        )
        summary_rows.append({
            "subject": subject,
            "n_ecog_channels": int(ecog.shape[0]),
            "n_samples": int(ecog.shape[1]),
            "n_blocks": n_blocks,
        })

    if not summary_rows:
        raise CurveIconError("No subject could be converted; see the printed exclusions.")
    return pd.DataFrame(summary_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert-raw-to-npz",
        description=(
            "Convert the raw DABI pilot layout into per-recording .npz "
            "files for build_kinematic_table.py. Read-only on the data."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory of per-subject folders with "
                             "filtered_downsampled_Data_FT_format.mat files")
    parser.add_argument("--workbook", type=Path, required=True,
                        help="Localization workbook (.xlsx) with Sheet1 covariates")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kinematic_npz"),
                        help="Where to write per-recording .npz files "
                             "(default: outputs/kinematic_npz, gitignored)")
    args = parser.parse_args(argv)

    print(DATA_USE_REMINDER, file=sys.stderr)
    try:
        summary = convert(args.raw_dir, args.workbook, args.output_dir)
        print(f"Converted {len(summary)} recordings into {args.output_dir}")
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

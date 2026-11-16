#!/usr/bin/env python3
"""Build the per-recording kinematic feature table from local recordings.

The evaluation entry point (``scripts/evaluate_real_kinematics.py``)
consumes a plain per-recording CSV. This script is the extraction half
of that pipeline: it walks a directory of per-recording ``.npz`` files,
runs the prespecified kinematic extraction
(:func:`curve_icon_dbso.kinematics.extract_kinematic_recording`), and
writes the feature table.

**File contract (this project's neutral interchange format):** one
``.npz`` per recording with keys

* ``subject``         — recording identifier (string)
* ``fs_ecog``         — ECoG sampling rate in Hz (float)
* ``fs_glove``        — glove sampling rate in Hz (float)
* ``restmove_segs``   — ``(n_blocks, >=4)`` array; columns 1–2 are
  0-based half-open sample bounds, column 4 is the block kind
  (0 = REST, 1 = MOVE) — exactly the methodology's layout
* ``ecog``            — ``(n_channels, n_samples)`` array
* ``glove``           — ``(n_channels, n_samples)`` array
* ``age``, ``dbs_volt``, ``dbs_freq`` — the prespecified clinical
  covariates (methodology: localization workbook, first row per
  subject) as numeric scalars

``restmove_segs`` bounds are applied sample-wise to both arrays, so
``ecog`` and ``glove`` must share the block grid of ``restmove_segs``
(store the arrays on a common time base, pre-sliced per block grid).

This format is deliberately source-agnostic: adapting a concrete public
dataset later means writing a thin loader from that source's raw tables
into these files — no claims are made here about any real source.

QC behavior (methodology): recordings failing tap QC (fewer than 2
contributing channel-blocks, no MOVE blocks, unusable glove data) are
**excluded from the emitted table** and reported explicitly with their
QC reason — never imputed, never zero-filled. Data and outputs live in
local/gitignored locations (``data/``, ``outputs/``, ``artifacts/`` or
any local path) and are simply never committed; an in-repo output path
outside those directories prints a warning, and a non-blocking data-use
reminder is printed on every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from curve_icon_dbso.errors import CurveIconError
from curve_icon_dbso.kinematics import (
    KINEMATIC_FEATURES,
    extract_kinematic_recording,
    resolve_output_dir,
)

REQUIRED_KEYS = (
    "subject",
    "fs_ecog",
    "fs_glove",
    "restmove_segs",
    "ecog",
    "glove",
    # Prespecified clinical covariates (methodology: localization workbook,
    # first row per subject) — required predictors, never imputed here:
    "age",
    "dbs_volt",
    "dbs_freq",
)

QC_COLUMNS = (
    "n_tap_contributing",
    "tap_qc_status",
    "tap_qc_reason",
    "n_rest_ok_records",
    "n_rest_failed_records",
)

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): keep raw data, subject codes, and "
    "per-subject outputs out of version control (gitignored data/, "
    "outputs/, artifacts/ are the designated local locations); findings "
    "are published as findings, never as data. See docs/data_access.md. "
    "This reminder never prevents a run."
)


def build_table(input_dir: Path) -> pd.DataFrame:
    """Extract every ``*.npz`` recording in ``input_dir`` into feature rows.

    QC-failed recordings are dropped from the returned frame (they are
    reported by :func:`main`, which keeps their explicit QC records).
    Raises :class:`CurveIconError` on a missing directory, no recordings,
    or a file missing required keys.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise CurveIconError(
            f"Input directory not found: {input_dir}. The script never ships "
            "with or searches for data — point --input-dir at your local copy."
        )
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise CurveIconError(
            f"No *.npz recordings found in {input_dir}. See the module "
            "docstring for the per-recording file contract."
        )

    rows: list[dict] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                raise CurveIconError(
                    f"{path.name}: missing required keys {missing} — see the "
                    "module docstring for the per-recording file contract."
                )
            subject = str(np.asarray(data["subject"]).reshape(-1)[0])
            # Prespecified covariates come straight from the recording
            # file; HFO/tap features come from the extraction:
            covariates = {
                covariate: float(data[covariate])
                for covariate in ("age", "dbs_volt", "dbs_freq")
            }
            record = extract_kinematic_recording(
                ecog=np.asarray(data["ecog"], dtype=float),
                glove=np.asarray(data["glove"], dtype=float),
                fs_ecog=float(data["fs_ecog"]),
                fs_glove=float(data["fs_glove"]),
                restmove_segs=np.asarray(data["restmove_segs"], dtype=float),
            )
        row: dict = {"subject": subject, **covariates}
        hfo_features = [name for name in KINEMATIC_FEATURES
                        if name.startswith("hfo_")]
        row.update({name: record[name] for name in hfo_features})
        row.update({"tap_rate_mean": record["tap_rate_mean"]})
        row.update({name: record[name] for name in QC_COLUMNS})
        if record["tap_qc_status"] == "ok":
            rows.append(row)
        else:
            print(
                f"EXCLUDED (tap_qc_status=failed): {path.name} "
                f"(subject {row['subject']}) — {record['tap_qc_reason']}"
            )
    if not rows:
        raise CurveIconError(
            "Every recording failed tap QC; no feature rows were produced. "
            "See the printed QC reasons above."
        )
    columns = ["subject", *KINEMATIC_FEATURES, "tap_rate_mean", *QC_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build-kinematic-table",
        description=(
            "Extract the prespecified kinematic features from local "
            "per-recording .npz files into the feature table consumed by "
            "evaluate_real_kinematics.py."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing one .npz file per recording "
                             "(see the module docstring for the key contract)")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Where to write the per-recording feature table "
                             "(gitignored data/, outputs/, artifacts/ are fine)")
    args = parser.parse_args(argv)

    print(DATA_USE_REMINDER, file=sys.stderr)
    try:
        repo_root = Path(__file__).resolve().parents[1]
        output_csv = resolve_output_dir(args.output_csv, repo_root)
        table = build_table(args.input_dir)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"Wrote {len(table)} recording rows to {output_csv}")
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

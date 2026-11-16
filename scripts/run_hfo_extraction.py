#!/usr/bin/env python3
"""Run HFO feature extraction on locally supplied ECoG ``.mat`` data.

Point this at your local copy of the authorized dataset (see
docs/data_access.md for the authorization basis and the non-blocking
data guidance). The script never ships with or searches for data: a
missing input directory is a hard, actionable error — never a
placeholder. It prints a non-blocking reminder about the owner's
data-use responsibilities; nothing is gated on any register or
approval status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from curve_icon_dbso.config import HFOConfig
from curve_icon_dbso.errors import CurveIconError
from curve_icon_dbso.hfo import aggregate_to_subject, extract_dataset

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): keep raw data, subject codes, and "
    "per-subject outputs out of version control (gitignored data/, "
    "outputs/, artifacts/ are the designated local locations); findings "
    "are published as findings, never as data. See docs/data_access.md. "
    "This reminder never prevents a run."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-hfo-extraction",
        description="Extract channel-level HFO features from per-subject .mat folders.",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing one subfolder per subject with .mat files")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Where to write the channel-level feature table")
    parser.add_argument("--aggregated-csv", type=Path, default=None,
                        help="Optionally also write subject-level means (ok channels only)")
    parser.add_argument("--band-low", type=float, default=80.0, help="HFO band lower edge (Hz)")
    parser.add_argument("--band-high", type=float, default=500.0, help="HFO band upper edge (Hz)")
    parser.add_argument("--fs-override", type=float, default=None,
                        help="Sampling rate in Hz if not stored in the files")
    args = parser.parse_args(argv)

    print(DATA_USE_REMINDER, file=sys.stderr)
    try:
        band = HFOConfig(low_hz=args.band_low, high_hz=args.band_high)
        channels = extract_dataset(args.input_dir, band=band, fs_override=args.fs_override)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        channels.to_csv(args.output_csv, index=False)
        n_failed = int((channels["qc_status"] == "failed").sum())
        print(f"Wrote {len(channels)} channel rows to {args.output_csv} "
              f"({n_failed} failed QC — retained as 'failed', never zero-filled)")
        if args.aggregated_csv is not None:
            subjects = aggregate_to_subject(channels)
            args.aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
            subjects.to_csv(args.aggregated_csv, index=False)
            print(f"Wrote {len(subjects)} subject-level rows to {args.aggregated_csv}")
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""LOCAL-ONLY driver: the v2 adaptive-transfer study (all tracks).

Runs the complete frozen v2 matrix (``docs/adaptive_transfer_protocol.md``,
contract ``curve_icon_dbso.transfer.contract``) against the owner's
local authorized copy. Outputs go to a VERSIONED, gitignored run
directory (never overwrites earlier runs); provenance (git SHA, lock
hash, contract version + hash, input manifest SHA-256 values, seeds,
stage) is recorded in the run directory and never committed. Execution
is a deliberate owner action; nothing here gates on it.

Run modes (v2.1.0). Only ``--definitive`` runs — exact frozen null
draws, ablations on, comparators on — are named ``run_…`` and may write
the public results document. EVERY other run (altered draws, skipped
ablations, skipped comparators) is named ``smoke_…`` and can never
touch ``docs/results_adaptive_transfer_real.md``, regardless of the
draw count.

Example:
    python scripts/run_transfer_study.py \\
        --input-dir outputs/kinematic_npz \\
        --workbook "/local/path/ECoG_localization.xlsx" \\
        --output-dir outputs/transfer_v2 \\
        --definitive \\
        --acknowledge-data-use "authorized definitive run"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curve_icon_dbso.transfer import contract  # noqa: E402
from curve_icon_dbso.transfer.cohort import Recording, load_recording_npz  # noqa: E402
from curve_icon_dbso.transfer.pipeline import run_transfer_study  # noqa: E402
from curve_icon_dbso.transfer.summary import (  # noqa: E402
    build_run_summary,
    render_results_doc,
    write_calibration_curve,
    write_cohort_flow,
    write_summary,
)
from curve_icon_dbso.transfer.workbook import read_anchor_positions, read_signal_quality_flags  # noqa: E402

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): keep raw data, subject codes, and "
    "per-subject outputs out of version control (gitignored data/, "
    "outputs/, artifacts/ are the designated local locations); findings "
    "are published as findings, never as data. See docs/data_access.md. "
    "This reminder never prevents a run."
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.SubprocessError:
        return "UNKNOWN"


def _lock_hash() -> str:
    lock = ROOT / "uv.lock"
    if lock.is_file():
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    return "UNKNOWN"


def _input_manifest(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _definitive_violations(args: argparse.Namespace) -> list[str]:
    """Every frozen requirement a non-definitive run's settings violate
    (``contract.DEFINITIVE_RUN_REQUIREMENTS``)."""
    violations: list[str] = []
    requirements = contract.DEFINITIVE_RUN_REQUIREMENTS
    draws = args.null_draws
    if draws is not None and draws != requirements["detection_null_draws"]:
        violations.append(
            f"null draws {draws} != the frozen "
            f"{requirements['detection_null_draws']}")
    if args.no_ablations and requirements["ablations_required"]:
        violations.append("ablations skipped (--no-ablations)")
    if args.no_xgboost and requirements["xgboost_comparators_required"]:
        violations.append("XGBoost comparators skipped (--no-xgboost)")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-transfer-study",
        description="Run the frozen v2 adaptive-transfer study (all tracks) "
                    "against the local authorized copy.",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of converted per-recording .npz files")
    parser.add_argument("--workbook", type=Path, required=True,
                        help="Localization workbook (.xlsx) with the "
                             "motor-anchor positions (M1_Contact_Img)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/transfer_v2"),
                        help="Versioned gitignored output root")
    parser.add_argument("--acknowledge-data-use", type=str, default="",
                        help="Optional note recorded in the run provenance")
    parser.add_argument("--null-draws", type=int, default=None,
                        help="Override the frozen null draw counts (default: "
                             "the contract values; any override forces the "
                             "non-definitive path)")
    parser.add_argument("--no-ablations", action="store_true",
                        help="Skip the prespecified ablations (definitive "
                             "runs must not pass this)")
    parser.add_argument("--no-xgboost", action="store_true",
                        help="Skip the optional XGBoost comparators (also "
                             "applied to Track A; definitive runs must not "
                             "pass this)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel LOSO folds / feature builds across "
                             "spawned worker processes (pure performance; "
                             "identical scientific output; each worker holds "
                             "one copy of the in-memory payload; 1 = serial)")
    parser.add_argument("--definitive", action="store_true",
                        help="Declare the DEFINITIVE run: requires exact "
                             "frozen null draws, ablations, and comparators "
                             "(rejects the launch otherwise). Only a "
                             "definitive run may write the public results "
                             "document.")
    parser.add_argument("--write-results-doc", action="store_true",
                        help="Also write docs/results_adaptive_transfer_real.md "
                             "(definitive runs only)")
    args = parser.parse_args(argv)

    if args.write_results_doc and not args.definitive:
        print("ERROR: --write-results-doc requires --definitive; a "
              "non-definitive run can never overwrite the public results "
              "document.", file=sys.stderr)
        return 2
    if args.definitive:
        violations = _definitive_violations(args)
        if violations:
            print("ERROR: --definitive rejected — the launch deviates from "
                  "the frozen contract:", file=sys.stderr)
            for line in violations:
                print(f"  - {line}", file=sys.stderr)
            return 2
    definitive = bool(args.definitive)

    print(DATA_USE_REMINDER, file=sys.stderr)
    started = time.time()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Collision-proof run directory: millisecond stamp + short uuid, and
    # never overwrite an existing directory.
    run_dir = args.output_dir / (
        f"{'run' if definitive else 'smoke'}_{contract.CONTRACT_VERSION}"
        f"_{stamp}_{uuid.uuid4().hex[:8]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[transfer-v2] run directory: {run_dir}"
          + ("" if definitive else "  (NON-DEFINITIVE / smoke)"))

    # ---- load cohort (ordinal labels only; codes never printed) ----
    anchor_positions = read_anchor_positions(args.workbook)
    try:
        quality_flags = read_signal_quality_flags(args.workbook)
    except Exception:  # noqa: BLE001 — the flag column is optional at load
        quality_flags = {}
    npz_paths = sorted(args.input_dir.glob("*.npz"))
    if not npz_paths:
        print("ERROR: no converted .npz recordings in the input directory",
              file=sys.stderr)
        return 1
    recordings: list[Recording] = []
    excluded_load: list[str] = []
    for position, path in enumerate(npz_paths, start=1):
        anchor = anchor_positions.get(path.stem)
        if anchor is None or not np.isfinite(anchor):
            excluded_load.append(
                f"recording #{position:03d}: no workbook anchor position")
            continue
        try:
            recordings.append(load_recording_npz(
                path, position, int(anchor),
                signal_quality_flag=quality_flags.get(path.stem)))
        except Exception as exc:  # noqa: BLE001 — record and continue
            excluded_load.append(
                f"recording #{position:03d}: {type(exc).__name__}: {exc}")
    for line in excluded_load:
        print(f"[transfer-v2] load-excluded {line}")
    print(f"[transfer-v2] loaded {len(recordings)} recordings")

    # ---- run the frozen matrix ----
    artifacts = run_transfer_study(
        recordings,
        with_ablations=not args.no_ablations,
        null_draws=args.null_draws,
        with_xgboost=not args.no_xgboost,
        workers=args.workers,
    )
    summary = build_run_summary(artifacts)
    # The input manifest (filename → SHA-256) is LOCAL-ONLY: real data
    # filenames carry subject codes, so the aggregate summary records
    # only the manifest file's own hash.
    manifest = _input_manifest(npz_paths)
    manifest_path = run_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary["provenance"] = {
        "git_sha": _git_sha(),
        "lock_hash": _lock_hash(),
        "contract_version": contract.CONTRACT_VERSION,
        "contract_hash": contract.contract_hash(),
        "v1_contract_version_immutable": contract.V1_CONTRACT_VERSION,
        "stage": contract.STAGE,
        "run_mode": "definitive" if definitive else "smoke",
        "definitive": bool(definitive),
        "definitive_checklist": {
            "null_draws_exact": args.null_draws is None,
            "ablations": not args.no_ablations,
            "xgboost": not args.no_xgboost,
        },
        "null_draws_override": args.null_draws,
        "smoke_run": not definitive,
        "ablations": not args.no_ablations,
        "xgboost": not args.no_xgboost,
        "workers": args.workers,
        "data_use_note": args.acknowledge_data_use or None,
        "n_inputs": len(npz_paths),
        "input_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "started_utc": stamp,
        "duration_s": round(time.time() - started, 1),
        "n_recordings_loaded": len(recordings),
        "n_load_excluded": len(excluded_load),
    }
    summary["provenance"]["seeds"] = {
        "bootstrap": contract.BOOTSTRAP_SEED,
        "null": contract.NULL_SEED,
    }

    # ---- write artifacts (aggregate to be committable; per-subject
    # tables stay in this gitignored directory) ----
    write_summary(summary, run_dir)
    write_cohort_flow(artifacts, run_dir)
    write_calibration_curve(artifacts, run_dir)
    doc = render_results_doc(summary, str(run_dir))
    (run_dir / "results_aggregate.md").write_text(doc)
    if args.write_results_doc:
        assert definitive  # gated above
        (ROOT / "docs" / "results_adaptive_transfer_real.md").write_text(doc)
        print("[transfer-v2] wrote docs/results_adaptive_transfer_real.md")

    verdicts = summary["verdicts"]
    print(json.dumps(verdicts, indent=2, sort_keys=True, default=str))
    print(f"[transfer-v2] done in {time.time() - started:.1f} s -> {run_dir}")
    if not definitive:
        print("[transfer-v2] NOTE: non-definitive settings — SMOKE run; "
              "it must not be presented as definitive and cannot write "
              "the public results document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Follow-on MOVE methodology-transfer study on external (authorized)
data — LOCAL-ONLY command, implemented but NOT executed by any test or
agent phase. Running it is a deliberate owner action against the local
authorized copy (docs/data_access.md); nothing is gated on it.

Builds the frozen windowed dataset per recording (500 ms windows, 100 ms
updates, the frozen lag stack; windows never cross block boundaries),
runs the three prespecified decoders with subject-held-out folds and
inner subject-grouped selection, computes per-subject-first metrics, the
autocorrelation-preserving circular-shift null, the lag and band-block
ablations, and writes sanitized aggregates + plots to a VERSIONED
run directory (never overwrites earlier runs, including the recorded
first authorized run and the subject-level follow-on run).

Example (owner, local machine, against the local authorized copy):

    python scripts/run_followon_move_modeling.py \
      --input-dir /local/outside/repo/all_subs_preprocessed_data_npz \
      --output-dir outputs/followon_move_modeling \
      --stage exploratory \
      --acknowledge-data-use "USC DABI pilot collection; owner distributor agreement." \
      --shifts 1000

Input contract: the same per-recording ``.npz`` layout as
``scripts/build_kinematic_table.py``. Participant identifiers never
appear in tracked files — per-subject predictions and metadata stay in
the gitignored output directory; only aggregates whose publication the
recorded terms allow may ever be promoted into tracked docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from curve_icon_dbso.dmd_representation import (
    dmd_edge_features,
    dmd_feature_frame,
    run_representation_comparison,
)
from curve_icon_dbso.errors import CurveIconError
from curve_icon_dbso.glove_phenotyping import glove_move_envelope
from curve_icon_dbso.kinematics import parse_restmove_blocks, resolve_output_dir
from curve_icon_dbso.move_modeling import (
    LAG_ABLATIONS,
    MOVE_FEATURE_BLOCKS,
    WindowDataset,
    bootstrap_median_r,
    build_window_dataset,
    move_circular_shift_null,
    move_transfer_criterion,
    plot_move_results,
    run_move_cv,
)
from curve_icon_dbso.protocol import (
    PROTOCOL_VERSION,
    protocol_hash,
    resolve_spectral_bands,
)

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): keep raw data, subject codes, and "
    "per-subject outputs out of version control (gitignored data/, "
    "outputs/, artifacts/ are the designated local locations); findings "
    "are published as findings, never as data. See docs/data_access.md. "
    "This reminder never prevents a run."
)

REQUIRED_NPZ_KEYS = ("subject", "fs_ecog", "fs_glove", "restmove_segs",
                     "ecog", "glove")


def _input_manifest(input_dir: Path) -> dict[str, dict]:
    """Per-file byte counts and SHA-256 digests of the run's inputs —
    the identical-inputs guarantee is verifiable, not asserted."""
    manifest: dict[str, dict] = {}
    for path in sorted(input_dir.glob("*.npz")):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                size += len(chunk)
                digest.update(chunk)
        manifest[path.name] = {"bytes": size, "sha256": digest.hexdigest()}
    if not manifest:
        raise CurveIconError(f"no .npz recordings in {input_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-followon-move-modeling",
        description=(
            "LOCAL-ONLY frozen MOVE methodology-transfer study; writes "
            "sanitized aggregates and plots to a versioned gitignored "
            "run directory."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of per-recording .npz files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base output directory (default "
                             "outputs/followon_move_modeling)")
    parser.add_argument("--stage", choices=("exploratory", "confirmatory"),
                        default="exploratory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shifts", type=int, default=1000,
                        help="Circular-shift null draws (contract floor)")
    parser.add_argument("--with-xgboost", action="store_true",
                        help="Optional DD-003 comparator (skipped if absent)")
    parser.add_argument("--with-dmd", action="store_true",
                        help="Optional DMD representation comparison "
                             "(multiband power vs DMD nodes vs combined; "
                             "edges only with --with-dmd-edges)")
    parser.add_argument("--with-dmd-edges", action="store_true",
                        help="Secondary DMD spatial-edge ablation (much "
                             "larger dimensionality; never a primary block)")
    parser.add_argument("--acknowledge-data-use", default=None,
                        help="Optional provenance note (never required)")
    return parser


def _versioned_output_dir(base: Path) -> Path:
    candidate = base / f"v{PROTOCOL_VERSION}"
    if not candidate.exists():
        return candidate
    index = 2
    while (base / f"v{PROTOCOL_VERSION}_run{index}").exists():
        index += 1
    return base / f"v{PROTOCOL_VERSION}_run{index}"


def load_window_datasets(input_dir: Path,
                         capture_windows: bool = False,
                         ) -> tuple[list[WindowDataset], dict]:
    """Per-recording .npz files → frozen window datasets (one per
    recording; subject positions are synthetic labels derived from the
    file order — participant codes stay in the local .npz metadata only)."""
    if not input_dir.is_dir():
        raise CurveIconError(f"input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise CurveIconError(f"no .npz recordings in {input_dir}")
    min_fs = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            fs = float(data["fs_ecog"])
            min_fs = fs if min_fs is None else min(min_fs, fs)
    bands = resolve_spectral_bands(min_fs)
    datasets: list[WindowDataset] = []
    per_recording_qc: dict = {}
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
            if missing:
                raise CurveIconError(f"{path.name}: missing keys {missing}")
            ecog = np.asarray(data["ecog"], dtype=float)
            glove = np.asarray(data["glove"], dtype=float)
            segs = np.asarray(data["restmove_segs"], dtype=float)
            fs = float(data["fs_ecog"])
            envelope = glove_move_envelope(glove, float(data["fs_glove"]), segs)
        try:
            dataset = build_window_dataset(
                ecog=ecog, fs=fs, restmove_segs=segs, envelope=envelope,
                subject_position=f"pos{index:03d}",
                recording_position=f"rec{index:03d}",
                bands_resolved=bands,
                capture_windows=capture_windows,
            )
        except CurveIconError as exc:
            # QC doctrine: a recording with zero usable windows is an
            # explicit exclusion — the run continues, the failure stays
            # visible, nothing is imputed.
            per_recording_qc[f"rec{index:03d}"] = {
                "excluded": True,
                "reason": str(exc),
                "envelope_qc": {k: v for k, v in envelope.items()
                                if k not in ("stream", "block_ids")},
            }
            print(f"EXCLUDED (no usable windows): recording index {index}")
            continue
        datasets.append(dataset)
        per_recording_qc[f"rec{index:03d}"] = {
            "envelope_qc": {k: v for k, v in envelope.items()
                            if k not in ("stream", "block_ids")},
            "window_qc": dataset.qc,
            "n_windows": int(dataset.features.shape[0]),
            "n_move_blocks": int(len(parse_restmove_blocks(segs)) and len(
                [b for b in parse_restmove_blocks(segs) if b[2] == "move"])),
        }
    return datasets, {"per_recording": per_recording_qc, "bands": bands}


def merge_datasets(datasets: list[WindowDataset]) -> WindowDataset:
    """Concatenate per-recording datasets into one (positional index
    continuity preserved for the fold machinery)."""
    features = pd.concat([d.features for d in datasets], ignore_index=True)
    target = np.concatenate([d.target for d in datasets])
    groups = np.concatenate([d.groups for d in datasets])
    metadata = pd.concat([d.metadata for d in datasets], ignore_index=True)
    qc = {"dropped_windows": sum(d.qc["dropped_windows"] for d in datasets),
          "reasons": {}}
    for dataset in datasets:
        for reason, count in dataset.qc["reasons"].items():
            qc["reasons"][reason] = qc["reasons"].get(reason, 0) + count
    return WindowDataset(features=features, target=target, groups=groups,
                         metadata=metadata, qc=qc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    print(DATA_USE_REMINDER, file=sys.stderr)
    if args.acknowledge_data_use:
        print(f"Data-use note (recorded in provenance): {args.acknowledge_data_use}")
    try:
        base = args.output_dir or repo_root / "outputs" / "followon_move_modeling"
        output_dir = resolve_output_dir(base, repo_root)
        output_dir = _versioned_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory (versioned, never overwrites): {output_dir}")

        datasets, qc = load_window_datasets(
            args.input_dir, capture_windows=args.with_dmd
        )
        dataset = merge_datasets(datasets)
        manifest = _input_manifest(args.input_dir)
        (output_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        print(f"Windows: {dataset.features.shape[0]} from "
              f"{len(datasets)} recordings; "
              f"dropped: {dataset.qc['dropped_windows']}")

        model_names = ("lagged_ridge_primary", "svr_linear_move")
        if args.with_xgboost:
            model_names = model_names + ("xgboost_shallow_move",)

        # Primary block × models on the full lag stack.
        results = run_move_cv(dataset, model_names=model_names,
                              block="multiband_power")
        rows = []
        for result in results:
            result.per_subject.to_csv(
                output_dir / f"per_subject_{result.model_name}.csv", index=False
            )
            rows.append({"model": result.model_name,
                         "block": result.block, **result.summary})
            (output_dir / f"baselines_{result.model_name}.json").write_text(
                json.dumps(result.baselines, indent=2)
            )
        pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)

        # Band-block ablation (primary decoder).
        block_ablation = {}
        for block in MOVE_FEATURE_BLOCKS:
            try:
                block_result = run_move_cv(dataset, model_names=(model_names[0],),
                                           block=block)
            except CurveIconError:
                continue  # band inactive at this cohort's fs
            block_ablation[block] = block_result[0]

        # Lag ablation (primary decoder).
        lag_ablation = {}
        for lag_name, lags in LAG_ABLATIONS.items():
            lag_result = run_move_cv(dataset, model_names=(model_names[0],),
                                     lags=lags)
            lag_ablation[lag_name] = lag_result[0]

        # Optional DMD representation comparison (frozen discipline:
        # identical folds, same primary decoder, held-out subjects only).
        if args.with_dmd:
            merged_windows: list = []
            for per_recording in datasets:
                merged_windows.extend(per_recording.windows or [])
            dmd_frame, dmd_diag = dmd_feature_frame(merged_windows)
            dmd_frame.to_csv(output_dir / "dmd_node_features.csv", index=False)
            # Diagnostics: the module's aggregate keys (mean one-step and
            # multi-step errors, computed fraction, unstable-mode mean) —
            # the multi-step error is None for unstable spectra, so raw
            # per-window lists are summarized, never mean-ed blindly.
            from collections import Counter

            diagnostics_document = {
                key: value for key, value in dmd_diag.items()
                if not isinstance(value, list)
            }
            diagnostics_document["reconstruction_status_counts"] = dict(
                Counter(dmd_diag["reconstruction_statuses"])
            )
            one_step = np.asarray(dmd_diag["one_step_relative_errors"], float)
            diagnostics_document["one_step_relative_error_min"] = (
                float(np.min(one_step)) if one_step.size else None
            )
            diagnostics_document["one_step_relative_error_median"] = (
                float(np.median(one_step)) if one_step.size else None
            )
            diagnostics_document["one_step_relative_error_max"] = (
                float(np.max(one_step)) if one_step.size else None
            )
            (output_dir / "dmd_diagnostics.json").write_text(json.dumps(
                diagnostics_document, indent=2, default=str
            ))
            comparison = run_representation_comparison(
                dataset, dmd_frame, model_name=model_names[0],
                with_edges=args.with_dmd_edges,
                edge_frame=(
                    pd.DataFrame([
                        dmd_edge_features(
                            window, [(0, 1), (1, 2), (2, 3)]
                        )[0] for window in merged_windows
                    ])
                    if args.with_dmd_edges else None
                ),
            )
            comparison_rows = []
            for representation, result in comparison.items():
                comparison_rows.append({
                    "representation": representation,
                    "model": result.model_name,
                    **result.summary,
                })
                result.per_subject.to_csv(
                    output_dir / f"per_subject_{representation}.csv",
                    index=False,
                )
            pd.DataFrame(comparison_rows).to_csv(
                output_dir / "representation_comparison.csv", index=False
            )

        # Autocorrelation-preserving nulls (primary decoder, full stack):
        # BOTH the prespecified circular null (corrected v1.0.1 effective-
        # displacement sampling) and the secondary blockwise null.
        null = move_circular_shift_null(
            dataset, model_name=model_names[0], n_shifts=args.shifts,
            seed=args.seed,
        )
        (output_dir / "circular_shift_null.json").write_text(json.dumps(
            {key: (value.tolist() if isinstance(value, np.ndarray) else value)
             for key, value in null.items()}, indent=2, default=str
        ))
        blockwise_null = move_circular_shift_null(
            dataset, model_name=model_names[0], n_shifts=args.shifts,
            seed=args.seed, blockwise=True,
        )
        (output_dir / "blockwise_shift_null.json").write_text(json.dumps(
            {key: (value.tolist() if isinstance(value, np.ndarray) else value)
             for key, value in blockwise_null.items()}, indent=2, default=str
        ))

        plots = plot_move_results(
            results, null=null, out_dir=output_dir,
            lag_ablation=lag_ablation, block_ablation=block_ablation,
        )
        # Frozen §12 transfer criterion — evaluated IN CODE from the
        # result's own components (v1.0.0 never computed the baseline-MAE
        # half; corrected in v1.0.1), plus the subject-cluster bootstrap
        # of the median r saved as a machine-readable artifact (draws,
        # seed, statistic, endpoints).
        transfer = move_transfer_criterion(results[0])
        bootstrap = bootstrap_median_r(
            results[0].per_subject["pearson_r"].to_numpy(dtype=float),
            n_resamples=2000, seed=args.seed,
        )
        (output_dir / "bootstrap_median_r.json").write_text(json.dumps(
            bootstrap, indent=2
        ))
        bootstrap_excludes_zero = (
            bootstrap["ci_low"] > 0 or bootstrap["ci_high"] < 0
        )
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "stage": args.stage,
            "seed": args.seed,
            "n_recordings": len(datasets),
            "n_windows": int(dataset.features.shape[0]),
            "dropped_windows": dataset.qc["dropped_windows"],
            "data_use_note": args.acknowledge_data_use,
            "plots": plots,
            "transfer_criterion": (
                "lagged Ridge: median across-subject Pearson r > 0 with a "
                "subject-cluster bootstrap 95% CI excluding 0 AND median MAE "
                "better than persistence and block-mean baselines (§12)"
            ),
            "transfer_criterion_evaluated": {
                **transfer,
                "bootstrap_median_r": {
                    "statistic": bootstrap["statistic"],
                    "ci_low": bootstrap["ci_low"],
                    "ci_high": bootstrap["ci_high"],
                    "ci_excludes_zero": bootstrap_excludes_zero,
                    "n_resamples": bootstrap["n_resamples"],
                    "seed": bootstrap["seed"],
                },
                "criterion_a_met": bool(
                    transfer["criterion_a_median_r_positive"]
                    and bootstrap_excludes_zero
                ),
                "nulls": {
                    "circular": {
                        "p_value": null["p_value"],
                        "n_shifts": null["n_shifts"],
                    },
                    "blockwise": {
                        "p_value": blockwise_null["p_value"],
                        "n_shifts": blockwise_null["n_shifts"],
                    },
                },
            },
            "input_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        from curve_icon_dbso import tracking

        summary["git_sha"] = tracking.git_sha(repo_root)
        summary["lock_hash"] = tracking.lock_hash(repo_root)
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(f"Run complete; sanitized outputs in {output_dir}")
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

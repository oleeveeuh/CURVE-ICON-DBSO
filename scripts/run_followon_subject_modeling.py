#!/usr/bin/env python
"""Follow-on subject-level REST/DELTA modeling on external (authorized)
data — LOCAL-ONLY command, implemented but NOT executed by any test or
agent phase. Running it is a deliberate owner action against the local
authorized copy (docs/data_access.md); it is never required for
repository work and nothing is gated on it.

Outputs go to a VERSIONED directory (default
``outputs/followon_subject_modeling/v<contract-version>_seed<seed>``);
an existing directory is never overwritten — a suffixed variant is
created instead, so the recorded first authorized run and every earlier
follow-on run are preserved exactly.

Example (owner, local machine, against the local authorized copy):

    python scripts/run_followon_subject_modeling.py \
      --input-dir /local/outside/repo/all_subs_preprocessed_data_npz \
      --output-dir outputs/followon_subject_modeling \
      --stage exploratory \
      --acknowledge-data-use "USC DABI pilot collection; owner distributor agreement." \
      --mlflow

The input contract is the same per-recording ``.npz`` layout as
``scripts/build_kinematic_table.py`` (subject, fs_ecog, fs_glove,
restmove_segs, ecog, glove, age, dbs_volt, dbs_freq). Everything is
computed with the frozen machinery: protocol.resolve_spectral_bands →
neural_features.build_subject_level_features → glove_phenotyping →
subject_modeling.run_arm (identical outer folds; inner-only selection) →
bootstrap/permutation/influence/stability, with provenance recorded
(git SHA, lock hash, protocol version + hash, stage, data-use note).
Participant-level tables stay in the gitignored output directory; only
aggregate findings whose publication the recorded terms allow may ever
be promoted into tracked docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from curve_icon_dbso.bootstrap import holm_adjusted_pvalues, subject_bootstrap_ci
from curve_icon_dbso.errors import CurveIconError
from curve_icon_dbso.glove_phenotyping import extract_glove_targets
from curve_icon_dbso.kinematics import resolve_output_dir
from curve_icon_dbso.neural_features import build_subject_level_features
from curve_icon_dbso.protocol import (
    HOLM_FAMILY_RULE,
    PRIMARY_SUBJECT_TARGET,
    PROTOCOL_VERSION,
    SECONDARY_SUBJECT_TARGETS,
    protocol_hash,
    resolve_spectral_bands,
)
from curve_icon_dbso.subject_modeling import (
    build_estimator,
    coefficient_sign_stability,
    leave_one_subject_influence,
    permutation_null_nested,
    promotion_table,
    resting_trait_evidence,
    run_arm,
    select_block,
    state_change_evidence,
)

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): keep raw data, subject codes, and "
    "per-subject outputs out of version control (gitignored data/, "
    "outputs/, artifacts/ are the designated local locations); findings "
    "are published as findings, never as data. See docs/data_access.md. "
    "This reminder never prevents a run."
)

REQUIRED_NPZ_KEYS = (
    "subject", "fs_ecog", "fs_glove", "restmove_segs", "ecog", "glove",
    "age", "dbs_volt", "dbs_freq",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-followon-subject-modeling",
        description=(
            "LOCAL-ONLY frozen REST/DELTA subject-level modeling on the "
            "authorized local data copy; writes versioned gitignored "
            "outputs and never overwrites earlier runs."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of per-recording .npz files (build "
                             "them with the documented converter)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base output directory (default "
                             "outputs/followon_subject_modeling)")
    parser.add_argument("--stage", choices=("exploratory", "confirmatory"),
                        default="exploratory",
                        help="Analysis stage label (SCI-011; the DABI cohort "
                             "is exploratory under the frozen contract)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Contract floor for reported tail probabilities")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--with-xgboost", action="store_true",
                        help="Optional DD-003 comparator (skipped if absent)")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log lineage to the local gitignored MLflow store")
    parser.add_argument("--acknowledge-data-use", default=None,
                        help="Optional provenance note recorded in the run "
                             "summary (never required)")
    return parser


def _versioned_output_dir(base: Path) -> Path:
    """Pick a run directory that does not exist yet — never overwrite."""
    candidate = base / f"v{PROTOCOL_VERSION}"
    if not candidate.exists():
        return candidate
    index = 2
    while (base / f"v{PROTOCOL_VERSION}_run{index}").exists():
        index += 1
    return base / f"v{PROTOCOL_VERSION}_run{index}"


def _input_manifest(input_dir: Path) -> dict:
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


def load_and_extract(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Per-recording .npz files → per-recording feature frame + QC."""
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

    rows: list[dict] = []
    qc_records: dict[str, dict] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
            if missing:
                raise CurveIconError(f"{path.name}: missing keys {missing}")
            recording_id = path.stem
            features = build_subject_level_features(
                ecog=np.asarray(data["ecog"], dtype=float),
                fs=float(data["fs_ecog"]),
                restmove_segs=np.asarray(data["restmove_segs"], dtype=float),
                bands_resolved=bands,
            )
            glove = extract_glove_targets(
                glove=np.asarray(data["glove"], dtype=float),
                fs_glove=float(data["fs_glove"]),
                restmove_segs=np.asarray(data["restmove_segs"], dtype=float),
            )
            # Covariates must be read INSIDE the npz context (the archive
            # closes when the block exits).
            covariates = {
                "subject": str(np.asarray(data["subject"]).reshape(-1)[0]),
                "age": float(data["age"]),
                "dbs_volt": float(data["dbs_volt"]),
                "dbs_freq": float(data["dbs_freq"]),
            }
            # The frozen five-feature reference predictors, computed with
            # the FROZEN implementation (never re-derived from the new
            # spectral features — that would redefine the comparator).
            from curve_icon_dbso.kinematics import extract_kinematic_recording

            legacy = extract_kinematic_recording(
                ecog=np.asarray(data["ecog"], dtype=float),
                glove=np.asarray(data["glove"], dtype=float),
                fs_ecog=float(data["fs_ecog"]),
                fs_glove=float(data["fs_glove"]),
                restmove_segs=np.asarray(data["restmove_segs"], dtype=float),
            )
            covariates["hfo_power_rest_mean"] = legacy["hfo_power_rest_mean"]
            covariates["hfo_peak_freq_rest_mean"] = legacy["hfo_peak_freq_rest_mean"]
        row: dict = {"recording": recording_id, **covariates}
        # REST/MOVE aggregated features carry the frozen block names
        # verbatim (theta_log_relative_power, hjorth_mobility, …); DELTA
        # keys are already disjoint from them (…_log_ratio_power,
        # …_delta, hfo_power_log_ratio). No prefixes — the frozen
        # select_block resolves them directly.
        for key, value in features["rest"]["features"].items():
            row[key] = value
        for key, value in features["delta"].items():
            row[key] = value
        row["tap_rate_mean"] = glove["primary"]["tap_rate_mean"]
        for target, record in glove["secondary"].items():
            row[target] = record["value"]
        qc_records[recording_id] = {
            "neural_qc": features["qc"], "glove_qc": glove["qc"],
            "glove_primary": {k: v for k, v in glove["primary"].items()
                              if k != "definition"},
            "meta": features["meta"],
        }
        rows.append(row)
    return pd.DataFrame(rows), {"glove": qc_records,
                                "bands": bands}


def aggregate_to_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    """Median across recordings per subject (counts carried; a subject
    with no primary target is excluded EXPLICITLY and reported)."""
    value_columns = [c for c in frame.columns
                     if c not in ("recording", "subject")]
    medians = frame.groupby("subject", sort=True)[value_columns].median()
    counts = frame.groupby("subject", sort=True)["recording"].size()
    out = medians.reset_index()
    out["n_recordings"] = out["subject"].map(counts)
    excluded = out[out["tap_rate_mean"].isna()]["subject"].tolist()
    if excluded:
        print(f"EXCLUDED (no primary target after QC): {len(excluded)} subject(s)")
        out = out[out["tap_rate_mean"].notna()]
    if len(out) < 2:
        raise CurveIconError(
            "fewer than 2 subjects survive primary-target QC; refusing to "
            "evaluate an empty cohort"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    print(DATA_USE_REMINDER, file=sys.stderr)
    if args.acknowledge_data_use:
        print(f"Data-use note (recorded in provenance): {args.acknowledge_data_use}")
    try:
        input_dir = args.input_dir
        base = args.output_dir or repo_root / "outputs" / "followon_subject_modeling"
        output_dir = resolve_output_dir(base, repo_root)
        output_dir = _versioned_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory (versioned, never overwrites): {output_dir}")

        # Input identity is recorded BEFORE any heavy compute — the
        # identical-inputs guarantee is verifiable from the first minute
        # of a run, not only in hindsight.
        manifest = _input_manifest(input_dir)
        (output_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        frame, qc = load_and_extract(input_dir)
        frame.to_csv(output_dir / "per_recording_features.csv", index=False)
        subjects = aggregate_to_subjects(frame)

        model_names = (
            "dummy_mean", "dummy_median", "ridge_primary", "svr_linear",
            "bayesian_ridge", "gp_matern32",
        ) + (("xgboost_shallow",) if args.with_xgboost else ())
        arm_frame = subjects

        results = run_arm(
            arm_frame, "tap_rate_mean", "subject",
            blocks=("clinical", "rest_spectral", "rest_dynamics",
                    "delta_spectral", "delta_dynamics", "combined",
                    "delta_combined"),
            model_names=model_names, seed=args.seed,
            include_intervals=True, collect_coefs=True,
        )
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "stage": args.stage,
            "seed": args.seed,
            "n_subjects": int(len(arm_frame)),
            "n_recordings": int(len(frame)),
            "input_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "data_use_note": args.acknowledge_data_use,
            "permutation_floor_used": args.permutations,
            "primary_target": PRIMARY_SUBJECT_TARGET,
            "secondary_targets": {
                "extracted": list(SECONDARY_SUBJECT_TARGETS),
                "modeled": [],
                "status": (
                    "deferred — extracted but never modeled under v1.0.0 "
                    "or this corrective run; recorded omission, no "
                    "secondary inference performed (any later run is a "
                    "separate, labeled analysis)"
                ),
            },
        }
        from curve_icon_dbso import tracking

        summary["git_sha"] = tracking.git_sha(repo_root)
        summary["lock_hash"] = tracking.lock_hash(repo_root)

        # Aggregates, promotion, uncertainty.
        aggregate_rows = []
        for block, block_results in results["blocks"].items():
            for result in block_results:
                aggregate_rows.append(
                    {"block": block, **result.aggregate, "model": result.model_name}
                )
        pd.DataFrame(aggregate_rows).to_csv(
            output_dir / "aggregate_metrics.csv", index=False
        )
        for block, block_results in results["blocks"].items():
            table = promotion_table(block_results,
                                    results["reference_aggregates"])
            table.to_csv(output_dir / f"promotion_{block}.csv", index=False)
            ridge = next(
                (r for r in block_results if r.model_name == "ridge_primary"),
                None,
            )
            if ridge is not None:
                leave_one_subject_influence(ridge.subject_table).to_csv(
                    output_dir / f"influence_{block}.csv", index=False
                )
                X = select_block(arm_frame, block)
                stability = coefficient_sign_stability(
                    ridge.fold_coefs, list(X.columns)
                )
                pd.DataFrame(
                    [{"feature": k, "sign_stability": v}
                     for k, v in stability.items()]
                ).to_csv(output_dir / f"coefficient_stability_{block}.csv",
                         index=False)
        # Bootstrap CIs for ridge_primary on the combined block.
        combined_results = results["blocks"]["combined"]
        ridge = next(r for r in combined_results
                     if r.model_name == "ridge_primary")
        table = ridge.subject_table
        cis = subject_bootstrap_ci(
            table["y_true"].to_numpy(), table["y_pred"].to_numpy(),
            table["group"].to_numpy(), n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        (output_dir / "bootstrap_combined_ridge.json").write_text(json.dumps(
            {k: cis[k].to_dict() for k in cis}, indent=2, default=str
        ))
        # Corrected permutation null through the complete nested pipeline
        # (combined block, the FULL frozen ten-test Holm family — v1.0.1
        # corrective: the optional XGBoost members are executed when the
        # dependency is present, and entered conservatively at p = 1 as
        # not_executed otherwise; the family size never shrinks).
        family_models = ("ridge_primary", "svr_linear", "bayesian_ridge",
                         "gp_matern32", "xgboost_shallow")
        executed_models: list[str] = []
        for model_name in family_models:
            if model_name == "xgboost_shallow":
                try:
                    build_estimator(model_name, {})
                    executed_models.append(model_name)
                except ImportError:
                    pass  # optional DD-003 dependency absent
            else:
                executed_models.append(model_name)
        null_results = permutation_null_nested(
            select_block(arm_frame, "combined"),
            arm_frame["tap_rate_mean"].to_numpy(dtype=float),
            arm_frame["subject"].to_numpy(),
            model_names=tuple(executed_models),
            n_permutations=args.permutations, seed=args.seed,
        )
        (output_dir / "permutation_null_combined.json").write_text(json.dumps(
            {model: {metric: entry for metric, entry in per_model.items()}
             for model, per_model in null_results.items()},
            indent=2, default=str,
        ))
        # The complete ten-test family with raw AND Holm-adjusted values,
        # machine-readable (the v1.0.0 run reported adjusted values over
        # only its eight executed members — corrected here).
        family_entries: dict[tuple[str, str], dict] = {}
        for model_name in family_models:
            for metric in ("MAE", "RMSE"):
                key = (model_name, metric)
                if model_name in null_results:
                    entry = null_results[model_name][metric]
                    family_entries[key] = {
                        "status": "executed",
                        "observed": entry["observed"],
                        "p_value_raw": entry["p_value"],
                        "null_mean": entry["null_mean"],
                        "n_permutations": entry["n_permutations"],
                        "seed": entry["seed"],
                    }
                else:
                    family_entries[key] = {
                        "status": "not_executed",
                        "observed": None,
                        "p_value_raw": 1.0,  # conservative entry
                        "null_mean": None,
                        "n_permutations": args.permutations,
                        "seed": args.seed,
                        "reason": "optional dependency absent (DD-003)",
                    }
        raw_pvalues = {
            f"{model}::{metric}": entry["p_value_raw"]
            for (model, metric), entry in family_entries.items()
        }
        adjusted_pvalues = holm_adjusted_pvalues(raw_pvalues)
        for key, entry in family_entries.items():
            entry["p_value_holm"] = adjusted_pvalues[f"{key[0]}::{key[1]}"]
        holm_document = {
            "family_rule": HOLM_FAMILY_RULE,
            "n_tests": len(family_entries),
            "block": "combined",
            "target": PRIMARY_SUBJECT_TARGET,
            "entries": {
                f"{model}::{metric}": entry
                for (model, metric), entry in family_entries.items()
            },
        }
        (output_dir / "holm_family_combined.json").write_text(
            json.dumps(holm_document, indent=2, default=str)
        )
        # Interpretation matrix (frozen §12) — evaluated in code, never
        # asserted in prose. The subject-level script computes the REST
        # and DELTA halves; the global-null verdict also needs the MOVE
        # arm's transfer Boolean, stored by the MOVE run.
        resting = resting_trait_evidence(
            results["blocks"]["combined"], results["reference_aggregates"],
            family_entries,
        )
        state_change = state_change_evidence(
            results["blocks"]["delta_combined"],
            results["blocks"]["combined"],
            results["reference_aggregates"],
        )
        summary["interpretation_matrix"] = {
            "resting_trait_evidence": resting,
            "state_change_evidence": state_change,
            "movement_state_transfer": {
                "evaluated_in": (
                    "scripts/run_followon_move_modeling.py — its run "
                    "summary carries the transfer components and Boolean"
                ),
                "met": None,
            },
            "global_null": {
                "met": None,
                "status": (
                    "pending the MOVE arm's transfer Boolean; the REST "
                    "and DELTA halves above are evaluated in code"
                ),
            },
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(f"Run complete; outputs in {output_dir}")
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

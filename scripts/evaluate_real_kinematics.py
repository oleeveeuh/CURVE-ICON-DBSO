#!/usr/bin/env python
"""Kinematic evaluation on external (non-synthetic) data.

**Non-blocking data guidance (docs/data_access.md):** before using any
non-synthetic data, confirm it is lawful to use — a genuine public
source, license/terms permitting the use, and no restricted, private,
identifiable, or former-lab material (owner responsibilities in
``docs/approvals.md``). This script prints that reminder when
``--acknowledge-data-use`` is not given; the flag is optional and is
recorded in the run summary as provenance when supplied. Nothing is
ever refused on its basis, and repository work is never gated on any
register or approval status.

Outputs default outside the repository; an in-repository output path
produces a warning, not an error (docs/data_access.md reminders).

**What is implemented vs intentionally not:** the evaluation half
(grouped CV with baselines, bootstrap CIs, permutation null, provenance,
MLflow lineage) is complete and rehearsed on synthetic fixtures in the
test suite. The raw-file extraction half (parsing a source's raw tables
into a per-recording feature table) is not implemented yet: it is best
developed against a concrete dataset. This script therefore consumes a
per-recording feature table (one row per recording) produced by that
future extraction step — which is why its input contract is a plain
CSV.

Example:

    python scripts/evaluate_real_kinematics.py \
      --features-csv /local/outside/repo/kinematic_recordings.csv \
      --output-dir /local/outside/repo/kinematic_results \
      --acknowledge-data-use "Public dataset X, license CC-BY-4.0, recorded in docs/data_access.md." \
      --stage exploratory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from curve_icon_dbso.bootstrap import run_permutation_null, subject_bootstrap_ci
from curve_icon_dbso.errors import CurveIconError
from curve_icon_dbso.evaluation import (
    baseline_models,
    default_models,
    run_group_cv,
    xgboost_model,
)
from curve_icon_dbso.kinematics import (
    KINEMATIC_TARGET,
    build_kinematic_model_matrix,
    resolve_output_dir,
)
from curve_icon_dbso.reporting import save_outputs

PROVENANCE_REAL_KINEMATIC = (
    "public third-party data — locally supplied copy; "
    "exploratory kinematic target; NOT DBS outcome"
)

DATA_USE_REMINDER = (
    "REMINDER (non-blocking): before using non-synthetic data, confirm it is "
    "lawful to use for this purpose — genuinely public source, license/terms "
    "permitting the use, and no restricted, private, identifiable, or "
    "former-lab material. See docs/data_access.md and docs/approvals.md. "
    "This reminder never prevents a run."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-real-kinematics",
        description=(
            "Subject-grouped evaluation of the prespecified kinematic analysis "
            "on external (non-synthetic) data. Prints a non-blocking data-use "
            "reminder; never gated on any register or approval status."
        ),
    )
    parser.add_argument("--features-csv", type=Path, required=True,
                        help="Per-recording kinematic feature table (one row per "
                             "recording; KINEMATIC_FEATURES + tap_rate_mean + subject)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory — recommended outside this repository "
                             "(an in-repo path prints a warning and still runs)")
    parser.add_argument(
        "--acknowledge-data-use", default=None,
        help="Optional provenance note recorded in the run summary (e.g. the "
             "dataset name and its recorded license/terms). Purely "
             "informational — never required.",
    )
    parser.add_argument("--stage", choices=("exploratory", "confirmatory"),
                        required=True, help="Analysis stage label (SCI-011)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=200,
                        help="SCI-009 protocol default: 200")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000,
                        help="SCI-008 protocol default: 2000")
    parser.add_argument("--with-xgboost", action="store_true",
                        help="Optional DD-003 parity exercise (never a default)")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log the run to a local MLflow store (MOD-004)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        # Non-blocking data-use reminder (docs/data_access.md); nothing here
        # ever refuses to run.
        if args.acknowledge_data_use:
            print(f"Data-use note (recorded in provenance): {args.acknowledge_data_use}")
        else:
            print(DATA_USE_REMINDER, file=sys.stderr)
        output_dir = resolve_output_dir(args.output_dir, repo_root)

        results = _evaluate(
            pd.read_csv(args.features_csv),
            repo_root=repo_root,
            seed=args.seed,
            permutations=args.permutations,
            bootstrap_resamples=args.bootstrap_resamples,
            with_xgboost=args.with_xgboost,
            stage=args.stage,
            output_dir=output_dir,
            use_mlflow=args.mlflow,
            acknowledgement=args.acknowledge_data_use,
        )
        print(
            "Evaluation complete: "
            + ", ".join(f"{r.model_name} MAE={r.metrics['MAE']:.3f}" for r in results)
        )
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _evaluate(frame, *, repo_root, seed, permutations, bootstrap_resamples,
              with_xgboost, stage, output_dir, use_mlflow,
              acknowledgement: str | None = None):
    """The evaluation half — identical machinery to the synthetic demo."""
    from curve_icon_dbso import tracking
    from curve_icon_dbso.evaluation import describe_fold_strategy

    x, y, groups = build_kinematic_model_matrix(frame)
    models = {**baseline_models(), **default_models()}
    if with_xgboost:
        models["XGBoost"] = xgboost_model()
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)

    bootstrap_summary = {
        result.model_name: subject_bootstrap_ci(
            y.to_numpy(), result.oof_predictions, groups.to_numpy(),
            n_resamples=bootstrap_resamples, seed=seed,
        )
        for result in results
    }
    permutation_summary = None
    if permutations > 0:
        permutation_summary = run_permutation_null(
            x, y.to_numpy(), groups.to_numpy(), models,
            n_permutations=permutations, seed=seed,
        )

    run_provenance = {
        "git_sha": tracking.git_sha(repo_root),
        "lock_hash": tracking.lock_hash(repo_root),
        "feature_schema_hash": tracking.kinematic_schema_hash(),
        "seed": seed,
    }
    if acknowledgement:
        run_provenance["data_use_note"] = acknowledgement
    paths = save_outputs(
        output_dir=output_dir,
        results=results,
        n_subjects=int(groups.nunique()),
        n_recordings=int(len(frame)),
        provenance=PROVENANCE_REAL_KINEMATIC,
        target=KINEMATIC_TARGET,
        analysis_stage=stage,
        fold_strategy=describe_fold_strategy(int(groups.nunique())),
        missing_data_handling=(
            "fold-local median imputation + standard scaling inside the "
            "sklearn Pipeline; QC-excluded recordings (tap QC) never enter "
            "the model matrix"
        ),
        run_provenance=run_provenance,
        bootstrap_summary=bootstrap_summary,
        permutation_summary=permutation_summary,
    )
    if use_mlflow:
        tracking.log_evaluation_run(
            run_name=f"kinematic-real-seed{seed}",
            config={
                "n_subjects": int(groups.nunique()),
                "n_recordings": int(len(frame)),
                "splitter": describe_fold_strategy(int(groups.nunique())),
                "preprocessing": "fold-local median imputation + standard scaling",
                "baselines": "DummyRegressor mean and median",
                "with_xgboost": with_xgboost,
            },
            model_params={},
            aggregate_metrics={
                r.model_name: {k: float(v) for k, v in r.metrics.items()}
                for r in results
            },
            fold_metrics={
                r.model_name: tracking.fold_metrics_from_result(r) for r in results
            },
            bootstrap_summary=bootstrap_summary,
            permutation_summary=permutation_summary,
            provenance=PROVENANCE_REAL_KINEMATIC,
            seed=seed,
            stage=stage,
            git_sha_value=run_provenance["git_sha"],
            lock_hash_value=run_provenance["lock_hash"],
            schema_hash_value=run_provenance["feature_schema_hash"],
            target=KINEMATIC_TARGET,
            artifacts={
                "metrics": paths["metrics"],
                "oof": paths["oof"],
                "summary": paths["summary"],
            },
        )
    return results


if __name__ == "__main__":
    raise SystemExit(main())

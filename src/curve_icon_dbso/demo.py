"""Deterministic synthetic end-to-end demo.

Runs the full corrected pipeline — synthetic cohort generation,
clinical composites, prespecified feature matrix, leave-one-subject-out
evaluation with baselines, subject-cluster bootstrap confidence
intervals, a label-permutation null, and provenance-labeled outputs —
entirely offline. Used by the quick start in the README and by the test
suite as the end-to-end determinism check.

The output is **pipeline verification on synthetic data**. It
demonstrates that the code runs and produces correctly aligned,
leakage-controlled metrics; it says nothing about real DBS outcomes.

Optional flags add the Phase 1 machinery without changing the core run:
``--with-xgboost`` (the DD-003 parity exercise, never a default),
``--mlflow`` (local lineage logging, MOD-004; requires ``.[tracking]``),
and ``--permutations``/``--bootstrap-resamples`` to size the uncertainty
procedures (defaults follow SCI-008/SCI-009).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import features as features_mod
from . import reporting, tracking
from .bootstrap import (
    BOOTSTRAP_N_RESAMPLES,
    PERMUTATION_N_PERMUTATIONS,
    run_permutation_null,
    subject_bootstrap_ci,
)
from .config import DemoConfig
from .errors import CurveIconError
from .evaluation import (
    baseline_models,
    default_models,
    describe_fold_strategy,
    run_group_cv,
    xgboost_model,
)
from .synthetic import generate_synthetic_cohort


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="curve-icon-demo",
        description=(
            "Run the corrected subject-grouped evaluation pipeline on a "
            "deterministic SYNTHETIC cohort (offline; no real data involved)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for outputs (default: outputs/synthetic_demo)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument(
        "--n-subjects", type=int, default=None,
        help="Number of synthetic subjects (default: 8)",
    )
    parser.add_argument(
        "--stage", choices=tracking.ANALYSIS_STAGES, default="exploratory",
        help="Analysis stage label (SCI-011); every run is labeled explicitly",
    )
    parser.add_argument(
        "--with-xgboost", action="store_true",
        help="Add the optional fixed-parameter XGBoost comparison (DD-003; "
             "requires the 'analysis' dependency group; never a default)",
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_N_RESAMPLES,
        help="Subject-cluster bootstrap resamples (SCI-008 default: 2000)",
    )
    parser.add_argument(
        "--permutations", type=int, default=25,
        help="Subject-level label permutations through the full CV workflow. "
             "This demo default (25) is a fast pipeline-verification count; "
             "real evaluation runs must use the SCI-009 protocol default of "
             f"at least {PERMUTATION_N_PERMUTATIONS} (pass --permutations "
             f"{PERMUTATION_N_PERMUTATIONS}).",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log the run to a local, gitignored MLflow store (MOD-004; "
             "requires the 'tracking' dependency group)",
    )
    args = parser.parse_args(argv)

    try:
        defaults = DemoConfig()
        config = DemoConfig(
            seed=args.seed if args.seed is not None else defaults.seed,
            n_subjects=args.n_subjects if args.n_subjects is not None else defaults.n_subjects,
            output_dir=args.output_dir if args.output_dir is not None else defaults.output_dir,
        )
        if args.bootstrap_resamples < 1:
            raise CurveIconError("--bootstrap-resamples must be >= 1")
        if args.permutations < 0:
            raise CurveIconError("--permutations must be >= 0")
        _run(
            config,
            stage=args.stage,
            with_xgboost=args.with_xgboost,
            bootstrap_resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            use_mlflow=args.mlflow,
        )
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _model_params(config: DemoConfig, with_xgboost: bool) -> dict[str, dict]:
    """Fixed hyperparameters per model, exactly as used in the evaluation."""
    cfg = config.models
    params: dict[str, dict] = {
        "Mean baseline": {"strategy": "mean"},
        "Median baseline": {"strategy": "median"},
        "Random Forest": {
            "n_estimators": cfg.rf_n_estimators,
            "max_depth": cfg.rf_max_depth,
            "random_state": cfg.seed,
            "n_jobs": 1,
        },
        "Gradient Boosting": {
            "n_estimators": cfg.gbr_n_estimators,
            "max_depth": cfg.gbr_max_depth,
            "learning_rate": cfg.gbr_learning_rate,
            "random_state": cfg.seed,
        },
        "Linear Regression": {},
    }
    if with_xgboost:
        from .evaluation import XGB_FIXED_PARAMS

        xgb = dict(XGB_FIXED_PARAMS)
        xgb["random_state"] = cfg.seed
        params["XGBoost"] = xgb
    return params


def _run(
    config: DemoConfig,
    *,
    stage: str,
    with_xgboost: bool,
    bootstrap_resamples: int,
    permutations: int,
    use_mlflow: bool,
) -> None:
    cohort = generate_synthetic_cohort(n_subjects=config.n_subjects, seed=config.seed)
    enriched = features_mod.add_clinical_composites(cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)

    models = {**baseline_models(), **default_models(config.models)}
    if with_xgboost:
        models["XGBoost"] = xgboost_model(config.models)
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)

    # SCI-008: subject-cluster bootstrap CIs per model, from the model's own
    # aligned out-of-fold predictions.
    bootstrap_summary = {
        result.model_name: subject_bootstrap_ci(
            y.to_numpy(), result.oof_predictions, groups.to_numpy(),
            n_resamples=bootstrap_resamples, seed=config.seed,
        )
        for result in results
    }

    # SCI-009: subject-level label permutations through the complete CV
    # workflow (baselines included — their nulls calibrate the comparison).
    permutation_summary = None
    if permutations > 0:
        permutation_summary = run_permutation_null(
            x, y.to_numpy(), groups.to_numpy(), models,
            n_permutations=permutations, seed=config.seed,
        )

    run_provenance = {
        "git_sha": tracking.git_sha(),
        "lock_hash": tracking.lock_hash(),
        "feature_schema_hash": tracking.outcome_schema_hash(),
        "seed": config.seed,
    }
    paths = reporting.save_outputs(
        output_dir=config.output_dir,
        results=results,
        n_subjects=int(groups.nunique()),
        n_recordings=int(len(cohort)),
        provenance=reporting.PROVENANCE_SYNTHETIC,
        target=features_mod.TARGET_COLUMN,
        analysis_stage=stage,
        fold_strategy=describe_fold_strategy(int(groups.nunique())),
        missing_data_handling=(
            "fold-local median imputation + standard scaling inside the "
            "sklearn Pipeline (training folds only); QC failures excluded "
            "with explicit records, never zero-filled"
        ),
        run_provenance=run_provenance,
        bootstrap_summary=bootstrap_summary,
        permutation_summary=permutation_summary,
    )

    if use_mlflow:
        tracking.log_evaluation_run(
            run_name=f"synthetic-demo-seed{config.seed}",
            config={
                "n_subjects": config.n_subjects,
                "n_recordings": int(len(cohort)),
                "splitter": describe_fold_strategy(int(groups.nunique())),
                "preprocessing": "fold-local median imputation + standard scaling",
                "baselines": "DummyRegressor mean and median",
                "with_xgboost": with_xgboost,
                "bootstrap_resamples": bootstrap_resamples,
                "permutations": permutations,
            },
            model_params=_model_params(config, with_xgboost),
            aggregate_metrics={
                result.model_name: {
                    k: float(v) for k, v in result.metrics.items()
                }
                for result in results
            },
            fold_metrics={
                result.model_name: tracking.fold_metrics_from_result(result)
                for result in results
            },
            bootstrap_summary=bootstrap_summary,
            permutation_summary=permutation_summary,
            provenance=reporting.PROVENANCE_SYNTHETIC,
            seed=config.seed,
            stage=stage,
            git_sha_value=run_provenance["git_sha"],
            lock_hash_value=run_provenance["lock_hash"],
            schema_hash_value=run_provenance["feature_schema_hash"],
            target=features_mod.TARGET_COLUMN,
            artifacts={
                "metrics": paths["metrics"],
                "oof": paths["oof"],
                "summary": paths["summary"],
            },
        )

    print(reporting.format_summary(
        results,
        n_subjects=int(groups.nunique()),
        n_recordings=int(len(cohort)),
        provenance=reporting.PROVENANCE_SYNTHETIC,
        cohort_note="(synthetic cohort — pipeline verification only)",
        fold_strategy=describe_fold_strategy(int(groups.nunique())),
    ))
    print("\nOutputs written:")
    for label, path in paths.items():
        print(f"  {label:>8}: {path}")


if __name__ == "__main__":
    raise SystemExit(main())

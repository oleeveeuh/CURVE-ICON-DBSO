#!/usr/bin/env python
"""Compliance-gated kinematic evaluation (external data — currently closed).

**Gate:** running this script against non-synthetic data requires a
dataset recorded ``VERIFIED`` in the Dataset compliance register of
``docs/data_access.md`` (public-source, license/terms, provenance
boundary, and no-endorsement checks — semantics in
``docs/approvals.md``). With no verified dataset the external-data path
is CLOSED and this script must not see any data beyond synthetic test
fixtures.

**Acknowledgement rule (docs/data_access.md):** the script refuses to run
without ``--acknowledge-data-use`` carrying the exact compliance
statement, and refuses in-repository output paths — real per-subject
outputs stay outside the checkout.

**What is implemented vs intentionally not:** the evaluation half
(grouped CV with baselines, bootstrap CIs, permutation null, provenance,
MLflow lineage) is complete and rehearsed on synthetic fixtures in the
test suite. The raw-file extraction half (parsing a source's raw tables
into a per-recording feature table) is deliberately NOT implemented: it
can only be developed and verified against a concrete,
compliance-verified dataset, and a speculative unverified loader would
repeat the historical placeholder mistake. It will be added as the
first compliance-gated step. This script therefore consumes a
per-recording feature table (one row per recording) produced by that
future extraction step — which is why its input contract is a plain
CSV.

Example (compliance-verified public dataset required):

    python scripts/evaluate_real_kinematics.py \
      --features-csv /local/outside/repo/kinematic_recordings.csv \
      --output-dir /local/outside/repo/kinematic_results \
      --acknowledge-data-use "This dataset is compliance-verified public data, lawfully licensed for this use, and not restricted, private, identifiable, or former-lab data." \
      --stage exploratory
"""

from __future__ import annotations

import argparse
import re
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
    "public third-party data — compliance-verified local copy; "
    "exploratory kinematic target; NOT DBS outcome"
)

REQUIRED_ACKNOWLEDGEMENT = (
    "This dataset is compliance-verified public data, lawfully licensed "
    "for this use, and not restricted, private, identifiable, or "
    "former-lab data."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-real-kinematics",
        description=(
            "Subject-grouped evaluation of the prespecified kinematic analysis "
            "on external (non-synthetic) data. Compliance-gated: requires a "
            "VERIFIED entry in the docs/data_access.md dataset compliance "
            "register before use."
        ),
    )
    parser.add_argument("--features-csv", type=Path, required=True,
                        help="Per-recording kinematic feature table (one row per "
                             "recording; KINEMATIC_FEATURES + tap_rate_mean + subject)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory — must be OUTSIDE this repository")
    parser.add_argument(
        "--acknowledge-data-use", required=True,
        help=f'Must be exactly: "{REQUIRED_ACKNOWLEDGEMENT}"',
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
        # ---- Gate 1: explicit acknowledgement (docs/data_access.md rule) ----
        if args.acknowledge_data_use != REQUIRED_ACKNOWLEDGEMENT:
            raise CurveIconError(
                "External-data evaluation requires --acknowledge-data-use to be "
                f'exactly: "{REQUIRED_ACKNOWLEDGEMENT}". This flag exists so no '
                "run can touch non-synthetic data by accident or in a script."
            )
        # ---- Gate 2: outputs outside the repository ----
        output_dir = resolve_output_dir(args.output_dir, repo_root)
        # ---- Gate 3: dataset compliance register (checked at execution time) ----
        _check_compliance_gate(repo_root)

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
        )
        print(
            "Evaluation complete: "
            + ", ".join(f"{r.model_name} MAE={r.metrics['MAE']:.3f}" for r in results)
        )
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _check_compliance_gate(repo_root: Path) -> None:
    """Read the Dataset compliance register in docs/data_access.md and
    refuse to run while no third-party dataset is recorded VERIFIED."""
    doc = repo_root / "docs" / "data_access.md"
    if not doc.exists():
        raise CurveIconError(
            "docs/data_access.md is missing — the compliance state is unknown, "
            "so the external-data path stays CLOSED."
        )
    text = doc.read_text()
    if "Dataset compliance register" not in text:
        raise CurveIconError(
            "The 'Dataset compliance register' section is missing from "
            "docs/data_access.md — the compliance state is unknown, so the "
            "external-data path stays CLOSED."
        )
    statuses = re.findall(r"^- Verification status: (\S+)", text, flags=re.MULTILINE)
    if not any(status == "VERIFIED" for status in statuses):
        raise CurveIconError(
            "The external-data path is CLOSED: no dataset in the "
            "docs/data_access.md compliance register is VERIFIED (statuses "
            f"found: {', '.join(statuses) if statuses else 'none'}). Complete "
            "the register for a genuinely public dataset first — see "
            "docs/approvals.md."
        )


def _evaluate(frame, *, repo_root, seed, permutations, bootstrap_resamples,
              with_xgboost, stage, output_dir, use_mlflow):
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

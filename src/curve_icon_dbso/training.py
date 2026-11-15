"""Deterministic synthetic model training and artifact promotion (Phase 2).

Trains the public synthetic-track serving artifact:

1. Generate the deterministic synthetic cohort (fixed seed).
2. Run the corrected subject-grouped (LOSO) evaluation over the fixed
   candidates AND the naive baselines — the manifest metrics are honest
   out-of-fold numbers from that evaluation, never full-fit numbers.
3. Apply the DD-006 promotion rule: the candidate with the lowest MAE
   (the prespecified primary metric; ties broken by the fixed candidate
   order) is selected. ``beats_baseline`` — beat BOTH naive baselines on
   BOTH MAE and RMSE — decides whether it is labeled a performance model
   or a clearly-labeled demonstration model (SCI-014).
4. Refit the selected estimator with the SAME fold-local preprocessing
   (median impute + standardize) on the full synthetic cohort and
   package it with :func:`curve_icon_dbso.artifact.build_artifact`.

Determinism: with the default offline path, two runs with the same seed
produce **byte-identical** artifacts on the same environment — the
manifest carries no wall-clock timestamp and its ``run_id`` is a
deterministic fingerprint of (source SHA, lock hash, schema hash,
model hash, seed, cohort size). ``--mlflow`` logs the run to the local
synthetic store and stamps the real (nondeterministic) MLflow run ID;
that path is explicitly outside the byte-determinism promise.

This module trains on synthetic data ONLY by construction (the cohort
comes from :mod:`curve_icon_dbso.synthetic`). Real-data training stays
inside the institutional track and is gated by docs/approvals.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from . import artifact, reporting, schema, tracking
from .config import ModelConfig
from .evaluation import (
    baseline_models,
    default_models,
    describe_fold_strategy,
    run_group_cv,
)
from .features import add_clinical_composites, build_model_matrix
from .reporting import PROVENANCE_SYNTHETIC
from .synthetic import generate_synthetic_cohort

MODEL_VERSION = "synthetic-demo-1.0.0"

INTENDED_USE = "research only; not for clinical use"

PROHIBITED_USES: list[str] = [
    "clinical decision support, diagnosis, or treatment selection",
    "patient care, counseling, cohort selection, or DBS programming",
    "any use with real patient data (real data requires the institutional "
    "track and recorded approvals; see docs/approvals.md)",
    "biomarker discovery or clinical interpretation of predictions",
    "performance or generalization claims beyond this synthetic "
    "demonstration",
]

#: Prespecified primary metric for the DD-006 selection rule.
PRIMARY_METRIC = "MAE"

SELECTION_RULE = (
    f"DD-006: lowest {PRIMARY_METRIC} among the fixed candidates (ties broken by "
    "the fixed candidate order); beats_baseline (both naive baselines on "
    "both MAE and RMSE) decides the performance-vs-demonstration label"
)


def train_synthetic_artifact(
    output_dir: str | Path,
    *,
    seed: int = 42,
    n_subjects: int = 8,
    stage: str = "exploratory",
    use_mlflow: bool = False,
    model_version: str = MODEL_VERSION,
) -> dict:
    """Train, evaluate, select and package the synthetic serving artifact.

    Returns the completed manifest dict (identical to the written
    ``manifest.json`` content).

    Raises:
        curve_icon_dbso.errors.ArtifactError: If ``output_dir`` already
            exists (artifacts are immutable; choose a fresh directory).
        ImportError: If ``use_mlflow`` is set without the optional
            ``tracking`` dependency group.
    """
    if stage not in tracking.ANALYSIS_STAGES:
        raise ValueError(f"stage must be one of {tracking.ANALYSIS_STAGES}, got {stage!r}")

    cohort = generate_synthetic_cohort(n_subjects=n_subjects, seed=seed)
    enriched = add_clinical_composites(cohort)
    x, y, groups = build_model_matrix(enriched)

    model_cfg = ModelConfig(seed=seed)
    models = {**baseline_models(), **default_models(model_cfg)}
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)

    # DD-006 promotion rule, computed from the corrected out-of-fold metrics.
    verdicts = reporting.beats_baseline_strict(results)
    candidates = [r for r in results if "baseline" not in r.model_name.lower()]
    if not candidates:  # defensive: default_models always provides some
        raise ValueError("no candidate models to select from")
    selected = min(candidates, key=lambda r: (r.metrics[PRIMARY_METRIC], candidates.index(r)))
    beats_baseline = bool(verdicts[selected.model_name])
    promotion_label = "performance" if beats_baseline else "demonstration_only"

    # The served object: selected estimator + the SAME fold-local
    # preprocessing steps used inside the evaluation pipeline.
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    estimator = clone(dict(default_models(model_cfg))[selected.model_name])
    pipeline = Pipeline([*evaluation_preprocessing(), ("model", estimator)])
    pipeline.fit(x, y.to_numpy())

    source_sha = tracking.git_sha()
    lock = tracking.lock_hash()
    model_version_final = model_version

    metrics = {
        r.model_name: {k: _number(v) for k, v in r.metrics.items()} for r in results
    }
    baselines = {
        name: metrics[name]
        for name in ("Mean baseline", "Median baseline")
        if name in metrics
    }
    candidate_metrics = {
        r.model_name: metrics[r.model_name]
        for r in candidates
    }

    run_id = deterministic_run_id(
        source_sha=source_sha,
        lock_hash=lock,
        seed=seed,
        n_subjects=n_subjects,
        model_version=model_version_final,
    )
    manifest: dict = {
        "manifest_version": artifact.MANIFEST_VERSION,
        "model_version": model_version_final,
        "model_class": f"{type(pipeline).__module__}.{type(pipeline).__name__}",
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "library_versions": artifact.library_versions(),
        "training_seed": seed,
        "training_rows": int(len(cohort)),
        "training_subjects": int(groups.nunique()),
        "data_provenance": PROVENANCE_SYNTHETIC,
        "intended_use": INTENDED_USE,
        "prohibited_uses": list(PROHIBITED_USES),
        "selection_rule": SELECTION_RULE,
        "primary_metric": PRIMARY_METRIC,
        "promotion_label": promotion_label,
        "fold_strategy": describe_fold_strategy(int(groups.nunique())),
        "metrics": {selected.model_name: metrics[selected.model_name]},
        "baselines": baselines,
        "candidates": candidate_metrics,
        "beats_baseline": beats_baseline,
        "run_id": run_id,
        "source_sha": source_sha,
        "lock_hash": lock,
        "analysis_stage": stage,
        "research_only": True,
        "not_for_clinical_use": True,
    }

    if use_mlflow:
        mlflow_run_id = _log_training_run(
            manifest=manifest,
            results=results,
            model_params={
                name: params for name, params in _model_params(model_cfg).items()
                if name == selected.model_name
            },
        )
        manifest["mlflow_run_id"] = mlflow_run_id
        manifest["run_id"] = mlflow_run_id

    completed = artifact.build_artifact(output_dir, model=pipeline, manifest=manifest)
    return completed


def evaluation_preprocessing() -> list[tuple[str, object]]:
    """The fold-local preprocessing steps, shared with the served pipeline."""
    from .evaluation import make_preprocessing

    return make_preprocessing()


def deterministic_run_id(
    *, source_sha: str, lock_hash: str, seed: int, n_subjects: int, model_version: str
) -> str:
    """Stable run fingerprint for the offline (no-MLflow) path.

    Deliberately labeled ``synthetic-train-`` so it can never be
    mistaken for an MLflow run ID; ``--mlflow`` replaces it with a real
    run ID (documented as outside the determinism promise).
    """
    payload = "|".join([
        "synthetic-train",
        source_sha,
        lock_hash,
        schema.schema_hash(),
        model_version,
        str(seed),
        str(n_subjects),
    ])
    return f"synthetic-train-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _model_params(model_cfg: ModelConfig) -> dict[str, dict]:
    """Fixed hyperparameters per candidate, mirroring demo._model_params."""
    return {
        "Mean baseline": {"strategy": "mean"},
        "Median baseline": {"strategy": "median"},
        "Random Forest": {
            "n_estimators": model_cfg.rf_n_estimators,
            "max_depth": model_cfg.rf_max_depth,
            "random_state": model_cfg.seed,
            "n_jobs": 1,
        },
        "Gradient Boosting": {
            "n_estimators": model_cfg.gbr_n_estimators,
            "max_depth": model_cfg.gbr_max_depth,
            "learning_rate": model_cfg.gbr_learning_rate,
            "random_state": model_cfg.seed,
        },
        "Linear Regression": {},
    }


def _log_training_run(*, manifest: dict, results, model_params: dict) -> str:
    """Log the training+evaluation run to the local synthetic MLflow store."""
    return tracking.log_evaluation_run(
        run_name=f"synthetic-train-{manifest['model_version']}-seed{manifest['training_seed']}",
        config={
            "model_version": manifest["model_version"],
            "selection_rule": manifest["selection_rule"],
            "promotion_label": manifest["promotion_label"],
            "training_rows": manifest["training_rows"],
            "training_subjects": manifest["training_subjects"],
            "baselines": "DummyRegressor mean and median",
        },
        model_params=model_params,
        aggregate_metrics={
            result.model_name: {k: float(v) for k, v in result.metrics.items()}
            for result in results
        },
        provenance=manifest["data_provenance"],
        seed=manifest["training_seed"],
        stage=manifest["analysis_stage"],
        git_sha_value=manifest["source_sha"],
        lock_hash_value=manifest["lock_hash"],
        schema_hash_value=schema.schema_hash(),
        target=schema.TARGET,
    )


def _number(value):
    """Plain Python number for JSON (numpy scalars -> float/int)."""
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``python -m curve_icon_dbso.training``)."""
    parser = argparse.ArgumentParser(
        prog="curve-icon-train",
        description=(
            "Deterministically train and package the SYNTHETIC serving "
            "artifact (model + schema + manifest). Research demonstration "
            "only — the served model is trained on synthetic data and makes "
            "no clinical claim."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/synthetic-demo"),
        help="Artifact directory to create (default: artifacts/synthetic-demo; "
             "refuses to overwrite an existing artifact)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed (default: 42)")
    parser.add_argument(
        "--n-subjects", type=int, default=8,
        help="Number of synthetic subjects (default: 8)",
    )
    parser.add_argument(
        "--stage", choices=tracking.ANALYSIS_STAGES, default="exploratory",
        help="Analysis stage label (SCI-011)",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Also log the training run to the local MLflow store (MOD-004; "
             "requires the 'tracking' dependency group). NOTE: the run ID "
             "stamped into the manifest is then nondeterministic by design.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = train_synthetic_artifact(
            args.output_dir,
            seed=args.seed,
            n_subjects=args.n_subjects,
            stage=args.stage,
            use_mlflow=args.mlflow,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary; print message, exit 1
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Synthetic serving artifact written (research demonstration only):")
    print(f"  model_version : {manifest['model_version']}")
    print(f"  run_id        : {manifest['run_id']}")
    print(f"  schema        : v{manifest['schema_version']} ({manifest['schema_hash'][:12]}...)")
    print(f"  artifact hash : {manifest['artifact_sha256'][:12]}...")
    print(f"  selected      : {next(iter(manifest['metrics']))}")
    print(f"  beats_baseline: {manifest['beats_baseline']} ({manifest['promotion_label']})")
    print(f"  data          : {manifest['data_provenance']}")
    for filename in (artifact.MODEL_FILE, artifact.SCHEMA_FILE, artifact.MANIFEST_FILE):
        print(f"  {Path(args.output_dir) / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

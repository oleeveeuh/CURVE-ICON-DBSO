"""Experiment lineage: local MLflow tracking and provenance fingerprints.

Satisfies MOD-004 (parameters, fold metrics, aggregate metrics, bootstrap
and permutation summaries, git SHA, analysis stage) and REP-005 (source,
lock, schema, seed, provenance recorded per run), with two hard rules:

* **MLflow is optional.** This module never imports ``mlflow`` at module
  level; the core package works without it. ``log_evaluation_run`` raises
  an informative error only when actually called without the dependency
  (``pip install -e ".[tracking]"``). There is NO model registry (MOD-005)
  and none may be added without a protocol change.
* **Provenance is real.** ``git_sha``, ``lock_hash`` and
  ``feature_schema_hash`` compute actual fingerprints from the
  repository, the committed lock file, and the canonical feature
  contract — no placeholder values anywhere.

Privacy (MOD-006): the default local store (``mlruns/``, gitignored) is for
**synthetic** runs only. Former-lab or restricted-data runs must never be
logged anywhere in this project; a store for recorded public data stays
local and gitignored — never published in this repository or any cloud
account.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .errors import DataValidationError

# Default local tracking store — gitignored; synthetic runs only (see module
# docstring for the real-data rule).
DEFAULT_TRACKING_DIR = "mlruns"

# SCI-011: every run is labeled exploratory or confirmatory; anything else
# must fail loudly rather than default silently.
ANALYSIS_STAGES = ("exploratory", "confirmatory")


def validate_stage(stage: str) -> str:
    """Validate the ``exploratory|confirmatory`` label (SCI-011)."""
    if stage not in ANALYSIS_STAGES:
        raise DataValidationError(
            f"analysis stage must be one of {ANALYSIS_STAGES}, got {stage!r}. "
            "Every run must be explicitly labeled (SCI-011)."
        )
    return stage


def git_sha(repo_root: str | Path | None = None) -> str:
    """Full SHA of the current commit, or ``"unknown"`` outside a git repo.

    Never guesses: an unreadable repository reports ``"unknown"`` so a
    broken fingerprint is visible instead of silently plausible.
    """
    root = Path(repo_root) if repo_root is not None else None
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not sha:
        return "unknown"
    return sha


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_hash(repo_root: str | Path | None = None) -> str:
    """SHA-256 of the committed ``uv.lock``, or ``"absent"`` if none exists."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    lock = root / "uv.lock"
    if not lock.exists():
        return "absent"
    return file_sha256(lock)


def feature_schema_hash(features: list[str], target: str, group: str) -> str:
    """Stable SHA-256 fingerprint of a feature-schema contract.

    The canonical form is a sorted-key JSON document of the feature list,
    target and group column — any change to the contract changes the hash.
    """
    canonical = json.dumps(
        {"features": list(features), "group": group, "target": target},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def outcome_schema_hash() -> str:
    """Schema fingerprint of the (paused) 7-feature outcome contract."""
    from . import features as features_mod

    return feature_schema_hash(
        features_mod.PRESPECIFIED_FEATURES,
        features_mod.TARGET_COLUMN,
        features_mod.GROUP_COLUMN,
    )


def kinematic_schema_hash() -> str:
    """Schema fingerprint of the (active) 5-feature kinematic contract."""
    from .kinematics import KINEMATIC_FEATURES, KINEMATIC_GROUP, KINEMATIC_TARGET

    return feature_schema_hash(KINEMATIC_FEATURES, KINEMATIC_TARGET, KINEMATIC_GROUP)


def fold_metrics_from_result(result) -> list[dict]:
    """Per-fold metrics from a :class:`~curve_icon_dbso.evaluation.CVResult`.

    Recomputed from the aligned per-subject out-of-fold frame (grouped by
    ``fold_id``), so the numbers trace to the same aligned predictions that
    produce the aggregate metrics.
    """
    folds: list[dict] = []
    for fold_id, group in result.per_subject.groupby("fold_id", sort=True):
        folds.append({
            "fold": int(fold_id),
            "MAE": float((group["y_pred"] - group["y_true"]).abs().mean()),
            "RMSE": float(np.sqrt(((group["y_pred"] - group["y_true"]) ** 2).mean())),
            "n_rows": int(group["n_rows"].sum()),
        })
    return folds


def log_evaluation_run(
    *,
    run_name: str,
    config: dict,
    model_params: dict[str, dict],
    aggregate_metrics: dict[str, dict[str, float]],
    fold_metrics: dict[str, list[dict]] | None = None,
    bootstrap_summary: dict | None = None,
    permutation_summary: dict | None = None,
    provenance: str,
    seed: int,
    stage: str,
    git_sha_value: str,
    lock_hash_value: str,
    schema_hash_value: str,
    target: str,
    artifacts: dict[str, str | Path] | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log one complete, reproducible evaluation run to a local MLflow store.

    Everything is explicit — the caller passes computed values, this
    function performs no computation of its own beyond validation — so a
    logged run is a faithful record of the analysis that produced it.

    Args:
        run_name: Human-readable run label.
        config: Flat configuration record (cohort filter, splitter,
            preprocessing, baselines flag, ...).
        model_params: ``{model_name: {param: value}}`` fixed hyperparameters.
        aggregate_metrics: ``{model_name: {metric: value}}``.
        fold_metrics: Optional ``{model_name: [{fold, MAE, RMSE, n_rows}]}``.
        bootstrap_summary: Optional ``{model_name: {metric: BootstrapCI dict}}``.
        permutation_summary: Optional ``{model_name: {metric: summary dict}}``.
        provenance: Data provenance label (synthetic or restricted-real).
        seed: Global analysis seed.
        stage: ``exploratory`` or ``confirmatory`` (validated, SCI-011).
        git_sha_value: Source fingerprint (from :func:`git_sha`).
        lock_hash_value: Lock fingerprint (from :func:`lock_hash`).
        schema_hash_value: Feature-schema fingerprint (from
            :func:`feature_schema_hash`).
        target: Target column name.
        artifacts: Optional ``{label: path}`` files to attach (e.g. the
            metrics/oof/summary outputs).
        tracking_uri: Explicit tracking store. Defaults to the gitignored
            local ``mlruns/`` directory — synthetic runs only.

    Returns:
        The MLflow run ID.

    Raises:
        DataValidationError: On an invalid stage.
        ImportError: If ``mlflow`` is not installed (see module docstring).
    """
    validate_stage(stage)
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "MLflow tracking requires the optional 'tracking' dependency: "
            "pip install -e \".[tracking]\" (or uv sync --extra tracking). "
            "The core package intentionally works without it."
        ) from exc

    if tracking_uri is None:
        # SQLite store in a gitignored directory: the recommended MLflow 3.x
        # local backend (the file backend is in maintenance mode). Artifacts
        # land beside it under mlartifacts/ (also gitignored).
        store = Path(DEFAULT_TRACKING_DIR).resolve()
        store.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{store / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name=run_name) as active:
        # Environment and contract fingerprints first: a run without them
        # is not reproducible and should be visibly incomplete.
        mlflow.log_params({
            "analysis_stage": stage,
            "data_provenance": provenance,
            "seed": seed,
            "git_sha": git_sha_value,
            "lock_hash": lock_hash_value,
            "feature_schema_hash": schema_hash_value,
            "target": target,
            **{f"config.{key}": value for key, value in config.items()},
        })
        for model_name, params in model_params.items():
            mlflow.log_params({
                f"model.{_safe_key(model_name)}.{key}": value
                for key, value in params.items()
            })
        for model_name, metrics in aggregate_metrics.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{_safe_key(model_name)}/{metric}", float(value))
        for model_name, folds in (fold_metrics or {}).items():
            for fold in folds:
                for metric in ("MAE", "RMSE"):
                    mlflow.log_metric(
                        f"{_safe_key(model_name)}/fold{fold['fold']}/{metric}",
                        float(fold[metric]),
                    )
        for model_name, per_metric in (bootstrap_summary or {}).items():
            for metric, ci in per_metric.items():
                payload = ci.to_dict() if hasattr(ci, "to_dict") else ci
                for bound in ("point", "ci_low", "ci_high"):
                    mlflow.log_metric(
                        f"{_safe_key(model_name)}/bootstrap_{metric}_{bound}",
                        float(payload[bound]),
                    )
        for model_name, per_metric in (permutation_summary or {}).items():
            for metric, result in per_metric.items():
                payload = result.summary() if hasattr(result, "summary") else result
                mlflow.log_metric(
                    f"{_safe_key(model_name)}/permutation_{metric}_p",
                    float(payload["p_value"]),
                )
        mlflow.set_tags({
            "analysis_stage": stage,
            "data_provenance": provenance,
            "research_only": "true",
            "not_for_clinical_use": "true",
        })
        for label, path in (artifacts or {}).items():
            mlflow.log_artifact(str(path), artifact_path=label)
        return active.info.run_id


def _safe_key(name: str) -> str:
    """Make a model name safe for metric/param keys (spaces and slashes out)."""
    return name.replace(" ", "_").replace("/", "_")

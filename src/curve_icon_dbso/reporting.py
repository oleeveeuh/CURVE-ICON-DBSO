"""Output tables and provenance-labeled reports.

Every artifact written by this package carries an explicit
``data_provenance`` field (``synthetic`` or a user-supplied descriptor)
so that pipeline-verification output can never masquerade as study
results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import CVResult, small_sample_warning

PROVENANCE_SYNTHETIC = "synthetic — pipeline verification only, NOT study results"


def metrics_table(results: list[CVResult]) -> pd.DataFrame:
    """One row per model with MAE/RMSE/R²/Pearson r and sample sizes."""
    rows = []
    for result in results:
        rows.append({"Model": result.model_name, **result.metrics})
    return pd.DataFrame(rows)


def per_subject_table(results: list[CVResult]) -> pd.DataFrame:
    """Long-format per-subject out-of-fold predictions and errors."""
    frames = []
    for result in results:
        frame = result.per_subject.copy()
        frame.insert(0, "Model", result.model_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def format_summary(
    results: list[CVResult],
    n_subjects: int,
    n_recordings: int,
    provenance: str,
    cohort_note: str = "",
) -> str:
    """Human-readable summary text with the small-sample caveat up front."""
    lines = [
        "=" * 72,
        "SUBJECT-GROUPED EVALUATION SUMMARY (leave-one-subject-out)",
        "=" * 72,
        f"Data provenance : {provenance}",
        f"Cohort          : {n_subjects} unique subjects, {n_recordings} recordings",
        cohort_note,
        "",
        small_sample_warning(n_subjects),
        "",
        f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Pearson r':>10}",
        "-" * 60,
    ]
    for result in results:
        m = result.metrics
        lines.append(
            f"{result.model_name:<20} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
            f"{m['R2']:>8.3f} {m['Pearson_r']:>10.3f}"
        )
    lines += [
        "",
        "Interpretation rules:",
        "  * A model is not considered useful unless it beats both DummyRegressor",
        "    baselines (mean and median) on MAE and RMSE.",
        "  * Negative R² means the model predicts worse than predicting the",
        "    training-set mean — an expected outcome at this sample size.",
        "  * These are exploratory feasibility numbers, not clinical performance.",
        "=" * 72,
    ]
    return "\n".join(line for line in lines if line is not None)


def save_outputs(
    output_dir: str | Path,
    results: list[CVResult],
    n_subjects: int,
    n_recordings: int,
    provenance: str,
) -> dict[str, Path]:
    """Write metrics CSV, aligned per-subject OOF predictions CSV, and a
    JSON summary. Returns the paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "metrics.csv"
    metrics_table(results).to_csv(metrics_csv, index=False)

    oof_csv = output_dir / "oof_predictions_per_subject.csv"
    per_subject_table(results).to_csv(oof_csv, index=False)

    beats_baseline = _baseline_comparison(results)
    summary_json = {
        "data_provenance": provenance,
        "intended_use": "exploratory research prototype; NOT for clinical use",
        "evaluation": "leave-one-subject-out cross-validation",
        "n_subjects": n_subjects,
        "n_recordings": n_recordings,
        "small_sample_warning": small_sample_warning(n_subjects),
        "beats_mean_baseline_on_mae": beats_baseline,
        "models": {
            result.model_name: {k: _jsonable(v) for k, v in result.metrics.items()}
            for result in results
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2) + "\n")

    return {"metrics": metrics_csv, "oof": oof_csv, "summary": summary_path}


def _jsonable(value):
    """Convert numpy scalars to plain Python for JSON serialization."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _baseline_comparison(results: list[CVResult]) -> dict[str, bool]:
    """Which models beat the mean baseline on MAE (empty when no baseline run)."""
    baselines = {r.model_name: r for r in results if "baseline" in r.model_name.lower()}
    if "Mean baseline" not in baselines:
        return {}
    ref = baselines["Mean baseline"].metrics["MAE"]
    return {
        r.model_name: bool(r.metrics["MAE"] < ref)
        for r in results
        if "baseline" not in r.model_name.lower()
    }

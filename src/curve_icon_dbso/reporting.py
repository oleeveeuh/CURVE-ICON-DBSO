"""Output tables and provenance-labeled reports.

Every artifact written by this package carries an explicit
``data_provenance`` field (``synthetic`` or a user-supplied descriptor)
so that pipeline-verification output can never masquerade as study
results. Summaries additionally carry the run's provenance fingerprints
(git SHA, lock hash, feature-schema hash, seed, analysis stage — REP-005)
and the cohort/protocol context required to interpret the metrics
(SCI-012). R² is never labeled "accuracy" anywhere in this package, and
no output revives the retracted historical result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .bootstrap import BootstrapCI
from .evaluation import CVResult, small_sample_warning
from .tracking import ANALYSIS_STAGES

PROVENANCE_SYNTHETIC = "synthetic — pipeline verification only, NOT study results"

# Where real per-subject outputs must go: outside the repository
# (docs/data_access.md). Synthetic demo outputs live in gitignored paths.
UNIT_OF_OBSERVATION = "subject (all recordings of a subject held out together)"


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
    fold_strategy: str | None = None,
) -> str:
    """Human-readable summary text with the small-sample caveat up front.

    ``fold_strategy`` names the splitter actually used (e.g. the
    :func:`curve_icon_dbso.evaluation.describe_fold_strategy` string) —
    the header must never claim a fold scheme the run didn't use
    (2026-11-16 review: the 31-subject run is 5-fold GroupKFold, not LOSO).
    """
    lines = [
        "=" * 72,
        "SUBJECT-GROUPED EVALUATION SUMMARY "
        f"({fold_strategy or 'subject-grouped cross-validation'})",
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
    target: str | None = None,
    analysis_stage: str = "exploratory",
    fold_strategy: str | None = None,
    missing_data_handling: str | None = None,
    run_provenance: dict | None = None,
    bootstrap_summary: dict[str, dict[str, BootstrapCI]] | None = None,
    permutation_summary: dict | None = None,
) -> dict[str, Path]:
    """Write metrics CSV, aligned per-subject OOF predictions CSV, and a
    machine-readable ``summary.json`` carrying the full run contract
    (provenance, protocol context, baseline comparison, uncertainty).

    Real-data callers are recommended to point ``output_dir`` outside the
    repository (docs/data_access.md); this function does not enforce it —
    the CLI entry points print a non-blocking reminder
    (:func:`curve_icon_dbso.kinematics.resolve_output_dir`).
    """
    if analysis_stage not in ANALYSIS_STAGES:
        raise ValueError(
            f"analysis_stage must be one of {ANALYSIS_STAGES}, got {analysis_stage!r}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "metrics.csv"
    metrics_table(results).to_csv(metrics_csv, index=False)

    oof_csv = output_dir / "oof_predictions_per_subject.csv"
    per_subject_table(results).to_csv(oof_csv, index=False)

    beats_baseline = _baseline_comparison(results)
    strict_beats = _beats_baseline_strict(results)
    summary_json = {
        "data_provenance": provenance,
        "intended_use": "exploratory research prototype; NOT for clinical use",
        # Never claim a fold scheme the run didn't use: the label follows
        # the fold_strategy the caller recorded (2026-11-16 review).
        "evaluation": fold_strategy or "subject-grouped cross-validation",
        "analysis_stage": analysis_stage,
        "n_subjects": n_subjects,
        "n_recordings": n_recordings,
        "unit_of_observation": UNIT_OF_OBSERVATION,
        "target": target,
        "fold_strategy": fold_strategy,
        "missing_data_handling": missing_data_handling,
        "small_sample_warning": small_sample_warning(n_subjects),
        "beats_mean_baseline_on_mae": beats_baseline,
        "models": {
            result.model_name: {
                **{k: _jsonable(v) for k, v in result.metrics.items()},
                "beats_baseline": strict_beats.get(result.model_name),
            }
            for result in results
        },
        "run_provenance": run_provenance or {},
    }
    if bootstrap_summary is not None:
        summary_json["bootstrap"] = {
            model: {metric: ci.to_dict() for metric, ci in per_metric.items()}
            for model, per_metric in bootstrap_summary.items()
        }
    if permutation_summary is not None:
        from .bootstrap import permutation_caveat

        summary_json["permutation_null"] = {
            model: {metric: result.summary() for metric, result in per_metric.items()}
            for model, per_metric in permutation_summary.items()
        }
        summary_json["permutation_interpretation"] = permutation_caveat(n_subjects)

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


def _beats_baseline_strict(results: list[CVResult]) -> dict[str, bool]:
    """Strict usefulness rule (methodology): beat BOTH mean and median
    baselines on BOTH MAE and RMSE. Baselines themselves are not candidates
    (absent from the result). Missing baselines -> not claimable -> False.

    SCI-014: a model that does not beat the baseline may still be served as
    an engineering demonstration, but only with ``beats_baseline: false``
    visible in its metadata — which is exactly this field."""
    baselines = {
        r.model_name: r.metrics for r in results if "baseline" in r.model_name.lower()
    }
    required = {"Mean baseline", "Median baseline"}
    verdicts: dict[str, bool] = {}
    for result in results:
        if "baseline" in result.model_name.lower():
            continue
        if not required.issubset(baselines):
            verdicts[result.model_name] = False
            continue
        verdicts[result.model_name] = bool(
            all(
                result.metrics[metric] < baselines[name][metric]
                for metric in ("MAE", "RMSE")
                for name in required
            )
        )
    return verdicts


def beats_baseline_strict(results: list[CVResult]) -> dict[str, bool]:
    """Public alias of :func:`_beats_baseline_strict` for artifact builders."""
    return _beats_baseline_strict(results)

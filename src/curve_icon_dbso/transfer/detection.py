"""Track A — cross-subject REST/MOVE detection (v2 contract §3, §7, §8).

Primary: class-weighted Ridge logistic regression, alpha selected by
inner subject-grouped GroupKFold(3) on outer-training subjects only
(mean balanced accuracy; ties → smaller frozen grid index). Comparator:
the fixed shallow XGBoost classifier on identical inputs. Outer unit:
true LOSO — every recording held out exactly once.

Null (frozen): within-held-out-subject block-kind permutation — the
held-out recording's block-kind assignment is permuted (balance- and
autocorrelation-preserving; the trained decoder is untouched), 1,000
draws, seed 42, conservative p = (1 + #{null ≥ observed}) / (1 + n).
Majority-class baseline: a constant-class prediction scores balanced
accuracy 0.5 by construction (reported as the chance floor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..bootstrap import generate_cluster_resamples
from ..errors import DataValidationError
from . import contract


@dataclass
class DetectionResult:
    per_subject: pd.DataFrame
    median_bac: float
    iqr_bac: tuple[float, float]
    bootstrap: dict
    null: dict
    majority_baseline_bac: float = 0.5
    qc: dict = field(default_factory=dict)

    @property
    def verdict(self) -> dict:
        """CRITERION verdict only (effect + CI + baseline; v2.1.0): the
        permutation p-values are reported for the FINAL adjustment
        elsewhere — ``success`` here deliberately does NOT gate on any
        Holm p, because the displayed detection verdict is derived from
        the primary-family adjusted p in ``pipeline._verdicts`` (the
        model-family ``p_holm`` below is descriptive only and never
        feeds the primary family)."""
        ci_lo, ci_hi = self.bootstrap["median_bac_ci"]
        return {
            "median_bac": self.median_bac,
            "ci": list(self.bootstrap["median_bac_ci"]),
            "ci_excludes_chance": bool(ci_lo > 0.5 or ci_hi < 0.5),
            "above_majority_baseline": bool(self.median_bac > self.majority_baseline_bac),
            "null_p_raw": self.null.get("p_raw"),
            "null_p_holm_model_family_descriptive": self.null.get("p_holm"),
            "criterion_met": bool(
                self.median_bac > 0.5
                and ci_lo > 0.5
                and self.median_bac > self.majority_baseline_bac
            ),
            "success": bool(
                self.median_bac > 0.5
                and ci_lo > 0.5
                and self.median_bac > self.majority_baseline_bac
            ),
        }


def _make_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # L2-penalized logistic (the default penalty for liblinear);
        # C = 1/alpha maps the frozen Ridge alpha grid onto C.
        ("clf", LogisticRegression(
            C=1.0 / alpha if alpha > 0 else 1e12,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=contract.NULL_SEED,
        )),
    ])


def _select_alpha(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    """Inner subject-grouped GroupKFold(3) selection on OUTER-TRAINING
    rows only; ties resolve to the smaller frozen grid index."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    n_splits = min(3, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for alpha in contract.RIDGE_ALPHAS:
        fold_scores: list[float] = []
        for train_idx, val_idx in splitter.split(X, y, groups):
            pipe = _make_pipeline(alpha)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            fold_scores.append(balanced_accuracy_score(
                y[val_idx], pipe.predict(X.iloc[val_idx])))
        scores.append(float(np.mean(fold_scores)))
    best = int(np.argmax(scores))  # np.argmax returns the FIRST maximum — smaller index wins ties
    return contract.RIDGE_ALPHAS[best]


def _xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # optional dependency (DD-003 convention)
        raise ImportError(
            "The optional XGBoost comparator requires the 'analysis' extra: "
            'pip install -e ".[analysis]". It is skipped cleanly when absent.'
        ) from exc
    return XGBClassifier(**{**contract.XGB_FIXED_PARAMS, "eval_metric": "logloss"})


def _balanced_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_score: np.ndarray | None) -> dict:
    metrics = {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    positives = y_true == 1
    negatives = ~positives
    metrics["sensitivity"] = float((y_pred[positives] == 1).mean()) if positives.any() else np.nan
    metrics["specificity"] = float((y_pred[negatives] == 0).mean()) if negatives.any() else np.nan
    if y_score is not None and positives.any() and negatives.any():
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auroc"] = np.nan
    return metrics


def event_detection_rate(labels: np.ndarray, predictions: np.ndarray,
                         block_ids: np.ndarray, min_consecutive: int = 3) -> float:
    """Fraction of MOVE blocks with ≥ ``min_consecutive`` consecutive
    positive updates (the frozen event rule, fixed before execution)."""
    detected, total = 0, 0
    for block in np.unique(block_ids):
        mask = block_ids == block
        if not labels[mask].any():
            continue  # REST blocks do not count toward the rate
        total += 1
        run = 0
        for prediction in predictions[mask]:
            run = run + 1 if prediction == 1 else 0
            if run >= min_consecutive:
                detected += 1
                break
    return float(detected / total) if total else np.nan


def loso_detection_cv(
    tables: list,   # list[FeatureTable]; one per recording, all blocks, lag-0 columns
    with_xgboost: bool = True,
    workers: int = 1,
) -> tuple[dict[str, DetectionResult], dict[str, list]]:
    """True LOSO detection CV over recordings.

    Every table's rows carry ``label`` and ``meta`` (block ids). Returns
    per-model results keyed by model name plus the fitted held-out-fold
    models (consumed by the frozen block-permutation null — the null
    never refits anything). ``workers > 1`` runs folds in fork-based
    worker processes (pure performance; identical, seed-deterministic
    output).
    """
    n = len(tables)
    if n < 3:
        raise DataValidationError("LOSO detection needs at least 3 recordings")
    models: dict[str, str] = {"ridge_logistic": "ridge"}
    if with_xgboost:
        try:
            _xgboost_classifier()
            models["xgboost_fixed"] = "xgb"
        except ImportError:
            models["xgboost_fixed_not_executed"] = "skip"

    per_model_rows: dict[str, list[dict]] = {
        name: [] for name in models if models[name] != "skip"
    }
    trained_models: dict[str, list] = {name: [] for name in per_model_rows}

    if workers > 1 and n > 2:
        import curve_icon_dbso.transfer.pipeline as _pl

        _pl._PARALLEL_TABLES = tables
        _pl._PARALLEL_DET_MODELS = models
        fold_results = _pl._map_folds(_pl._detection_fold_worker, n, workers)
        for rows_by_model, models_by_name in fold_results:
            for model_name, row in rows_by_model.items():
                per_model_rows[model_name].append(row)
                trained_models[model_name].append(models_by_name[model_name])
    else:
        for held_out in range(n):
            train_tables = [t for i, t in enumerate(tables) if i != held_out]
            X_train = pd.concat([t.features for t in train_tables], ignore_index=True)
            y_train = np.concatenate([np.asarray(t.label) for t in train_tables])
            g_train = np.concatenate([
                np.full(len(t.features), i) for i, t in enumerate(train_tables)
            ])
            test_table = tables[held_out]
            X_test = test_table.features
            y_test = np.asarray(test_table.label)
            block_keys = pd.factorize(test_table.meta["block"].values)[0]

            for model_name, kind in models.items():
                if kind == "skip":
                    continue
                if kind == "ridge":
                    alpha = _select_alpha(X_train, y_train, g_train)
                    model = _make_pipeline(alpha)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_score = model.predict_proba(X_test)[:, 1]
                    chosen = {"alpha": alpha}
                else:
                    model = _xgboost_classifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_score = model.predict_proba(X_test)[:, 1]
                    chosen = dict(contract.XGB_FIXED_PARAMS)
                trained_models[model_name].append(model)
                metrics = _balanced_metrics(y_test, y_pred, y_score)
                metrics["event_detection_rate"] = event_detection_rate(
                    y_test, y_pred, block_keys)
                per_model_rows[model_name].append({
                    "held_out_position": held_out,
                    "n_windows": int(len(y_test)),
                    "chosen": chosen,
                    **metrics,
                })

    results: dict[str, DetectionResult] = {}
    for model_name, rows in per_model_rows.items():
        frame = pd.DataFrame(rows)
        bac = frame["balanced_accuracy"].to_numpy(dtype=float)
        median = float(np.median(bac))
        iqr = (float(np.percentile(bac, 25)), float(np.percentile(bac, 75)))
        ci = _median_bootstrap_ci(bac)
        results[model_name] = DetectionResult(
            per_subject=frame,
            median_bac=median,
            iqr_bac=iqr,
            bootstrap={"median_bac_ci": ci,
                       "n_resamples": contract.BOOTSTRAP_N_RESAMPLES,
                       "seed": contract.BOOTSTRAP_SEED},
            null={},
            qc={},
        )
    return results, trained_models


def _median_bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Subject-cluster bootstrap percentile CI for a per-subject median
    (2,000 resamples, seed 42, reused cluster-draw machinery)."""
    values = np.asarray(values, dtype=float)
    groups = np.arange(values.size)
    draws = generate_cluster_resamples(groups, contract.BOOTSTRAP_N_RESAMPLES,
                                       contract.BOOTSTRAP_SEED)
    stats = np.asarray([float(np.median(values[idx])) for idx in draws])
    lo = float(np.percentile(stats, 100 * (1 - contract.CI_LEVEL) / 2))
    hi = float(np.percentile(stats, 100 * (1 + contract.CI_LEVEL) / 2))
    return lo, hi


def detection_null_pvalues(
    tables: list,
    fold_models: dict[str, list],
    observed_median: dict[str, float],
    n_draws: int = contract.DETECTION_NULL_DRAWS,
    seed: int = contract.NULL_SEED,
    observed_secondary: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Per-model block-permutation null p-values for the observed
    across-subject median BAC (conservative convention) — and, since
    v2.1.0, for the SECONDARY detection family (``auroc``,
    ``sensitivity``, ``specificity``, ``event_detection_rate``), which
    is Holm-adjusted within itself from the SAME draws.

    Each draw permutes every held-out recording's block-kind assignment
    independently and re-scores that recording with ITS OWN unchanged
    fold model (never a refit); the null statistic is the median across
    subjects. ``fold_models`` maps model name → the per-fold fitted
    models from ``loso_detection_cv``. Predictions are computed once per
    fold (the trained decoder is never touched by the permutation)."""
    rng = np.random.default_rng(seed)
    n = len(tables)
    names = list(fold_models)
    null_stats: dict[str, list[float]] = {name: [] for name in names}
    secondary_nulls: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in contract.SECONDARY_FAMILY_DETECTION}
        for name in names
    }
    skipped = 0
    # Predictions are label-independent: score each fold model ONCE.
    per_fold_predictions: dict[str, list] = {}
    for held_out in range(n):
        table = tables[held_out]
        for name in names:
            model = fold_models[name][held_out]
            y_pred = model.predict(table.features)
            y_score = model.predict_proba(table.features)[:, 1]
            per_fold_predictions.setdefault(name, []).append((y_pred, y_score))
    for _draw in range(n_draws):
        per_subject: dict[str, list[float]] = {name: [] for name in names}
        per_subject_secondary: dict[str, dict[str, list[float]]] = {
            name: {metric: [] for metric in contract.SECONDARY_FAMILY_DETECTION}
            for name in names
        }
        for held_out in range(n):
            table = tables[held_out]
            block_numbers = np.asarray(table.meta["block"].values)
            kinds = np.asarray(table.meta["block_kind"].values)
            unique_blocks = np.unique(block_numbers)
            block_kinds = np.asarray(
                [kinds[block_numbers == b][0] for b in unique_blocks])
            n_move = int((block_kinds == "move").sum())
            if n_move in (0, len(unique_blocks)):
                skipped += 1
                continue
            perm = rng.permutation(len(unique_blocks))
            relabeled = np.zeros(len(block_numbers), dtype=int)
            for new_idx, block in enumerate(unique_blocks):
                relabeled[block_numbers == block] = (
                    1 if block_kinds[perm[new_idx]] == "move" else 0)
            for name in names:
                y_pred, y_score = per_fold_predictions[name][held_out]
                per_subject[name].append(
                    float(balanced_accuracy_score(relabeled, y_pred)))
                metrics = _balanced_metrics(relabeled, y_pred, y_score)
                metrics["event_detection_rate"] = event_detection_rate(
                    relabeled, y_pred, block_numbers)
                for metric in contract.SECONDARY_FAMILY_DETECTION:
                    value = metrics.get(metric, np.nan)
                    if np.isfinite(value):
                        per_subject_secondary[name][metric].append(value)
        for name in names:
            if per_subject[name]:
                null_stats[name].append(float(np.median(per_subject[name])))
            for metric, values in per_subject_secondary[name].items():
                if values:
                    secondary_nulls[name][metric].append(float(np.median(values)))
    out: dict = {"n_draws": n_draws, "seed": seed,
                 "recording_draws_skipped_too_few_blocks": skipped}
    for name, values in null_stats.items():
        arr = np.asarray(values, dtype=float)
        observed = observed_median[name]
        count = int(np.count_nonzero(arr >= observed)) if arr.size else 0
        p_raw = (1 + count) / (1 + arr.size) if arr.size else 1.0
        out[name] = {
            "observed_median_bac": observed,
            "null_median_mean": float(np.mean(arr)) if arr.size else np.nan,
            "null_median_p95": float(np.percentile(arr, 95)) if arr.size else np.nan,
            "count_at_least_observed": count,
            "p_raw": float(p_raw),
        }
        # Secondary family from the SAME draws (higher = favorable for
        # all four members); Holm within the family.
        family_p: dict[str, float] = {}
        family_detail: dict[str, dict] = {}
        for metric in contract.SECONDARY_FAMILY_DETECTION:
            sec_arr = np.asarray(secondary_nulls[name][metric], dtype=float)
            observed_value = np.nan
            if observed_secondary and name in observed_secondary:
                observed_value = float(observed_secondary[name].get(metric, np.nan))
            if sec_arr.size and np.isfinite(observed_value):
                sec_count = int(np.count_nonzero(sec_arr >= observed_value))
                family_p[metric] = float((1 + sec_count) / (1 + sec_arr.size))
            else:
                family_p[metric] = 1.0
            family_detail[metric] = {
                "observed_median": observed_value,
                "null_median": float(np.median(sec_arr)) if sec_arr.size else np.nan,
                "p_raw": family_p[metric],
            }
        out[name]["secondary_family"] = {
            "family": list(contract.SECONDARY_FAMILY_DETECTION),
            "p_raw": family_p,
            "p_holm_within_family": holm_adjust(family_p),
            "detail": family_detail,
        }
    return out


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over a named family."""
    names = list(pvalues)
    ordered = sorted(names, key=lambda k: pvalues[k])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, name in enumerate(ordered):
        value = min(1.0, (m - i) * pvalues[name])
        running_max = max(running_max, value)
        adjusted[name] = float(running_max)
    return adjusted

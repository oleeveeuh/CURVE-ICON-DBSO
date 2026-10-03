"""Corrected subject-grouped evaluation.

This module is the fix for the two central flaws of the historical
pipeline:

1. **Prediction misalignment.** The historical code appended fold
   predictions in cross-validation order and later paired them with
   targets and subject IDs in original dataframe order, silently
   scoring many predictions against the wrong subject. Here an
   out-of-fold array is **preallocated** and written back with
   ``oof[test_idx] = predictions``, so every prediction stays attached
   to the row — and therefore the subject — that produced it.

2. **Leakage.** The historical code selected features, imputed and
   scaled on the full dataset before cross-validation. Here all
   preprocessing lives inside a :class:`sklearn.pipeline.Pipeline` and
   is refit on every training fold; no feature selection is performed
   at all (the feature set is prespecified, see
   :mod:`curve_icon_dbso.features`).

Subjects — not rows — are the units of evaluation:
:class:`~sklearn.model_selection.LeaveOneGroupOut` is used by default
so that repeated recordings from one subject never straddle the
train/test boundary. ``DummyRegressor`` mean/median baselines are
provided because no model should be called "better than chance"
without beating them.

With eight subjects every metric is extremely unstable: leave-one-
subject-out gives exactly eight held-out predictions, and R² (whose
expectation is negative for any model no better than the mean at these
sample sizes) and correlation can swing wildly under small perturbations.
Treat these numbers as feasibility checks, not performance estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .errors import DataValidationError

# n at or below this gets LeaveOneGroupOut; the statistical instability
# warning kicks in at or below SMALL_N_WARNING_THRESHOLD subjects.
LOSO_MAX_GROUPS = 12
SMALL_N_WARNING_THRESHOLD = 10


@dataclass
class CVResult:
    """Out-of-fold results for one model, aligned to original row order.

    Attributes:
        model_name: Human-readable model name.
        oof_predictions: Array of length ``n_rows``; entry ``i`` is the
            prediction for row ``i`` made by a model that never saw
            row ``i`` (or any row from its subject) during training.
        fold_id: Fold index per row, ``-1`` if unassigned (never the
            case after a successful run).
        metrics: MAE, RMSE, R², Pearson r and sample-size counts.
        per_subject: One row per held-out subject with true value,
            prediction and error.
    """

    model_name: str
    oof_predictions: np.ndarray
    fold_id: np.ndarray
    metrics: dict[str, float | int]
    per_subject: pd.DataFrame


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute MAE, RMSE, R² and Pearson r from aligned vectors.

    Kept dependency-free of the CV machinery so it can be verified
    against hand-calculated values (see tests/test_metrics.py).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise DataValidationError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size < 2:
        raise DataValidationError("At least 2 samples are required to compute metrics")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson_r": pearson,
            "n_rows": int(y_true.size)}


def small_sample_warning(n_subjects: int) -> str:
    """Return the standard small-cohort caveat for reported metrics."""
    if n_subjects > SMALL_N_WARNING_THRESHOLD:
        return ""
    return (
        f"WARNING: metrics are computed from n={n_subjects} held-out subjects. "
        "R² and correlation are highly unstable at this sample size (a single subject "
        "can change the sign of R²). Values describe this cohort only and must not be "
        "read as generalizable performance."
    )


def _default_splitter(n_groups: int):
    """LOSO for small cohorts; 5-fold GroupKFold above ``LOSO_MAX_GROUPS``."""
    if n_groups <= LOSO_MAX_GROUPS:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=5)


def _check_group_integrity(groups: np.ndarray, splits) -> None:
    """Defence in depth: assert subject sets never straddle folds."""
    for train_idx, test_idx in splits:
        train_subjects = set(np.asarray(groups)[train_idx].tolist())
        test_subjects = set(np.asarray(groups)[test_idx].tolist())
        overlap = train_subjects & test_subjects
        if overlap:
            raise DataValidationError(
                f"Subject groups appear in both train and test folds: {sorted(overlap)}. "
                "This is a splitter bug; refusing to evaluate."
            )


def make_preprocessing() -> list[tuple[str, object]]:
    """Fold-local preprocessing steps (median impute, standardize).

    Returned steps are fit by the Pipeline on training data only —
    never on the full dataset.
    """
    from sklearn.impute import SimpleImputer  # local import keeps module import light

    return [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]


def baseline_models() -> dict[str, DummyRegressor]:
    """Mean and median ``DummyRegressor`` baselines."""
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Median baseline": DummyRegressor(strategy="median"),
    }


def default_models(model_cfg=None) -> dict[str, object]:
    """Fixed, documented models (no hyperparameter tuning anywhere)."""
    cfg = model_cfg
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators if cfg else 200,
            max_depth=cfg.rf_max_depth if cfg else 3,
            random_state=cfg.seed if cfg else 42,
            # n_jobs=1 (not -1): parallel tree-averaging sums floats in
            # nondeterministic order, breaking byte-identical reruns.
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=cfg.gbr_n_estimators if cfg else 50,
            max_depth=cfg.gbr_max_depth if cfg else 2,
            learning_rate=cfg.gbr_learning_rate if cfg else 0.05,
            random_state=cfg.seed if cfg else 42,
        ),
        "Linear Regression": LinearRegression(),
    }


def run_group_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict[str, object],
    preprocessing: list[tuple[str, object]] | None = None,
    splitter=None,
) -> list[CVResult]:
    """Run subject-grouped cross-validation with aligned out-of-fold predictions.

    Args:
        X: Feature matrix (pandas DataFrame; row order defines the
            alignment contract).
        y: Continuous target, length ``len(X)``.
        groups: Subject label per row; rows sharing a label always land
            on the same side of every split.
        models: Mapping of model name to unfitted estimator.
        preprocessing: Pipeline steps fit within each training fold.
            Defaults to :func:`make_preprocessing`.
        splitter: Group-aware splitter instance. Defaults to
            LeaveOneGroupOut (<= 12 subjects) else GroupKFold(5).

    Returns:
        One :class:`CVResult` per model, each with out-of-fold
        predictions indexed identically to ``X``.

    Raises:
        DataValidationError: On shape mismatches, missing groups, a
            splitter that leaves rows unpredicted, or any train/test
            subject overlap.
    """
    if not isinstance(X, pd.DataFrame):
        raise DataValidationError("X must be a pandas DataFrame so row alignment is explicit")
    n = len(X)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if y.shape[0] != n or groups.shape[0] != n:
        raise DataValidationError(
            f"X, y and groups must have the same number of rows ({n}); got "
            f"y={y.shape[0]}, groups={groups.shape[0]}"
        )
    if len(np.unique(groups)) < 2:
        raise DataValidationError("Need at least 2 unique subjects for grouped evaluation")

    splitter = splitter or _default_splitter(len(np.unique(groups)))
    preprocessing = preprocessing if preprocessing is not None else make_preprocessing()

    splits = list(splitter.split(X, y, groups))
    _check_group_integrity(groups, splits)

    results: list[CVResult] = []
    for model_name, model in models.items():
        # Preallocated out-of-fold array: predictions are written back
        # at their ORIGINAL row indices, never appended in fold order.
        oof = np.full(n, np.nan)
        fold_id = np.full(n, -1, dtype=int)

        for fold, (train_idx, test_idx) in enumerate(splits):
            steps = [(name, clone(step)) for name, step in preprocessing]
            pipe = Pipeline([*steps, ("model", clone(model))])
            pipe.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict(X.iloc[test_idx])
            fold_id[test_idx] = fold

        unassigned = int(np.isnan(oof).sum())
        if unassigned:
            raise DataValidationError(
                f"Splitter left {unassigned} of {n} rows without an out-of-fold "
                "prediction; every row must be held out exactly once."
            )

        metrics = compute_regression_metrics(y, oof)
        metrics["n_subjects"] = int(len(np.unique(groups)))

        per_subject = (
            pd.DataFrame({
                "subject": groups,
                "y_true": y,
                "y_pred": oof,
                "fold_id": fold_id,
            })
            .groupby("subject", sort=True)
            .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"),
                 fold_id=("fold_id", "first"), n_rows=("y_pred", "size"))
            .reset_index()
        )
        per_subject["error"] = per_subject["y_pred"] - per_subject["y_true"]

        results.append(CVResult(
            model_name=model_name,
            oof_predictions=oof,
            fold_id=fold_id,
            metrics=metrics,
            per_subject=per_subject,
        ))
    return results


def preprocessing_steps(preprocessing: list[tuple[str, object]]):
    """Return a copy of the step list; steps are cloned per fold in
    :func:`run_group_cv` (cloning the (name, estimator) tuples directly
    is not supported by sklearn, so callers clone per element)."""
    return list(preprocessing)

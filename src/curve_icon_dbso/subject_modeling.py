"""Subject-level REST and DELTA modeling protocols for the follow-on
study (frozen contract v1.0.1, Prompt 3).

Nested subject-grouped evaluation of the prespecified model set — mean/
median baselines, Ridge (primary), linear SVR, Bayesian Ridge, fixed
Matérn-3/2 GP, and one optional shallow boosted-tree comparator — over
the seven distinct feature blocks (clinical, rest_spectral, rest_dynamics,
delta_spectral, delta_dynamics, combined, delta_combined), with
everything the contract freezes:

* **Identical outer folds** for every model and feature block:
  ``evaluation._default_splitter`` (LOSO ≤ 12 subjects, else
  ``GroupKFold(5)``), computed once and reused; group integrity checked
  on every run.
* **Inner selection on inner-training subjects only**:
  ``GroupKFold(3)`` over outer-training subjects (inner-LOSO when the
  outer-training cohort has fewer than 6 subjects), criterion = inner-CV
  mean MAE, ties → smaller frozen grid index. Imputation, scaling, and
  hyperparameter choice never see held-out subjects.
* **Aligned out-of-fold subject predictions** — the corrected preallocated
  write-back pattern, aggregated per subject for all statistics.
* Per-fold and aggregate **MAE, RMSE, R², Pearson, Spearman** (R²/
  correlation are ``None`` within folds holding fewer than 2 subjects).
* **Subject-cluster bootstrap intervals** and the **corrected
  target-permutation null through the complete nested pipeline** (each
  permutation reruns inner selection too; its null statistic is scored
  against that permutation's own permuted targets — never the original
  ones; conservative p-value convention).
* **Leave-one-subject influence** (aggregate metric deltas when each
  subject's rows are dropped from the evaluation table; descriptive).
* **Foldwise coefficient/sign stability** for linear models (standardized
  space — signs are comparable across folds only after the fold-local
  scaler; stated explicitly).
* **Prediction intervals** for Bayesian Ridge and the GP
  (``return_std`` through the fitted pipeline), labeled exploratory.
* **Promotion rule** (frozen, DD-006-compatible): a model promotes only
  if it improves BOTH MAE and RMSE versus BOTH naive baselines AND the
  frozen five-feature reference on the identical outer folds — never on
  a favorable correlation alone. No SHAP anywhere; no outer-test
  selection of any kind.

Verified on deterministic synthetic fixtures only (linear, nonlinear,
null, and outlier-driven scenarios plus deliberate fingerprint leakage
tests). The authorized real-data entry point is
``scripts/run_followon_subject_modeling.py`` — a separate local-only
command, not executed by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import protocol
from .errors import DataValidationError

# ---------------------------------------------------------------------------
# Frozen feature blocks
# ---------------------------------------------------------------------------

BLOCK_NAMES: tuple[str, ...] = (
    "clinical", "rest_spectral", "rest_dynamics",
    "delta_spectral", "delta_dynamics", "combined", "delta_combined",
)

CLINICAL_COLUMNS: tuple[str, ...] = ("age", "dbs_volt", "dbs_freq")

BAND_KEYS = ("theta", "alpha", "beta_low", "beta_high", "gamma_low", "gamma_high")

DYNAMICS_PREFIXES = (
    "hjorth_mobility", "hjorth_complexity", "line_length_per_s",
    "spectral_entropy", "beta_burst_occupancy",
    "beta_burst_mean_duration_ms", "beta_burst_median_amplitude_ratio",
)


def select_block(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    """Select one frozen feature block's columns from a per-recording or
    per-subject feature frame (exact names, no fuzzy matching).

    ``combined`` = rest_spectral + rest_dynamics + clinical;
    ``delta_combined`` = delta_spectral + delta_dynamics + clinical (the
    frozen §11 "DELTA analogously" block, completed in the v1.0.1
    corrective run). A missing column raises with the full missing list —
    nothing is imputed into existence here (fold-local imputation
    handles genuine missingness).
    """
    if block not in BLOCK_NAMES:
        raise DataValidationError(
            f"unknown feature block '{block}'; frozen blocks: {BLOCK_NAMES}"
        )
    columns: list[str] = []
    if block == "clinical":
        columns = list(CLINICAL_COLUMNS)
    elif block == "rest_spectral":
        columns = (
            [f"{band}_log_relative_power" for band in BAND_KEYS]
            + ["aperiodic_exponent", "aperiodic_offset"]
            + [f"{band}_power_aperiodic_adjusted" for band in BAND_KEYS]
            + ["hfo_power", "hfo_peak_freq"]
        )
    elif block == "rest_dynamics":
        columns = list(DYNAMICS_PREFIXES)
    elif block == "delta_spectral":
        columns = (
            [f"{band}_log_ratio_power" for band in BAND_KEYS]
            + ["aperiodic_exponent_delta", "aperiodic_offset_delta"]
            + [f"{band}_power_aperiodic_adjusted_log_ratio" for band in BAND_KEYS]
            + ["hfo_power_log_ratio"]
        )
    elif block == "delta_dynamics":
        columns = [f"{name}_delta" for name in DYNAMICS_PREFIXES]
    elif block == "combined":
        columns = (
            [f"{band}_log_relative_power" for band in BAND_KEYS]
            + ["aperiodic_exponent", "aperiodic_offset"]
            + [f"{band}_power_aperiodic_adjusted" for band in BAND_KEYS]
            + ["hfo_power", "hfo_peak_freq"]
            + list(DYNAMICS_PREFIXES)
            + list(CLINICAL_COLUMNS)
        )
    elif block == "delta_combined":
        columns = (
            [f"{band}_log_ratio_power" for band in BAND_KEYS]
            + ["aperiodic_exponent_delta", "aperiodic_offset_delta"]
            + [f"{band}_power_aperiodic_adjusted_log_ratio" for band in BAND_KEYS]
            + ["hfo_power_log_ratio"]
            + [f"{name}_delta" for name in DYNAMICS_PREFIXES]
            + list(CLINICAL_COLUMNS)
        )
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"feature block '{block}' is missing columns {missing}; present "
            f"columns: {sorted(map(str, frame.columns))}"
        )
    return frame[columns].astype(float)


# ---------------------------------------------------------------------------
# Prespecified models and bounded grids (frozen contract §8)
# ---------------------------------------------------------------------------

LINEAR_MODEL_NAMES = ("ridge_primary", "svr_linear", "bayesian_ridge")
INTERVAL_MODEL_NAMES = ("bayesian_ridge", "gp_matern32")


def grid_for(model_name: str) -> list[dict]:
    """Frozen hyperparameter grid as an ordered list of param dicts
    (declaration order = the frozen tie-break order)."""
    spec = protocol.MODELS_SUBJECT_LEVEL.get(model_name)
    if spec is None:
        raise DataValidationError(
            f"unknown model '{model_name}'; frozen models: "
            f"{sorted(protocol.MODELS_SUBJECT_LEVEL)}"
        )
    grid = spec.get("grid", {})
    if not grid:
        return [{}]
    combos = [{}]
    for param, values in grid.items():  # declaration order (dict = frozen order)
        combos = [dict(combo, **{param: value}) for combo in combos for value in values]
    return combos


def epsilon_from_mad(y_train: np.ndarray, factor: float) -> float:
    """Fold-local SVR epsilon: factor × training-target MAD. If the MAD is
    0 (constant targets), fall back to factor × std, then to factor —
    a documented, deterministic, training-only quantity."""
    y = np.asarray(y_train, dtype=float)
    mad = float(np.median(np.abs(y - np.median(y))))
    if mad > 0:
        return float(factor * mad)
    std = float(np.std(y))
    return float(factor * std) if std > 0 else float(factor)


def build_estimator(model_name: str, params: dict, seed: int = 42,
                    epsilon: float | None = None):
    """Unfitted estimator for one grid point (import-time optional for
    XGBoost per DD-003: callers skip it cleanly when absent)."""
    if model_name == "dummy_mean":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="mean")
    if model_name == "dummy_median":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="median")
    if model_name == "ridge_primary":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(params["alpha"]), random_state=seed)
    if model_name == "svr_linear":
        from sklearn.svm import SVR

        return SVR(kernel="linear", C=float(params["C"]),
                   epsilon=float(epsilon if epsilon is not None
                                 else params["epsilon_factor_of_train_MAD"]))
    if model_name == "bayesian_ridge":
        from sklearn.linear_model import BayesianRidge

        return BayesianRidge()
    if model_name == "gp_matern32":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        kernel = (
            C(1.0, constant_value_bounds="fixed")
            * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds="fixed")
        )
        return GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
            alpha=1e-6, random_state=seed,
        )
    if model_name == "xgboost_shallow":
        from .evaluation import xgboost_model

        return xgboost_model()
    raise DataValidationError(f"unknown model '{model_name}'")


def _supports_intervals(model_name: str) -> bool:
    return model_name in INTERVAL_MODEL_NAMES


def _collects_coefs(model_name: str) -> bool:
    return model_name in LINEAR_MODEL_NAMES


# ---------------------------------------------------------------------------
# Nested subject-grouped evaluation
# ---------------------------------------------------------------------------


@dataclass
class SubjectCVResult:
    """Nested-CV output for one (block, model) pair.

    ``oof`` is aligned to the ORIGINAL row order (preallocated write-back);
    ``subject_table`` aggregates predictions per subject. ``fold_metrics``
    holds per-fold MAE/RMSE plus R²/Pearson/Spearman when the fold held
    ≥ 2 subjects (else ``None``). ``prediction_std`` (Bayesian Ridge / GP
    only) is the per-row exploratory predictive std. ``fold_coefs`` holds
    linear-model coefficients in the fold-local standardized space.
    """

    model_name: str
    block: str
    oof: np.ndarray
    fold_id: np.ndarray
    subject_table: pd.DataFrame
    fold_metrics: list[dict]
    aggregate: dict
    prediction_std: np.ndarray | None = None
    fold_coefs: list[np.ndarray] = field(default_factory=list)
    selected_params: list[dict] = field(default_factory=list)


def inner_splitter(n_outer_train_subjects: int):
    """Frozen inner rule: GroupKFold(3), or LOSO within the outer-training
    subjects when that cohort has fewer than 6 subjects."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    if n_outer_train_subjects < protocol.INNER_LOSO_MIN_OUTER_TRAIN_SUBJECTS:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=protocol.INNER_SPLITS)


def _fold_local_mad_epsilon(model_name: str, params: dict, y_fit: np.ndarray) -> dict:
    fit_params = dict(params)
    if model_name == "svr_linear":
        fit_params["epsilon"] = epsilon_from_mad(
            y_fit, params["epsilon_factor_of_train_MAD"]
        )
    return fit_params


def select_params(
    model_name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
    seed: int = 42,
) -> dict:
    """Inner subject-grouped hyperparameter selection — fitted ONLY on the
    subjects passed in (the caller passes outer-training data only).

    Criterion: mean MAE across inner folds (``evaluation.make_preprocessing``
    inside each inner pipeline); ties resolve to the smaller frozen grid
    index. With an empty grid (fixed models) returns ``{}`` immediately —
    nothing is evaluated, nothing leaks.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from .evaluation import make_preprocessing

    combos = grid_for(model_name)
    if combos == [{}]:
        return {}
    splitter = inner_splitter(len(np.unique(groups)))
    inner_splits = list(splitter.split(X, y, groups))
    best_params: dict | None = None
    best_mae = float("inf")
    for params in combos:
        fold_maes: list[float] = []
        for train_idx, val_idx in inner_splits:
            fit_params = _fold_local_mad_epsilon(model_name, params, y[train_idx])
            steps = [(name, clone(step)) for name, step in make_preprocessing()]
            pipe = Pipeline([*steps, ("model", build_estimator(
                model_name, fit_params, seed=seed))]
            )
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[val_idx])
            fold_maes.append(float(np.mean(np.abs(pred - y[val_idx]))))
        mean_mae = float(np.mean(fold_maes))
        if mean_mae < best_mae:  # strict: ties keep the earlier grid index
            best_mae = mean_mae
            best_params = params
    return best_params if best_params is not None else combos[0]


def _subject_table(y: np.ndarray, oof: np.ndarray, groups: np.ndarray,
                   fold_id: np.ndarray) -> pd.DataFrame:
    table = (
        pd.DataFrame({
            "group": groups, "y_true": y, "y_pred": oof, "fold_id": fold_id,
        })
        .groupby("group", sort=True)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"),
             fold_id=("fold_id", "first"), n_rows=("y_pred", "size"))
        .reset_index()
    )
    table["error"] = table["y_pred"] - table["y_true"]
    return table


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        with_correlations: bool = True) -> dict:
    """MAE/RMSE always; R²/Pearson/Spearman only when there are ≥ 2
    samples with variance (else ``None`` — never fabricated)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict = {
        "MAE": float(np.mean(np.abs(y_pred - y_true))),
        "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "R2": None, "Pearson_r": None, "Spearman_rho": None,
    }
    if y_true.size < 2:
        return out
    from scipy.stats import pearsonr, spearmanr

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    out["R2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["Pearson_r"] = float(pearsonr(y_true, y_pred).statistic)
        out["Spearman_rho"] = float(spearmanr(y_true, y_pred).statistic)
    return out


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: tuple[str, ...],
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    seed: int = 42,
    include_intervals: bool = False,
    collect_coefs: bool = False,
    block: str = "block",
) -> list[SubjectCVResult]:
    """Nested subject-grouped CV with aligned OOF predictions.

    ``splits`` may be passed in to guarantee identical outer folds across
    models and blocks (``run_arm`` does exactly that); otherwise the
    frozen default splitter computes them once here. Inner selection sees
    outer-training subjects only; the chosen params are recorded per fold
    so no selection decision is ever hidden.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from .evaluation import _check_group_integrity, _default_splitter, make_preprocessing

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n = len(X)
    if not (y.shape[0] == groups.shape[0] == n) or n == 0:
        raise DataValidationError("X, y and groups must be aligned and non-empty")
    if len(np.unique(groups)) < 2:
        raise DataValidationError("need at least 2 subjects")
    if splits is None:
        splits = list(_default_splitter(len(np.unique(groups))).split(X, y, groups))
    _check_group_integrity(groups, splits)

    results: list[SubjectCVResult] = []
    for model_name in model_names:
        if model_name == "xgboost_shallow":
            try:
                build_estimator(model_name, {})
            except ImportError:
                continue  # optional DD-003 comparator, skipped cleanly
        oof = np.full(n, np.nan)
        fold_id = np.full(n, -1, dtype=int)
        prediction_std = (
            np.full(n, np.nan) if include_intervals and _supports_intervals(model_name)
            else None
        )
        fold_metrics: list[dict] = []
        fold_coefs: list[np.ndarray] = []
        selected_params: list[dict] = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            params = select_params(model_name, X.iloc[train_idx], y[train_idx],
                                   groups[train_idx], seed=seed)
            selected_params.append(dict(params))
            fit_params = _fold_local_mad_epsilon(model_name, params, y[train_idx])
            steps = [(name, clone(step)) for name, step in make_preprocessing()]
            pipe = Pipeline([*steps, ("model", build_estimator(
                model_name, fit_params, seed=seed))]
            )
            pipe.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict(X.iloc[test_idx])
            fold_id[test_idx] = fold
            if prediction_std is not None:
                transformed = pipe[:-1].transform(X.iloc[test_idx])
                _, std = pipe[-1].predict(transformed, return_std=True)
                prediction_std[test_idx] = std
            if collect_coefs and _collects_coefs(model_name):
                fold_coefs.append(np.asarray(pipe[-1].coef_, dtype=float))
            # Per-fold metrics on the subject-aggregated held-out rows.
            table = _subject_table(y[test_idx], oof[test_idx],
                                   groups[test_idx], fold_id[test_idx])
            fold_metrics.append(_regression_metrics(
                table["y_true"].to_numpy(), table["y_pred"].to_numpy(),
                with_correlations=table.shape[0] >= 2,
            ))
        if np.isnan(oof).any():
            raise DataValidationError(
                "splitter left rows without out-of-fold predictions; refusing "
                "to evaluate misaligned results"
            )
        subject_table = _subject_table(y, oof, groups, fold_id)
        aggregate = _regression_metrics(
            subject_table["y_true"].to_numpy(), subject_table["y_pred"].to_numpy()
        )
        aggregate["n_subjects"] = int(subject_table.shape[0])
        aggregate["n_rows"] = n
        results.append(SubjectCVResult(
            model_name=model_name, block=block,
            oof=oof, fold_id=fold_id, subject_table=subject_table,
            fold_metrics=fold_metrics, aggregate=aggregate,
            prediction_std=prediction_std, fold_coefs=fold_coefs,
            selected_params=selected_params,
        ))
    return results


def run_arm(
    frame: pd.DataFrame,
    target_col: str,
    groups_col: str,
    blocks: tuple[str, ...] = BLOCK_NAMES,
    model_names: tuple[str, ...] = (
        "dummy_mean", "dummy_median", "ridge_primary", "svr_linear",
        "bayesian_ridge", "gp_matern32",
    ),
    seed: int = 42,
    include_intervals: bool = False,
    collect_coefs: bool = False,
) -> dict:
    """One arm (REST or DELTA): every frozen feature block × model on the
    IDENTICAL outer folds, plus the frozen five-feature reference run on
    those same folds.

    ``frame`` is a per-subject (or per-recording with ``groups_col``)
    feature frame. Baselines are always included. The reference uses the
    frozen five kinematic predictors (``curve_icon_dbso.kinematics``
    KINEMATIC_FEATURES) with the original corrected model set — a frozen
    comparator, never re-tuned.
    """
    from .evaluation import (
        _default_splitter,
        baseline_models,
        default_models,
        run_group_cv,
    )
    from .kinematics import KINEMATIC_FEATURES

    if target_col not in frame.columns or groups_col not in frame.columns:
        raise DataValidationError("target/group columns missing from frame")
    y = frame[target_col].to_numpy(dtype=float)
    groups = frame[groups_col].to_numpy()
    splits = list(_default_splitter(len(np.unique(groups))).split(frame, y, groups))

    results: dict = {"blocks": {}, "reference": None,
                     "reference_aggregates": None}
    for block in blocks:
        X = select_block(frame, block)
        for column in X.columns:
            if not np.isfinite(X[column].to_numpy()).all():
                raise DataValidationError(
                    f"block '{block}' column '{column}' has non-finite values; "
                    "handle QC upstream (never impute into existence here)"
                )
        results["blocks"][block] = run_nested_cv(
            X, y, groups, model_names=tuple(model_names),
            splits=splits, seed=seed, block=block,
            include_intervals=include_intervals, collect_coefs=collect_coefs,
        )
    # Frozen five-feature reference on the identical outer folds: the
    # corrected original machinery, never re-tuned.
    missing = [c for c in KINEMATIC_FEATURES if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"five-feature reference columns missing: {missing}"
        )
    reference = run_group_cv(
        frame[KINEMATIC_FEATURES].astype(float), y, groups,
        {**baseline_models(), **default_models()}, splitter=None,
    )
    results["reference"] = reference
    # The recorded first-run selection: Linear Regression on the five
    # frozen predictors — the reference the promotion rule compares to.
    for reference_result in reference:
        if reference_result.model_name == "Linear Regression":
            results["reference_aggregates"] = reference_result.metrics
    return results


# ---------------------------------------------------------------------------
# Uncertainty, nulls, influence, stability, promotion
# ---------------------------------------------------------------------------


def leave_one_subject_influence(subject_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAE/RMSE deltas when each subject's rows are dropped from
    the evaluation table (descriptive; the dominant-subject detector).
    Subjects are referenced by their position in the table — never by any
    external identifier."""
    rows = []
    full_mae = float(np.mean(np.abs(subject_table["error"])))
    full_rmse = float(np.sqrt(np.mean(subject_table["error"] ** 2)))
    for position, (_, row) in enumerate(subject_table.iterrows()):
        rest = subject_table.drop(index=row.name)
        rows.append({
            "dropped_subject_position": position,
            "mae_delta": float(np.mean(np.abs(rest["error"])) - full_mae),
            "rmse_delta": float(np.sqrt(np.mean(rest["error"] ** 2)) - full_rmse),
            "n_remaining_subjects": int(rest.shape[0]),
        })
    return pd.DataFrame(rows)


def coefficient_sign_stability(fold_coefs: list[np.ndarray],
                               feature_names: list[str]) -> dict[str, float]:
    """Fraction of outer folds sharing the majority sign, per feature
    (standardized space — comparable across folds only after the fold-
    local scaler). Values near 0.5 mean the sign is fold-unstable."""
    if not fold_coefs:
        return {}
    matrix = np.vstack(fold_coefs)
    if matrix.shape[1] != len(feature_names):
        raise DataValidationError("coefficient width does not match features")
    stability: dict[str, float] = {}
    for position, name in enumerate(feature_names):
        signs = np.sign(matrix[:, position])
        signs = signs[signs != 0]
        if signs.size == 0:
            stability[name] = 0.0
            continue
        majority = 1.0 if signs.sum() > 0 else -1.0
        stability[name] = float(np.mean(signs == majority))
    return stability


def permutation_null_nested(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: tuple[str, ...],
    n_permutations: int = protocol.PERMUTATION_MIN_FOR_REPORTING,
    seed: int = protocol.PERMUTATION_SEED,
    metrics: tuple[str, ...] = ("MAE", "RMSE"),
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, dict[str, object]]:
    """Corrected subject-level label-permutation null through the COMPLETE
    nested pipeline (inner selection included in every permutation).

    Each permutation permutes the targets at the subject level (stratified
    by recording count — subjects, never rows), reruns
    :func:`run_nested_cv` with the identical outer folds, and scores the
    null statistic against THAT permutation's own permuted targets.
    Conservative p = (1 + #at-least-as-favorable) / (1 + n).
    """
    from .bootstrap import (
        METRIC_DIRECTIONS,
        permutation_p_value,
        permute_subject_targets,
    )

    if splits is None:
        from .evaluation import _default_splitter

        splits = list(_default_splitter(len(np.unique(groups))).split(X, y, groups))
    y = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups)
    observed = {
        result.model_name: result.aggregate
        for result in run_nested_cv(X, y, groups_arr, model_names, splits=splits,
                                    seed=seed)
    }
    rng = np.random.default_rng(seed)
    nulls: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metrics} for name in model_names
    }
    for _ in range(n_permutations):
        y_perm = permute_subject_targets(y, groups_arr, rng)
        permuted = run_nested_cv(X, y_perm, groups_arr, model_names,
                                 splits=splits, seed=seed)
        for result in permuted:
            table = result.subject_table
            perm_metrics = _regression_metrics(
                table["y_true"].to_numpy(), table["y_pred"].to_numpy(),
                with_correlations=False,
            )
            for metric in metrics:
                nulls[result.model_name][metric].append(perm_metrics[metric])
    results: dict[str, dict[str, object]] = {}
    for name in model_names:
        results[name] = {}
        for metric in metrics:
            null = np.asarray(nulls[name][metric], dtype=float)
            results[name][metric] = {
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "observed": float(observed[name][metric]),
                "p_value": permutation_p_value(
                    float(observed[name][metric]), null,
                    METRIC_DIRECTIONS[metric], n_permutations,
                ),
                "null_mean": float(np.mean(null)) if null.size else float("nan"),
                "n_permutations": n_permutations,
                "seed": seed,
            }
    return results


def promotion_table(
    block_results: list[SubjectCVResult],
    reference_aggregates: dict[str, dict],
) -> pd.DataFrame:
    """Frozen promotion rule: BOTH MAE and RMSE must improve versus BOTH
    naive baselines AND the frozen five-feature reference — point
    estimates on identical outer folds. Correlation never promotes."""
    baselines = {
        result.model_name: result.aggregate
        for result in block_results
        if result.model_name in ("dummy_mean", "dummy_median")
    }
    rows = []
    for result in block_results:
        if result.model_name in baselines:
            continue
        checks: dict[str, bool] = {}
        for baseline_name, aggregate in baselines.items():
            for metric in ("MAE", "RMSE"):
                checks[f"beats_{baseline_name}_{metric}"] = (
                    aggregate[metric] is not None
                    and result.aggregate[metric] < aggregate[metric]
                )
        for metric in ("MAE", "RMSE"):
            checks[f"beats_reference_{metric}"] = (
                metric in reference_aggregates
                and reference_aggregates[metric] is not None
                and result.aggregate[metric] < reference_aggregates[metric]
            )
        rows.append({
            "model": result.model_name,
            "block": result.block,
            "MAE": result.aggregate["MAE"],
            "RMSE": result.aggregate["RMSE"],
            "R2": result.aggregate["R2"],
            "Pearson_r": result.aggregate["Pearson_r"],
            "Spearman_rho": result.aggregate["Spearman_rho"],
            "promotes": all(checks.values()),
            **checks,
        })
    return pd.DataFrame(rows)


def resting_trait_evidence(
    combined_results: list[SubjectCVResult],
    reference_aggregates: dict[str, dict],
    holm_entries: dict[tuple[str, str], dict],
) -> dict:
    """Frozen §12 resting-trait criterion, evaluated IN CODE: Ridge on the
    combined REST block meets the full promotion rule AND its
    Holm-adjusted permutation p < 0.05 on MAE or RMSE. ``holm_entries``
    is keyed ``("ridge_primary", "MAE" | "RMSE")`` with a
    ``"p_value_holm"`` entry (a not-executed family member carries
    ``None`` and can never count as evidence)."""
    table = promotion_table(combined_results, reference_aggregates)
    ridge_rows = table[table["model"] == "ridge_primary"]
    promotes = bool(len(ridge_rows) and ridge_rows["promotes"].iloc[0])
    p_mae = holm_entries.get(("ridge_primary", "MAE"), {}).get("p_value_holm")
    p_rmse = holm_entries.get(("ridge_primary", "RMSE"), {}).get("p_value_holm")
    significant = (
        (p_mae is not None and p_mae < 0.05)
        or (p_rmse is not None and p_rmse < 0.05)
    )
    return {
        "criterion": (
            "ridge_primary (combined REST block) meets the full promotion "
            "rule AND Holm-adjusted permutation p < 0.05 on MAE or RMSE"
        ),
        "ridge_promotes": promotes,
        "holm_p_MAE": p_mae,
        "holm_p_RMSE": p_rmse,
        "holm_significant": bool(significant),
        "met": bool(promotes and significant),
    }


def state_change_evidence(
    delta_results: list[SubjectCVResult],
    rest_results: list[SubjectCVResult],
    reference_aggregates: dict[str, dict],
) -> dict:
    """Frozen §12 state-change criterion, evaluated IN CODE: the DELTA
    combined block beats the REST combined block on BOTH MAE and RMSE on
    identical outer folds AND meets the promotion rule versus baselines.

    The frozen text names the blocks, not a single model, so every
    prespecified (non-baseline) model is evaluated and evidence exists if
    ANY model satisfies both halves — the reading under which a "none"
    verdict is conservative. Returns the per-model components so the
    verdict is never asserted in prose without its parts."""
    rest = {r.model_name: r.aggregate for r in rest_results}
    delta_promotion = promotion_table(delta_results, reference_aggregates)
    promoted_models = set(
        delta_promotion[delta_promotion["promotes"]]["model"].tolist()
    )
    per_model: list[dict] = []
    for result in delta_results:
        if result.model_name in ("dummy_mean", "dummy_median"):
            continue
        rest_aggregate = rest.get(result.model_name)
        if rest_aggregate is None:
            continue
        beats_mae = (
            result.aggregate["MAE"] is not None
            and rest_aggregate["MAE"] is not None
            and result.aggregate["MAE"] < rest_aggregate["MAE"]
        )
        beats_rmse = (
            result.aggregate["RMSE"] is not None
            and rest_aggregate["RMSE"] is not None
            and result.aggregate["RMSE"] < rest_aggregate["RMSE"]
        )
        per_model.append({
            "model": result.model_name,
            "delta_mae": result.aggregate["MAE"],
            "rest_combined_mae": rest_aggregate["MAE"],
            "delta_rmse": result.aggregate["RMSE"],
            "rest_combined_rmse": rest_aggregate["RMSE"],
            "beats_rest_combined_both_metrics": bool(
                beats_mae and beats_rmse
            ),
            "meets_promotion_rule": result.model_name in promoted_models,
        })
    met = any(row["beats_rest_combined_both_metrics"]
              and row["meets_promotion_rule"] for row in per_model)
    return {
        "criterion": (
            "DELTA combined block beats the REST combined block on BOTH "
            "MAE and RMSE on identical folds AND meets the promotion rule "
            "versus baselines (any prespecified model; conservative for a "
            "'none' verdict)"
        ),
        "per_model": per_model,
        "met": bool(met),
    }

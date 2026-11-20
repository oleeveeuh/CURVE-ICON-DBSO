"""Track B — continuous kinematic decoding beyond temporal history
(v2 contract §3, §8, §9, §11).

Structure (frozen): an autoregressive kinematic baseline (least-squares
AR on the standardized target stream, order 10 bins = 1.0 s, fitted on
OUTER-TRAINING subjects only) captures the target's own temporal
history; a lagged Ridge predicts the RESIDUAL (observed − AR
prediction); the final forecast is AR prediction + predicted residual.
Comparators: neural-only Ridge (same features → target directly),
persistence, and the fixed shallow XGBoost on exactly the same
leakage-safe inputs.

Zero-shot scoring is correlation-only (scale-free): per-recording
target standardization fitted on a final test interval is a frozen
leakage violation, and no held-out-subject labeled data exists
zero-shot — error-magnitude claims are few-shot endpoints. Nulls:
v1.0.1-corrected circular shifts (effective displacement ≥ 2 s within
each MOVE block) and blockwise 2 s segment shifts; conservative
p = (1 + #{null ≥ observed}) / (1 + n).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..errors import DataValidationError
from . import contract

# ---------------------------------------------------------------------------
# Shared estimators
# ---------------------------------------------------------------------------


def make_ridge_pipeline(alpha: float) -> Pipeline:
    """Fold-local preprocessing (median impute + standardize) inside the
    pipeline, fitted on training rows only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def select_alpha_inner(X: pd.DataFrame, y: np.ndarray,
                       groups: np.ndarray) -> float:
    """Inner subject-grouped GroupKFold(3) alpha selection by mean MAE
    on OUTER-TRAINING rows only; ties → smaller frozen grid index."""
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=min(3, n_groups))
    scores: list[float] = []
    for alpha in contract.RIDGE_ALPHAS:
        fold_mae: list[float] = []
        for train_idx, val_idx in splitter.split(X, y, groups):
            pipe = make_ridge_pipeline(alpha)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            fold_mae.append(_mae(y[val_idx], pipe.predict(X.iloc[val_idx])))
        scores.append(float(np.mean(fold_mae)))
    return contract.RIDGE_ALPHAS[int(np.argmin(scores))]


def select_alpha_blocked_calibration(X: pd.DataFrame, y: np.ndarray,
                                     n_blocks: int = 3) -> float:
    """Alpha selection INSIDE a held-out subject's calibration segment:
    contiguous (chronological, non-shuffled) blocked splits — never the
    final test interval. This is a subject-specific parameter fitted
    only from the calibration segment, as the contract allows; the
    zero-shot-selected alpha is intentionally NOT reused (a value
    chosen for a different task and cohort over-shrinks the tiny
    calibration refit)."""
    n = len(y)
    if n < 30:
        return contract.RIDGE_ALPHAS[3]  # mid-grid default for tiny segments
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    scores: list[float] = []
    for alpha in contract.RIDGE_ALPHAS:
        fold_mae: list[float] = []
        for b in range(n_blocks):
            val_lo, val_hi = edges[b], edges[b + 1]
            if val_hi - val_lo < 5 or val_lo < 10:
                continue
            fit_idx = np.concatenate([
                np.arange(0, val_lo), np.arange(val_hi, n)])
            if len(fit_idx) < 10:
                continue
            pipe = make_ridge_pipeline(alpha)
            pipe.fit(X.iloc[fit_idx], y[fit_idx])
            fold_mae.append(_mae(y[val_lo:val_hi],
                                 pipe.predict(X.iloc[val_lo:val_hi])))
        scores.append(float(np.mean(fold_mae)) if fold_mae else np.inf)
    return contract.RIDGE_ALPHAS[int(np.argmin(scores))]


#: The frozen calibration blend options (contract §8, v2.0.1): the
#: calibration segment chooses HOW MUCH to trust the subject-specific
#: correction versus the transferred model — including "pure AR" (the
#: null case; the transferred residual can only hurt when it carries
#: nothing). Ordered most-conservative first; ties resolve to the
#: earlier option.
BLEND_OPTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),   # pure kinematic AR (no neural term)
    (1.0, 0.0),   # transferred zero-shot residual only
    (1.0, 0.5),   # transferred residual + half the calibration correction
    (1.0, 1.0),   # full calibration correction
)


def select_residual_blend_blocked(
    X_calib: pd.DataFrame, resid_target: np.ndarray,
    zs_resid_calib: np.ndarray, n_blocks: int = 3,
) -> tuple[float, float, float, Pipeline]:
    """Blocked-CV selection of ``(s, w)`` — the final residual forecast
    is ``s·(transferred prediction) + w·(calibration correction)`` —
    together with the correction Ridge's alpha and the final correction
    pipeline (refit on the FULL calibration segment at the selected
    alpha). Evaluated ONLY on the calibration segment (chronological
    blocked folds, correction refit inside each fold's training part;
    never the test interval). Returns ``(s, w, alpha, delta_pipe)``."""
    n = len(resid_target)
    base_calib = zs_resid_calib
    delta_target = resid_target - base_calib
    if n < 30:
        alpha = contract.RIDGE_ALPHAS[3]
        delta_pipe = make_ridge_pipeline(alpha)
        delta_pipe.fit(X_calib, delta_target)
        return (0.0, 0.0, alpha, delta_pipe)
    alpha = select_alpha_blocked_calibration(X_calib, delta_target)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    # Per-fold correction predictions (refit inside each fold's training
    # part at the selected alpha — no in-sample leakage into the choice).
    fold_val_preds: list[np.ndarray] = []
    fold_val_targets: list[np.ndarray] = []
    fold_base_vals: list[np.ndarray] = []
    for b in range(n_blocks):
        val_lo, val_hi = edges[b], edges[b + 1]
        if val_hi - val_lo < 5 or val_lo < 10:
            continue
        fit_idx = np.concatenate([
            np.arange(0, val_lo), np.arange(val_hi, n)])
        if len(fit_idx) < 10:
            continue
        pipe = make_ridge_pipeline(alpha)
        pipe.fit(X_calib.iloc[fit_idx], delta_target[fit_idx])
        fold_val_preds.append(np.asarray(
            pipe.predict(X_calib.iloc[val_lo:val_hi]), dtype=float))
        fold_val_targets.append(resid_target[val_lo:val_hi])
        fold_base_vals.append(base_calib[val_lo:val_hi])
    best: tuple[float, float] = BLEND_OPTIONS[0]
    best_mae = np.inf
    for s, w in BLEND_OPTIONS:
        fold_mae = [
            _mae(t, s * b + w * p)
            for t, b, p in zip(fold_val_targets, fold_base_vals,
                               fold_val_preds, strict=True)
        ]
        score = float(np.mean(fold_mae)) if fold_mae else np.inf
        if score < best_mae - 1e-15:
            best_mae = score
            best = (s, w)
    delta_pipe = make_ridge_pipeline(alpha)
    delta_pipe.fit(X_calib, delta_target)
    return (best[0], best[1], alpha, delta_pipe)


# ---------------------------------------------------------------------------
# Kinematic AR baseline
# ---------------------------------------------------------------------------


def fit_ar_on_streams(streams: list[np.ndarray],
                      order: int = contract.AR_ORDER_BINS,
                      block_ids: list[np.ndarray] | None = None) -> dict:
    """Fit ONE AR(order) with intercept across outer-training subjects'
    standardized streams (strictly past-value design rows; each stream
    is standardized by its own training-side constants — documented).

    Histories RESET at block boundaries (v2.1.0): a design row is kept
    only when the entire ``order``-bin history shares the current bin's
    block id, so no fit row ever conditions across a block, purge, or
    REST gap. ``block_ids`` (one array per stream, aligned) supplies the
    ids; ``None`` treats each stream as one segment."""
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for stream_index, stream in enumerate(streams):
        stream = np.asarray(stream, dtype=float)
        ids = (np.zeros(stream.size, dtype=int) if block_ids is None
               else np.asarray(block_ids[stream_index], dtype=int))
        if ids.size != stream.size:
            raise DataValidationError(
                "AR fit block_ids must align with their stream")
        for t in range(order, stream.size):
            if ids[t] != ids[t - order]:
                continue  # history crosses a block boundary — reset
            history = stream[t - order:t][::-1]
            if not np.isfinite(history).all() or not np.isfinite(stream[t]):
                continue
            rows.append(history)
            targets.append(stream[t])
    if not rows:
        raise DataValidationError("no AR rows from training streams")
    design = np.asarray(rows)
    with_intercept = np.column_stack([design, np.ones(design.shape[0])])
    solution, _, _, _ = np.linalg.lstsq(with_intercept, np.asarray(targets),
                                        rcond=None)
    return {"coefficients": solution[:-1], "intercept": float(solution[-1]),
            "order": order, "n_rows": int(design.shape[0])}


def ar_predict(ar_model: dict, stream: np.ndarray,
               block_ids: np.ndarray | None = None) -> np.ndarray:
    """One-step AR predictions (aligned to stream indices; NaN for the
    first ``order`` bins of every segment). Never reads future values,
    and never conditions across a block boundary (v2.1.0): predictions
    at the first ``order`` bins of each new block id stay NaN, and any
    NaN history propagates to NaN."""
    order = int(ar_model["order"])
    stream = np.asarray(stream, dtype=float)
    ids = (np.zeros(stream.size, dtype=int) if block_ids is None
           else np.asarray(block_ids, dtype=int))
    if ids.size != stream.size:
        raise DataValidationError("AR block_ids must align with the stream")
    predictions = np.full(stream.size, np.nan)
    for t in range(order, stream.size):
        if ids[t] != ids[t - order]:
            continue  # history crosses a block boundary — reset
        history = stream[t - order:t][::-1]
        if not np.isfinite(history).all():
            continue
        predictions[t] = float(np.dot(ar_model["coefficients"], history)
                               + ar_model["intercept"])
    return predictions


# ---------------------------------------------------------------------------
# Per-subject condition evaluation
# ---------------------------------------------------------------------------


@dataclass
class SubjectConditionResult:
    held_out_position: int
    condition: str
    n_test_windows: int
    mae: dict[str, float] = field(default_factory=dict)
    r: dict[str, float] = field(default_factory=dict)
    incremental_r2: float | None = None
    innovation_r: float | None = None
    chosen: dict = field(default_factory=dict)
    # Null bookkeeping (test-interval streams; never serialized to docs):
    observed_stream: np.ndarray | None = None
    combined_pred: np.ndarray | None = None
    ar_pred: np.ndarray | None = None
    persistence_pred: np.ndarray | None = None
    neural_residual_pred: np.ndarray | None = None
    block_ids: np.ndarray | None = None


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if valid.sum() < 1:
        return np.nan
    return float(np.mean(np.abs(y[valid] - pred[valid])))


def _persistence_within_blocks(stream: np.ndarray,
                               block_ids: np.ndarray | None) -> np.ndarray:
    """One-bin persistence with histories RESET at block boundaries
    (v2.1.0): the first bin of every block is NaN, never carried from
    the previous block across a REST gap or the purge."""
    stream = np.asarray(stream, dtype=float)
    persistence = np.full(stream.shape, np.nan)
    if block_ids is None:
        persistence[1:] = stream[:-1]
        return persistence
    ids = np.asarray(block_ids)
    if ids.size != stream.size:
        raise DataValidationError("persistence block_ids must align with the stream")
    same = ids[1:] == ids[:-1]
    persistence[1:][same] = stream[:-1][same]
    return persistence


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return None
    a, b = a[valid], b[valid]
    if np.std(a) <= 0 or np.std(b) <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _incremental_r2(y: np.ndarray, base: np.ndarray, enhanced: np.ndarray) -> float:
    ss_base = float(np.sum((y - base) ** 2))
    ss_enh = float(np.sum((y - enhanced) ** 2))
    if ss_base <= 0:
        return np.nan
    return 1.0 - ss_enh / ss_base


def evaluate_fewshot_subject(
    held_out_position: int,
    condition: str,
    X_calib: pd.DataFrame,
    y_calib: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_block_ids: np.ndarray,
    alpha_fixed: float,
    ar_model: dict | None,
    zero_shot_residual_pipe: Pipeline | None = None,
    zero_shot_neural_pipe: Pipeline | None = None,
    with_xgboost: bool = True,
    calib_block_ids: np.ndarray | None = None,
) -> SubjectConditionResult:
    """All frozen models for one held-out subject in a few-shot
    condition. ``y_calib``/``y_test`` are CALIBRATION-STANDARDIZED
    targets (the caller fits the constants on the calibration segment
    only — never the test interval). ``alpha_fixed`` (the OUTER-fold
    inner-selected alpha on training subjects) is recorded for
    provenance. ``test_block_ids``/``calib_block_ids`` carry each row's
    block id: persistence and the AR baseline RESET at every block
    boundary (v2.1.0) — the first bin of each block, and the first
    ``AR_ORDER_BINS`` bins of every AR segment, are NaN, never carried
    across a REST gap or the purge.

    The calibration fit is an ANCHORED FINE-TUNE (v2.0.1 rule): the
    transferred zero-shot residual model produces the base prediction;
    the calibration segment fits only the correction
    ``observed_residual − zero_shot_residual_prediction`` with an alpha
    selected by chronological blocked CV INSIDE the calibration segment.
    A from-scratch refit on ~a few hundred rows against ~440 features
    can only select maximal shrinkage — anchoring to the transferred
    model is the prespecified few-shot mechanism."""
    result = SubjectConditionResult(
        held_out_position=held_out_position, condition=condition,
        n_test_windows=int(y_test.size), chosen={"alpha_zero_shot_selected": alpha_fixed})
    persistence = _persistence_within_blocks(
        np.asarray(y_test, dtype=float), test_block_ids)
    result.persistence_pred = persistence
    valid_p = np.isfinite(persistence)
    result.mae["persistence"] = _mae(y_test[valid_p], persistence[valid_p])
    result.r["persistence"] = safe_corr(y_test[valid_p],
                                        persistence[valid_p]) or np.nan

    if ar_model is None:
        # Machinery failure recorded upstream; persistence-only scoring.
        result.observed_stream = np.asarray(y_test, dtype=float)
        result.block_ids = np.asarray(test_block_ids)
        return result
    ids_calib = (np.zeros(len(y_calib), dtype=int) if calib_block_ids is None
                 else np.asarray(calib_block_ids, dtype=int))
    ids_test = (np.zeros(len(y_test), dtype=int) if test_block_ids is None
                else np.asarray(test_block_ids, dtype=int))
    # Calibration and test are purge-separated spans: offset the test ids
    # so the AR can never bridge the junction even at identical numbers.
    offset = int(ids_calib.max()) + 1 if ids_calib.size else 0
    ar_stream = np.concatenate([np.asarray(y_calib, dtype=float),
                                np.asarray(y_test, dtype=float)])
    ar_full = ar_predict(ar_model, ar_stream,
                         block_ids=np.concatenate([ids_calib, ids_test + offset]))
    ar_pred = ar_full[len(y_calib):]
    valid = np.isfinite(ar_pred)
    if valid.sum() < 2:
        result.observed_stream = np.asarray(y_test, dtype=float)
        result.block_ids = np.asarray(ids_test)
        return result
    result.observed_stream = np.asarray(y_test, dtype=float)
    result.combined_pred = np.full(y_test.shape, np.nan)
    result.ar_pred = ar_pred
    result.block_ids = np.asarray(test_block_ids)
    result.mae["ar_baseline"] = _mae(y_test[valid], ar_pred[valid])
    result.r["ar_baseline"] = safe_corr(y_test[valid], ar_pred[valid]) or np.nan

    ar_calib_full = ar_full[:len(y_calib)]
    resid_valid = np.isfinite(ar_calib_full)
    if resid_valid.sum() < 10:
        result.chosen["reason"] = "calibration_too_short_for_residual_fit"
        result.mae["combined"] = result.mae["ar_baseline"]
        result.r["combined"] = result.r["ar_baseline"]
        return result
    resid_target = np.asarray(y_calib)[resid_valid] - ar_calib_full[resid_valid]
    X_resid = X_calib.iloc[np.flatnonzero(resid_valid)]

    # --- Transferred (zero-shot) residual prediction on calibration rows.
    if zero_shot_residual_pipe is not None:
        zs_resid_calib = np.asarray(
            zero_shot_residual_pipe.predict(X_resid), dtype=float)
        zs_resid_test = np.asarray(
            zero_shot_residual_pipe.predict(X_test), dtype=float)
    else:
        zs_resid_calib = np.zeros(resid_valid.sum())
        zs_resid_test = np.zeros(y_test.size)
    # The zero-shot residual model alone (a genuine zero-shot COMBINED
    # forecast once added to the AR baseline; scored here few-shot).
    zs_combined = ar_pred + zs_resid_test
    valid_z = np.isfinite(zs_combined)
    result.mae["combined_zero_shot_residual"] = _mae(y_test[valid_z],
                                                     zs_combined[valid_z])
    result.r["combined_zero_shot_residual"] = safe_corr(
        y_test[valid_z], zs_combined[valid_z]) or np.nan

    # --- Calibration-selected blend (v2.0.1 rule): the calibration
    # segment chooses between the pure kinematic AR, the transferred
    # residual forecast, and an anchored calibration correction.
    s, w, alpha_calib, delta_pipe = select_residual_blend_blocked(
        X_resid, resid_target, zs_resid_calib)
    result.chosen["blend_transferred_weight"] = s
    result.chosen["blend_correction_weight"] = w
    result.chosen["alpha_calibration_blocked_cv"] = alpha_calib
    delta_pred = np.asarray(delta_pipe.predict(X_test), dtype=float)
    neural_residual = s * zs_resid_test + w * delta_pred
    result.neural_residual_pred = neural_residual  # null bookkeeping
    combined = ar_pred + neural_residual
    result.combined_pred = combined
    valid_c = np.isfinite(combined)
    result.mae["combined"] = _mae(y_test[valid_c], combined[valid_c])
    result.r["combined"] = safe_corr(y_test[valid_c], combined[valid_c]) or np.nan
    obs_residual = np.asarray(y_test)[valid] - ar_pred[valid]
    result.innovation_r = safe_corr(neural_residual[valid], obs_residual) or np.nan
    result.incremental_r2 = _incremental_r2(
        np.asarray(y_test)[valid_c], ar_pred[valid_c], combined[valid_c])

    # --- Neural-only comparator: transferred zero-shot prediction plus
    # its own anchored calibration correction.
    if zero_shot_neural_pipe is not None:
        zs_neural_calib = np.asarray(
            zero_shot_neural_pipe.predict(X_resid), dtype=float)
        zs_neural_test = np.asarray(
            zero_shot_neural_pipe.predict(X_test), dtype=float)
        delta_neural_target = np.asarray(y_calib)[resid_valid] - zs_neural_calib
        alpha_neural = select_alpha_blocked_calibration(
            X_resid, delta_neural_target)
        delta_neural_pipe = make_ridge_pipeline(alpha_neural)
        delta_neural_pipe.fit(X_resid, delta_neural_target)
        neural_pred = zs_neural_test + np.asarray(
            delta_neural_pipe.predict(X_test), dtype=float)
        valid_n = np.isfinite(neural_pred)
        result.mae["neural_only"] = _mae(y_test[valid_n], neural_pred[valid_n])
        result.r["neural_only"] = safe_corr(
            y_test[valid_n], neural_pred[valid_n]) or np.nan

    if with_xgboost:
        try:
            from xgboost import XGBRegressor

            xgb = XGBRegressor(**contract.XGB_FIXED_PARAMS)
            xgb.fit(X_resid, resid_target)
            xgb_resid = np.asarray(xgb.predict(X_test))
            xgb_combined = ar_pred + xgb_resid
            valid_x = np.isfinite(xgb_combined)
            result.mae["xgboost_combined"] = _mae(y_test[valid_x],
                                                  xgb_combined[valid_x])
            result.r["xgboost_combined"] = safe_corr(
                y_test[valid_x], xgb_combined[valid_x]) or np.nan
        except ImportError:
            result.chosen["xgboost"] = "not_executed_optional_dependency_absent"
    return result


def evaluate_zero_shot_subject(
    held_out_position: int,
    X_test: pd.DataFrame,
    y_test_raw: np.ndarray,
    test_block_ids: np.ndarray,
    training_pipe: Pipeline,
) -> SubjectConditionResult:
    """Zero-shot condition: the training-fitted neural-only pipeline
    predicts the held-out subject's movement stream; scoring is
    correlation-only against the RAW observed stream (scale-free). No
    held-out-subject labeled value touches any fit."""
    result = SubjectConditionResult(
        held_out_position=held_out_position, condition="zero_shot",
        n_test_windows=int(len(y_test_raw)))
    y_raw = np.asarray(y_test_raw, dtype=float)
    pred = np.asarray(training_pipe.predict(X_test), dtype=float)
    persistence = _persistence_within_blocks(y_raw, test_block_ids)
    result.persistence_pred = persistence
    valid_p = np.isfinite(persistence)
    result.r["neural_only"] = safe_corr(y_raw, pred) or np.nan
    result.r["persistence"] = safe_corr(y_raw[valid_p],
                                        persistence[valid_p]) or np.nan
    result.observed_stream = y_raw
    result.combined_pred = pred           # the zero-shot neural prediction
    result.block_ids = np.asarray(test_block_ids)
    result.chosen = {"scoring": "correlation_only_scale_free"}
    return result


# ---------------------------------------------------------------------------
# Nulls (v1.0.1-corrected conventions; subject-aware)
# ---------------------------------------------------------------------------


def _effective_displacement(shift: int, span: int) -> int:
    return min(shift, span - shift)


def _shift_within_blocks(stream: np.ndarray, block_ids: np.ndarray,
                         min_bins: int, rng, segmentwise: bool = False,
                         segment_bins: int = 20) -> tuple[np.ndarray, int]:
    """Circular shift of one observed stream within each MOVE block
    (effective displacement ≥ min_bins; too-short blocks excluded and
    counted). ``segmentwise`` shifts 2 s segment blocks instead."""
    shifted = np.full(stream.shape, np.nan)
    excluded = 0
    for block in np.unique(block_ids):
        mask = block_ids == block
        segment = stream[mask]
        span = segment.size
        if span < 2 * (segment_bins if segmentwise else min_bins):
            excluded += 1
            shifted[mask] = segment
            continue
        if segmentwise:
            n_segments = span // segment_bins
            rotation = int(rng.integers(1, max(n_segments, 2)))
            reshaped = segment[: n_segments * segment_bins].reshape(
                n_segments, segment_bins)
            rotated = np.roll(reshaped, rotation, axis=0).ravel()
            out = segment.copy()
            out[: n_segments * segment_bins] = rotated
            shifted[mask] = out
        else:
            lo = max(min_bins, 1)
            hi = span - min_bins
            if hi < lo:
                excluded += 1
                shifted[mask] = segment
                continue
            shift = int(rng.integers(lo, hi + 1))
            if _effective_displacement(shift, span) < min_bins:
                shift = lo
            shifted[mask] = np.roll(segment, shift)
    return shifted, excluded


def _null_summary(observed: float, null_stats: np.ndarray,
                  direction: str) -> dict:
    if direction == "lower":   # error statistics: smaller is favorable
        count = int(np.count_nonzero(null_stats <= observed))
    else:                      # correlation statistics: larger is favorable
        count = int(np.count_nonzero(null_stats >= observed))
    p_raw = (1 + count) / (1 + null_stats.size) if null_stats.size else 1.0
    return {
        "observed": observed,
        "null_median": float(np.median(null_stats)) if null_stats.size else np.nan,
        "count_at_least_as_favorable": count,
        "p_raw": float(p_raw),
        "convention": contract.PERMUTATION_CONVENTION,
    }


def circular_shift_null_delta_mae(
    per_subject: list[SubjectConditionResult],
    n_draws: int = contract.REGRESSION_NULL_DRAWS,
    min_displacement_bins: int = 20,
    seed: int = contract.NULL_SEED,
) -> dict:
    """Null for the paired delta-MAE statistic: each draw shifts the
    OBSERVED stream (effective displacement ≥ 2 s within each MOVE
    block), recomputes combined and AR-baseline MAEs against the shifted
    stream (model outputs unchanged), and takes the across-subject
    median of MAE_combined − MAE_ar."""
    rng = np.random.default_rng(seed)
    null_stats: list[float] = []
    excluded_total = 0
    for _ in range(n_draws):
        deltas: list[float] = []
        for record in per_subject:
            if record.observed_stream is None or record.combined_pred is None \
                    or record.ar_pred is None or record.block_ids is None:
                continue
            shifted, excluded = _shift_within_blocks(
                record.observed_stream, record.block_ids,
                min_displacement_bins, rng)
            excluded_total += excluded
            mae_combined = _mae(shifted, record.combined_pred)
            mae_ar = _mae(shifted, record.ar_pred)
            if np.isfinite(mae_combined) and np.isfinite(mae_ar):
                deltas.append(mae_combined - mae_ar)
        if deltas:
            null_stats.append(float(np.median(deltas)))
    observed_deltas = [
        _mae(r.observed_stream, r.combined_pred) - _mae(r.observed_stream, r.ar_pred)
        for r in per_subject
        if r.observed_stream is not None and r.combined_pred is not None
        and r.ar_pred is not None
    ]
    observed = float(np.median(observed_deltas)) if observed_deltas else np.nan
    out = _null_summary(observed, np.asarray(null_stats), "lower")
    out.update({
        "statistic": "median_delta_mae",
        "n_draws": n_draws, "seed": seed,
        "min_displacement_bins": min_displacement_bins,
        "blocks_excluded_too_short_total": excluded_total,
    })
    return out


def blockwise_shift_null_delta_mae(
    per_subject: list[SubjectConditionResult],
    n_draws: int = contract.REGRESSION_NULL_DRAWS,
    segment_bins: int = 20,
    seed: int = contract.NULL_SEED,
) -> dict:
    """Secondary null: within-block segmentwise (2 s) shifts, same
    displacement guarantee, same statistic."""
    rng = np.random.default_rng(seed)
    null_stats: list[float] = []
    excluded_total = 0
    for _ in range(n_draws):
        deltas: list[float] = []
        for record in per_subject:
            if record.observed_stream is None or record.combined_pred is None \
                    or record.ar_pred is None or record.block_ids is None:
                continue
            shifted, excluded = _shift_within_blocks(
                record.observed_stream, record.block_ids, segment_bins, rng,
                segmentwise=True, segment_bins=segment_bins)
            excluded_total += excluded
            mae_combined = _mae(shifted, record.combined_pred)
            mae_ar = _mae(shifted, record.ar_pred)
            if np.isfinite(mae_combined) and np.isfinite(mae_ar):
                deltas.append(mae_combined - mae_ar)
        if deltas:
            null_stats.append(float(np.median(deltas)))
    observed_deltas = [
        _mae(r.observed_stream, r.combined_pred) - _mae(r.observed_stream, r.ar_pred)
        for r in per_subject
        if r.observed_stream is not None and r.combined_pred is not None
        and r.ar_pred is not None
    ]
    observed = float(np.median(observed_deltas)) if observed_deltas else np.nan
    out = _null_summary(observed, np.asarray(null_stats), "lower")
    out.update({
        "statistic": "median_delta_mae_blockwise",
        "n_draws": n_draws, "seed": seed, "segment_bins": segment_bins,
        "blocks_excluded_too_short_total": excluded_total,
    })
    return out


def _null_predictor(record: SubjectConditionResult, model_key: str) -> np.ndarray | None:
    """The prediction stream a null re-scores for ``model_key`` — the
    named model's OWN predictions (v2.1.0: the persistence null scores
    the persistence stream, not the neural predictions)."""
    if model_key == "persistence":
        return record.persistence_pred
    return record.combined_pred


def _secondary_stats_for_stream(
    shifted: np.ndarray, record: SubjectConditionResult,
) -> dict[str, float] | None:
    """The four secondary statistics of one subject under one (possibly
    shifted) observed stream — model outputs unchanged."""
    needed = (record.combined_pred, record.ar_pred,
              record.neural_residual_pred)
    if record.observed_stream is None or any(v is None for v in needed):
        return None
    combined, ar_pred, resid = needed
    valid = np.isfinite(shifted) & np.isfinite(combined) \
        & np.isfinite(ar_pred) & np.isfinite(resid)
    if valid.sum() < 10:
        return None
    y, comb, ar, res = shifted[valid], combined[valid], ar_pred[valid], resid[valid]
    ss_base = float(np.sum((y - ar) ** 2))
    ss_enh = float(np.sum((y - comb) ** 2))
    incremental = (1.0 - ss_enh / ss_base) if ss_base > 0 else np.nan
    innovation = safe_corr(res, y - ar)
    rmse = float(np.sqrt(np.mean((y - comb) ** 2)))
    delta = float(np.mean(np.abs(y - comb)) - np.mean(np.abs(y - ar)))
    return {"incremental_r2": incremental,
            "innovation_r": innovation if innovation is not None else np.nan,
            "rmse": rmse,
            "per_subject_spread": delta}


def circular_shift_nulls_fewshot_secondary(
    per_subject: list[SubjectConditionResult],
    n_draws: int = contract.REGRESSION_NULL_DRAWS,
    min_displacement_bins: int = 20,
    seed: int = contract.NULL_SEED,
) -> dict:
    """Nulls for the SECONDARY few-shot regression family (contract §13:
    ``incremental_r2``, ``innovation_r``, ``rmse``, ``per_subject_spread``
    — Holm-adjusted WITHIN the family). Each draw shifts the observed
    stream within blocks (predictions unchanged) and recomputes every
    family member; conservative p convention per member, direction-aware
    (higher favorable for the r²/r statistics, lower for rmse/spread).
    Independent row shuffling is never used (the shift machinery
    guarantees it)."""
    from .detection import holm_adjust

    rng = np.random.default_rng(seed)
    directions = {"incremental_r2": "higher", "innovation_r": "higher",
                  "rmse": "lower", "per_subject_spread": "lower"}
    null_draws: dict[str, list[float]] = {name: [] for name in directions}
    excluded_total = 0
    for _ in range(n_draws):
        per_subject_draw: dict[str, list[float]] = {
            name: [] for name in directions}
        for record in per_subject:
            if record.observed_stream is None or record.block_ids is None:
                continue
            shifted, excluded = _shift_within_blocks(
                record.observed_stream, record.block_ids,
                min_displacement_bins, rng)
            excluded_total += excluded
            stats = _secondary_stats_for_stream(shifted, record)
            if stats is None:
                continue
            for name, value in stats.items():
                if np.isfinite(value):
                    per_subject_draw[name].append(value)
        for name, values in per_subject_draw.items():
            if values:
                null_draws[name].append(float(np.median(values)))
    out: dict = {"n_draws": n_draws, "seed": seed,
                 "min_displacement_bins": min_displacement_bins,
                 "family": list(directions),
                 "blocks_excluded_too_short_total": excluded_total,
                 "statistic": "median_secondary_statistic_circular_shift"}
    p_raw: dict[str, float] = {}
    for name, values in null_draws.items():
        arr = np.asarray(values, dtype=float)
        observed_values = [
            _secondary_stats_for_stream(
                np.asarray(r.observed_stream, dtype=float), r)
            for r in per_subject
            if r.observed_stream is not None]
        observed_list = [s[name] for s in observed_values
                         if s is not None and np.isfinite(s.get(name, np.nan))]
        observed = float(np.median(observed_list)) if observed_list else np.nan
        if arr.size and np.isfinite(observed):
            if directions[name] == "higher":
                count = int(np.count_nonzero(arr >= observed))
            else:
                count = int(np.count_nonzero(arr <= observed))
            p_raw[name] = float((1 + count) / (1 + arr.size))
        else:
            p_raw[name] = 1.0
        out[name] = {
            "observed": observed,
            "null_median": float(np.median(arr)) if arr.size else np.nan,
            "count_at_least_as_favorable": count if (arr.size and np.isfinite(observed)) else 0,
            "p_raw": p_raw[name],
            "direction": directions[name],
        }
    out["p_raw"] = p_raw
    out["p_holm_within_family"] = holm_adjust(p_raw)
    return out


def circular_shift_null_median_r(
    per_subject: list[SubjectConditionResult],
    n_draws: int = contract.REGRESSION_NULL_DRAWS,
    min_displacement_bins: int = 20,
    seed: int = contract.NULL_SEED,
    model_key: str = "neural_only",
) -> dict:
    """Zero-shot correlation null: each draw shifts the observed stream
    (predictions unchanged) and recomputes per-subject r of the named
    model; null statistic = across-subject median r."""
    rng = np.random.default_rng(seed)
    null_stats: list[float] = []
    excluded_total = 0
    for _ in range(n_draws):
        rs: list[float] = []
        for record in per_subject:
            if record.observed_stream is None or record.block_ids is None:
                continue
            predictor = _null_predictor(record, model_key)
            if predictor is None:
                continue
            shifted, excluded = _shift_within_blocks(
                record.observed_stream, record.block_ids,
                min_displacement_bins, rng)
            excluded_total += excluded
            r = safe_corr(shifted, predictor)
            if r is not None:
                rs.append(r)
        if rs:
            null_stats.append(float(np.median(rs)))
    observed_rs = [
        r.r.get(model_key) for r in per_subject
        if r.r.get(model_key) is not None and np.isfinite(r.r.get(model_key))
    ]
    observed = float(np.median(observed_rs)) if observed_rs else np.nan
    out = _null_summary(observed, np.asarray(null_stats), "higher")
    out.update({
        "statistic": f"median_r_{model_key}",
        "n_draws": n_draws, "seed": seed,
        "min_displacement_bins": min_displacement_bins,
        "blocks_excluded_too_short_total": excluded_total,
    })
    return out

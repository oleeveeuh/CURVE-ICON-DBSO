"""Track D — event-level biological analysis (v2 contract §10; enabled).

Primary event outcome: log10 inter-tap interval (scale-free,
cross-subject comparable). Secondary family (Holm within family):
inter-tap-interval CV, within-block amplitude decrement, per-tap
amplitude z, peak-velocity z (within-person only).

Two separate claims, never conflated:

- **Association**: two-stage subject-replicated model — per subject the
  within-subject Spearman association between the event outcome and the
  anchor-cell HFA log power at an event-aligned pre-onset window
  (−0.5 s to onset, causal); across subjects, Fisher-z aggregation with
  a subject-cluster bootstrap CI and a within-recording circular-shift
  null. Subject — not tap — is the replication unit; the number of
  taps never increases the independent subject sample size.
- **Prediction**: held-out-subject validation — Ridge on event-aligned
  neural features versus an event-level AR(1) baseline on the outcome
  history, paired per-subject delta-MAE with bootstrap CI and shift
  null (mirrors Track B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..errors import DataValidationError
from ..neural_features import band_mask, welch_psd
from . import contract
from .regression import SubjectConditionResult, _mae, _shift_within_blocks, make_ridge_pipeline
from .spatial import build_grid_map
from .targets import extract_tap_events

#: The frozen event-aligned neural window: HFA log power ending at the
#: tap onset (causal; the same window math as the continuous track).
EVENT_WINDOW_S = contract.WINDOW_S_PRIMARY
EVENT_NEURAL_BAND = "hfa"


def build_event_table(
    glove: np.ndarray, fs: float,
    blocks: list[tuple[int, int, str]],
) -> list[dict]:
    """Flatten per-channel-block event records into one event-level
    table (taps nested within sensor-block, within recording, within
    subject). ITI outcomes are scale-free; amplitude/velocity are kept
    raw for within-recording z only."""
    events = extract_tap_events(glove, fs, blocks)
    rows: list[dict] = []
    for record in events:
        peaks = record["peak_samples"]
        iti = record["iti_s"]
        log_iti = record["log_iti_s"]
        amplitude = record["amplitude_raw"]
        velocity = record["peak_velocity_raw"]
        for i in range(len(peaks) - 1):
            rows.append({
                "block": record["block"],
                "channel": record["channel"],
                "onset_sample": int(peaks[i + 1]),   # the tap whose ITI is measured
                "log_iti_s": float(log_iti[i]),
                "iti_s": float(iti[i]),
                "amplitude_raw": float(amplitude[i + 1]),
                "peak_velocity_raw": float(velocity[i + 1]),
                "n_cycles_block": record["n_cycles"],
            })
    return rows


def event_neural_features(
    strip_ecog: np.ndarray, fs: float, anchor_position: int,
    onset_samples: np.ndarray,
) -> np.ndarray:
    """Anchor-cell HFA log power in the frozen pre-onset window
    [onset − window, onset) for each tap onset (causal). Onsets whose
    window would run off the recording yield NaN and drop downstream."""
    prepared, ok_mask, _ = _prepare(strip_ecog, fs)
    grid = build_grid_map(strip_ecog.shape[0], anchor_position)
    anchor_channel = grid.offset_to_channel.get(contract.ANCHOR_OFFSET)
    if anchor_channel is None:
        raise DataValidationError("anchor offset missing from the grid map")
    if not ok_mask[anchor_channel]:
        raise DataValidationError("anchor channel failed whole-recording QC")
    signal = prepared[anchor_channel]
    window = int(round(EVENT_WINDOW_S * fs))
    values = np.full(np.asarray(onset_samples).shape, np.nan)
    for i, onset in enumerate(np.asarray(onset_samples, dtype=int)):
        start, end = onset - window, onset
        if start < 0:
            continue
        segment = signal[start:end]
        if segment.size != window or not np.isfinite(segment).all():
            continue
        freqs, psd = welch_psd(segment, fs, segment_s=contract.WELCH_SEGMENT_S)
        hfa = _band_log_relative(freqs, psd,
                                 contract.BANDS_HZ[EVENT_NEURAL_BAND])
        if hfa is not None:
            values[i] = hfa
    return values


def _prepare(strip_ecog: np.ndarray, fs: float):
    from .features import prepare_strip

    return prepare_strip(strip_ecog, fs)


def _band_log_relative(freqs, psd, edges: tuple[float, float]) -> float | None:
    active = (freqs >= 4.0) & (freqs <= contract.BAND_TOP_HZ_MAX)
    total = float(np.trapezoid(psd[active], freqs[active])) if active.any() else 0.0
    in_band = band_mask(freqs, edges[0], edges[1], contract.BAND_TOP_HZ_MAX)
    if not in_band.any() or total <= 0:
        return None
    power = float(np.trapezoid(psd[in_band], freqs[in_band]))
    relative = power / total
    return float(np.log10(relative)) if relative > 0 else None


# ---------------------------------------------------------------------------
# Association (two-stage, subject-replicated)
# ---------------------------------------------------------------------------


def fisher_z_aggregate(rhos: np.ndarray) -> float:
    rhos = np.clip(np.asarray(rhos, dtype=float), -0.999999, 0.999999)
    if rhos.size == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(rhos))))


def association_analysis(
    per_recording: list[dict],
    n_resamples: int = contract.BOOTSTRAP_N_RESAMPLES,
    seed: int = contract.BOOTSTRAP_SEED,
    outcome: str = contract.EVENT_PRIMARY_OUTCOME,
) -> dict:
    """Stage-2 aggregate for one event outcome (default: the primary).

    ``per_recording`` entries carry ``spearman_r`` (within-subject
    association between the outcome and event-aligned anchor HFA) and
    ``n_taps``. Across subjects: Fisher-z aggregation, subject-cluster
    bootstrap CI (subjects resampled with replacement; seed 42). The
    stage-1 shift null runs separately (``association_null``).
    """
    rhos = np.asarray([
        r["spearman_r"] for r in per_recording
        if r.get("spearman_r") is not None and np.isfinite(r["spearman_r"])
    ])
    if rhos.size < 3:
        return {"status": "not_executable_insufficient_subjects",
                "n_subjects_with_events": int(rhos.size),
                "outcome": outcome}
    observed = fisher_z_aggregate(rhos)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, rhos.size, size=rhos.size)
        stats[i] = fisher_z_aggregate(rhos[idx])
    ci = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
    return {
        "status": "ok",
        "n_subjects_with_events": int(rhos.size),
        "outcome": outcome,
        "neural_feature": contract.EVENT_NEURAL_FEATURE,
        "observed_fisher_z_r": observed,
        "bootstrap_ci_fisher_z_r": list(ci),
        "n_resamples": n_resamples,
        "seed": seed,
        "subject_unit_rule": contract.EVENT_SUBJECT_UNIT_RULE,
    }


def association_null(
    per_recording_streams: list[dict],
    n_draws: int = contract.REGRESSION_NULL_DRAWS,
    min_displacement_bins: int = 20,
    seed: int = contract.NULL_SEED,
) -> dict:
    """Stage-1 null: shift each recording's neural feature stream
    (blockwise circular), recompute the within-subject Spearman,
    aggregate Fisher-z; alignment broken, autocorrelation preserved.
    ``per_recording_streams`` entries carry ``outcome``, ``feature``,
    ``block_ids``, and ``observed_spearman``."""
    rng = np.random.default_rng(seed)
    null_stats: list[float] = []
    excluded_total = 0
    for _ in range(n_draws):
        rhos: list[float] = []
        for record in per_recording_streams:
            outcome = np.asarray(record["outcome"], dtype=float)
            feature = np.asarray(record["feature"], dtype=float)
            block_ids = np.asarray(record["block_ids"])
            shifted, excluded = _shift_within_blocks(
                feature, block_ids, min_displacement_bins, rng)
            excluded_total += excluded
            valid = np.isfinite(shifted) & np.isfinite(outcome)
            if valid.sum() >= 10:
                rho = spearmanr(outcome[valid], shifted[valid]).statistic
                if np.isfinite(rho):
                    rhos.append(float(rho))
        if rhos:
            null_stats.append(fisher_z_aggregate(np.asarray(rhos)))
    observed_rhos = np.asarray([
        float(r["observed_spearman"]) for r in per_recording_streams
        if np.isfinite(r.get("observed_spearman", np.nan))
    ])
    observed = fisher_z_aggregate(observed_rhos) if observed_rhos.size else np.nan
    arr = np.asarray(null_stats)
    count = int(np.count_nonzero(arr >= observed)) if arr.size else 0
    p_raw = (1 + count) / (1 + arr.size) if arr.size else 1.0
    return {
        "statistic": "fisher_z_r_association",
        "n_draws": n_draws, "seed": seed,
        "blocks_excluded_too_short_total": excluded_total,
        "observed": observed,
        "null_median": float(np.median(arr)) if arr.size else np.nan,
        "count_at_least_as_favorable": count,
        "p_raw": float(p_raw),
        "convention": contract.PERMUTATION_CONVENTION,
    }


# ---------------------------------------------------------------------------
# Prediction (held-out-subject)
# ---------------------------------------------------------------------------

#: Frozen default alpha for the event-level Ridge (mid-grid; the event
#: model carries no inner tuning grid — one prespecified value).
EVENT_RIDGE_ALPHA = 1.0


def event_prediction_loso(
    per_recording_events: list[pd.DataFrame],
    n_draws: int = contract.REGRESSION_NULL_DRAWS,
    seed: int = contract.NULL_SEED,
) -> dict:
    """Held-out-subject prediction of log-ITI from event-aligned anchor
    HFA versus an event-level AR(1) baseline (previous log-ITI predicts
    the next, RESET at every block×channel boundary — v2.1.0; taps from
    different sensor channels are distinct series). Paired per-subject
    delta-MAE with the subject-cluster bootstrap CI (the frozen
    criterion requires it) and the shift null."""
    deltas: list[float] = []
    per_subject: list[SubjectConditionResult] = []
    n = len(per_recording_events)
    for held_out in range(n):
        test = per_recording_events[held_out]
        train = pd.concat([e for i, e in enumerate(per_recording_events)
                           if i != held_out], ignore_index=True)
        features = ["anchor_hfa_log_power"]
        fit = train[features + ["log_iti_s"]].dropna()
        if len(fit) < 20:
            continue
        pipe = make_ridge_pipeline(EVENT_RIDGE_ALPHA)
        pipe.fit(fit[features], fit["log_iti_s"])
        test_fit = test.dropna(subset=features + ["log_iti_s"]).reset_index(drop=True)
        if len(test_fit) < 10:
            continue
        pred = np.asarray(pipe.predict(test_fit[features]), dtype=float)
        obs = np.asarray(test_fit["log_iti_s"], dtype=float)
        # AR(1) within block×channel series: the previous tap of the SAME
        # sensor-block predicts the next; series starts are NaN.
        series_ids = (test_fit["block"].astype(int) * 1000
                      + test_fit["channel"].astype(int)).to_numpy()
        ar1 = np.full(obs.shape, np.nan)
        for i in range(1, obs.size):
            if series_ids[i] == series_ids[i - 1]:
                ar1[i] = obs[i - 1]
        record = SubjectConditionResult(
            held_out_position=held_out, condition="event_prediction",
            n_test_windows=int(obs.size))
        record.observed_stream = obs
        record.combined_pred = pred
        record.ar_pred = ar1
        record.block_ids = np.asarray(series_ids, dtype=int)
        per_subject.append(record)
        valid_ar = np.isfinite(ar1)
        if np.isfinite(_mae(obs, pred)) and valid_ar.sum() >= 2 \
                and np.isfinite(_mae(obs[valid_ar], ar1[valid_ar])):
            deltas.append(_mae(obs, pred) - _mae(obs[valid_ar], ar1[valid_ar]))
    if not deltas or len(per_subject) < 3:
        return {"status": "not_executable_insufficient_subjects",
                "n_subjects": len(deltas)}
    observed = float(np.median(deltas))
    # Subject-cluster bootstrap CI of the median delta-MAE (frozen
    # criterion: success requires the CI to exclude 0).
    from .detection import _median_bootstrap_ci

    ci = _median_bootstrap_ci(np.asarray(deltas, dtype=float))
    rng = np.random.default_rng(seed)
    null_stats: list[float] = []
    for _ in range(n_draws):
        draw_deltas: list[float] = []
        for record in per_subject:
            shifted, _excluded = _shift_within_blocks(
                record.observed_stream, record.block_ids, 2, rng)
            mae_neural = _mae(shifted, record.combined_pred)
            valid_ar = np.isfinite(record.ar_pred)
            mae_ar = _mae(shifted[valid_ar], record.ar_pred[valid_ar])
            draw_deltas.append(mae_neural - mae_ar)
        null_stats.append(float(np.median(draw_deltas)))
    arr = np.asarray(null_stats)
    count = int(np.count_nonzero(arr <= observed))
    return {
        "status": "ok",
        "n_subjects": len(deltas),
        "observed_median_delta_mae": observed,
        "bootstrap_ci_median_delta_mae": list(ci),
        "median_mae_neural": float(np.median([
            _mae(r.observed_stream, r.combined_pred) for r in per_subject])),
        "median_mae_ar1": float(np.median([
            _mae(r.observed_stream[np.isfinite(r.ar_pred)],
                 r.ar_pred[np.isfinite(r.ar_pred)]) for r in per_subject])),
        "null": {
            "statistic": "median_event_delta_mae",
            "n_draws": n_draws, "seed": seed,
            "null_median": float(np.median(arr)),
            "count_at_least_as_favorable": count,
            "p_raw": float((1 + count) / (1 + arr.size)),
        },
    }

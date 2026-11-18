"""Time-resolved MOVE methodology-transfer study (frozen contract v1.0.1,
Prompt 4): windowed ECoG features aligned with the continuous glove vigor
envelope, three prespecified decoders, subject-held-out evaluation with
per-subject-first statistics, autocorrelation-preserving nulls, and
sanitized aggregate outputs.

Frozen design (contract §6, §8, §10):

* **Windows**: 500 ms neural windows ending AT the target sample
  (exclusive end — the window never contains the target instant or
  anything after it), 100 ms update, preceding context at lag offsets
  {0, 0.1, 0.2, 0.3, 0.4, 0.5} s back from the target. Windows never
  cross REST/MOVE block boundaries; QC-invalid windows are counted with
  machine-readable reasons (never imputed, never repaired).
* **Target**: the frozen ``glove_vigor_envelope_z`` stream
  (``glove_phenotyping.glove_move_envelope``), one value per 100 ms
  update bin per MOVE block; the target time is the bin CENTER.
* **Features per window per lag**: log-relative band power over the
  active bands plus the aperiodic exponent, computed per QC-OK channel
  and aggregated by the frozen rule (median across OK channels).
* **Band-block ablation** (prespecified subsets): ``multiband_power``
  (all active bands), ``band_low`` (theta+alpha), ``band_beta``,
  ``band_gamma``. The DMD representation comparison is Prompt 5's
  separate module; a REST-versus-MOVE detection endpoint is NOT part of
  the frozen contract and is deliberately deferred (post-hoc guard).
* **Decoders**: lagged Ridge / Wiener cascade (primary, frozen 7-point
  alpha grid), linear SVR (frozen C grid), and the fixed shallow XGBoost
  (optional DD-003 comparator, skipped cleanly when absent). Baselines:
  fold-global mean, the held-out recording's own block-mean target
  (naive-target reference, labeled as such — never described as a
  model), and persistence (one-update lag).
* **Evaluation**: outer subject-held-out folds — every window of a
  subject stays in exactly one fold (grouped splitter; group integrity
  checked; subject-straddling folds raise outright). Inner subject-
  grouped selection (GroupKFold(3), inner-LOSO under 6 subjects, inner-CV
  mean MAE, frozen tie rule) sees outer-training windows only. Metrics
  are computed per held-out subject FIRST (window-level Pearson r and
  MAE), then summarized as median and IQR across subjects; pooled-window
  values are secondary descriptives carrying the non-independence
  caveat — thousands of windows are not thousands of participants.
* **Nulls**: circular shifts of the glove target WITHIN each recording
  block (features unchanged), each shift's EFFECTIVE circular
  displacement ``min(shift, span − shift)`` ≥ ``protocol.
  MOVE_CIRCULAR_SHIFT_MIN_S`` (≥ 20 update bins in both directions — a
  nominal shift near ``span`` is circularly a tiny shift in the opposite
  direction and is never sampled); blocks shorter than twice the minimum
  displacement are excluded and counted, never silently shifted.
  Retraining the primary decoder per shift at its frozen selection;
  blockwise (≥ 2 s segments, same effective-displacement guarantee)
  shifts as the secondary null. Independent row shuffling is NEVER used
  anywhere in this module.
* **Outputs**: sanitized aggregates and plots — per-subject metric
  distributions (subject positions, no identifiers), observed-vs-null,
  lag ablation, band-block ablation. Per-subject predictions and raw
  metadata stay in gitignored local outputs.

Verification: deterministic synthetic fixtures only (known-lag coupling,
off-by-one traps, block-boundary traps, fingerprint leakage contrast).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol
from .errors import DataValidationError
from .kinematics import parse_restmove_blocks
from .neural_features import spectral_features, welch_psd
from .signal_qc import (
    active_top_edge,
    common_median_reference,
    detect_line_frequency,
    notch_line_noise,
    qc_channel_block,
    recording_diff_mad,
    validate_fs,
)

# Frozen lag-stack subsets for the lag ablation (contract §6 stack).
LAG_ABLATIONS: dict[str, tuple[float, ...]] = {
    "full_stack": protocol.MOVE_LAGS_S,
    "current_window_only": (0.0,),
    "short_context": (0.0, 0.1, 0.2),
}

#: Prespecified band-block ablation subsets (§8 move ablation; DMD is a
#: separate Prompt-5 module and is intentionally absent here).
MOVE_FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "multiband_power": (
        "theta", "alpha", "beta_low", "beta_high", "gamma_low", "gamma_high",
    ),
    "band_low": ("theta", "alpha"),
    "band_beta": ("beta_low", "beta_high"),
    "band_gamma": ("gamma_low", "gamma_high"),
}


# ---------------------------------------------------------------------------
# Windowed dataset builder
# ---------------------------------------------------------------------------


@dataclass
class WindowDataset:
    """The aligned window×feature matrix with position-labeled metadata.

    ``features`` rows are windows; ``target[i]`` is the frozen envelope
    value at window i's target time; ``groups[i]`` is the SUBJECT
    POSITION label supplied by the caller (never a participant
    identifier — callers pass synthetic positions); ``metadata`` keeps
    subject/recording/block/timestamp locally (gitignored outputs only).
    """

    features: pd.DataFrame
    target: np.ndarray
    groups: np.ndarray
    metadata: pd.DataFrame
    qc: dict = field(default_factory=dict)
    windows: list | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def block_subset(self, block: str) -> pd.DataFrame:
        """Column subset for one prespecified band-block ablation
        (columns exist only for bands active at this cohort's fs)."""
        if block not in MOVE_FEATURE_BLOCKS:
            raise DataValidationError(
                f"unknown MOVE feature block '{block}'; prespecified: "
                f"{sorted(MOVE_FEATURE_BLOCKS)}"
            )
        wanted = MOVE_FEATURE_BLOCKS[block]
        columns = [c for c in self.features.columns
                   if any(c.startswith(f"{band}_") for band in wanted)
                   and not c.startswith("aperiodic")]
        if not columns:
            raise DataValidationError(
                f"band block '{block}' has no active columns for this cohort"
            )
        return self.features[columns]

    def lag_subset(self, columns: pd.DataFrame,
                   lags: tuple[float, ...]) -> pd.DataFrame:
        """Restrict a block's columns to the given lag-stack subset."""
        suffixes = tuple(f"{lag:.1f}s" for lag in lags)
        keep = [c for c in columns.columns if c.endswith(suffixes)]
        if not keep:
            raise DataValidationError(
                f"lag subset {lags} matches no feature columns"
            )
        return columns[keep]


def _window_index_math(target_bin: int, update_s: float, window_s: float,
                       fs: float, lag_s: float) -> tuple[int, int]:
    """Causal window indexing — THE off-by-one contract.

    The target bin's time is its CENTER ``(target_bin + 0.5)·update_s``.
    A lagged window ends at ``target_time − lag_s`` (sample EXCLUSIVE):
    it covers ``[end − round(window_s·fs), end)`` and therefore never
    contains the target instant or any sample after it. Returns
    ``(start, end)`` sample indices relative to the block start.
    """
    target_time_s = (target_bin + 0.5) * update_s
    end_sample = int(round((target_time_s - lag_s) * fs))
    start_sample = end_sample - int(round(window_s * fs))
    return start_sample, end_sample


def _block_window_features(
    block_signal: np.ndarray, fs: float, bands: dict[str, tuple[float, float]],
    top_edge: float, window_s: float, update_s: float, lags: tuple[float, ...],
    diff_mad: float, n_bins: int, ok_channels: np.ndarray,
    invalid_intervals_block: list[tuple[int, int]],
    capture_windows: bool = False,
) -> tuple[list[dict], list[dict], list | None]:
    """Feature rows for every target bin of one recording's MOVE block
    (features per OK channel, frozen median aggregation), QC records for
    every dropped window, and — when ``capture_windows`` — the per-window
    OK-channel lag-0 sample matrices (consumed by the DMD representation;
    never stored by default)."""
    rows: list[dict] = []
    qc_records: list[dict] = []
    kept_windows: list | None = [] if capture_windows else None

    def overlaps_invalid(start: int, end: int) -> bool:
        return any(start < b and end > a for a, b in invalid_intervals_block)

    for target_bin in range(n_bins):
        start0, end0 = _window_index_math(target_bin, update_s, window_s, fs, 0.0)
        if start0 < 0 or end0 > block_signal.shape[1]:
            qc_records.append({"target_bin": target_bin, "kept": False,
                               "reasons": ["window_outside_block"]})
            continue
        if overlaps_invalid(start0, end0):
            qc_records.append({"target_bin": target_bin, "kept": False,
                               "reasons": ["invalid_interval"]})
            continue
        block_median = np.median(block_signal[:, start0:end0], axis=0)
        qc = qc_channel_block(block_median, diff_mad)
        if qc["qc_status"] != "ok":
            qc_records.append({"target_bin": target_bin, "kept": False,
                               "reasons": [qc["qc_reason"].split(" (")[0]]})
            continue
        per_channel_rows: list[dict] = []
        for channel in np.flatnonzero(ok_channels):
            channel_signal = block_signal[channel]
            per_lag: dict[str, float] = {}
            ok = True
            for lag in lags:
                shift = int(round(lag * fs))
                start, end = start0 - shift, end0 - shift
                if start < 0 or end <= start:
                    ok = False
                    break
                freqs, psd = welch_psd(channel_signal[start:end], fs,
                                       segment_s=window_s)
                features = spectral_features(freqs, psd,
                                             {"bands_active": bands}, top_edge)
                for band in bands:
                    value = features[f"{band}_log_relative_power"]
                    if value is None:
                        ok = False
                        break
                    per_lag[f"{band}_{lag:.1f}s"] = value
                if not ok:
                    break
                per_lag[f"aperiodic_exponent_{lag:.1f}s"] = features[
                    "aperiodic_exponent"
                ]
            if ok:
                per_channel_rows.append(per_lag)
        if len(per_channel_rows) < max(1, int(ok_channels.sum()) // 2):
            qc_records.append({"target_bin": target_bin, "kept": False,
                               "reasons": ["insufficient_ok_channels"]})
            continue
        row: dict = {
            key: float(np.median([r[key] for r in per_channel_rows]))
            for key in per_channel_rows[0]
        }
        row["target_bin"] = target_bin
        rows.append(row)
        if kept_windows is not None:
            kept_windows.append(
                block_signal[np.flatnonzero(ok_channels)][:, start0:end0].copy()
            )
        qc_records.append({"target_bin": target_bin, "kept": True,
                           "reasons": []})
    return rows, qc_records, kept_windows


def build_window_dataset(
    ecog: np.ndarray,
    fs: float,
    restmove_segs: np.ndarray,
    envelope: dict,
    subject_position: str,
    recording_position: str,
    bands_resolved: dict,
    window_s: float = protocol.MOVE_WINDOW_S,
    update_s: float = protocol.MOVE_UPDATE_S,
    lags: tuple[float, ...] = protocol.MOVE_LAGS_S,
    invalid_intervals: list[tuple[int, int]] | None = None,
    capture_windows: bool = False,
) -> WindowDataset:
    """Align neural windows with the frozen glove envelope for one
    recording.

    ``ecog`` is ``(n_channels, n_samples)`` on the shared block grid;
    ``envelope`` is the ``glove_move_envelope`` result. Windows never
    cross block boundaries; ``invalid_intervals`` (recording-level sample
    ranges) drop windows whose lag-0 window overlaps them. Metadata
    keeps positions and timestamps only.

    Raises:
        DataValidationError: On shape mismatches, an envelope block count
            that disagrees with restmove_segs, a fully non-finite
            recording, or fs below the active band ceiling.
    """
    ecog = np.asarray(ecog, dtype=float)
    if ecog.ndim != 2:
        raise DataValidationError(f"ecog must be 2-D; got {ecog.shape}")
    if envelope.get("stream") is None or envelope.get("block_ids") is None:
        raise DataValidationError(
            "envelope stream unavailable (QC failed upstream); refusing to "
            "build windows without the frozen target"
        )
    top_edge = active_top_edge(bands_resolved)
    validate_fs(fs, top_edge)
    bands = {
        **bands_resolved["bands_active"],
        **({"gamma_high": tuple(bands_resolved["gamma_high"])}
           if bands_resolved.get("gamma_high") else {}),
    }
    move_bounds = [
        (int(s), int(e)) for s, e, kind in parse_restmove_blocks(restmove_segs)
        if kind == "move"
    ]
    block_ids = np.asarray(envelope["block_ids"])
    if len(set(block_ids.tolist())) != len(move_bounds):
        raise DataValidationError(
            f"envelope has {len(set(block_ids.tolist()))} blocks but "
            f"restmove_segs has {len(move_bounds)} MOVE blocks"
        )
    ok_mask = np.isfinite(ecog).all(axis=1)
    if not ok_mask.any():
        raise DataValidationError("every channel is non-finite")
    rereferenced = (
        ecog - np.median(ecog, axis=0) if ok_mask.all()
        else common_median_reference(ecog, ok_mask)
    )
    line = detect_line_frequency(np.mean(rereferenced[ok_mask], axis=0), fs)
    if line["line_freq_hz"] is not None:
        rereferenced = np.vstack([
            notch_line_noise(ch, fs, line["line_freq_hz"]) if ok else ch
            for ch, ok in zip(rereferenced, ok_mask, strict=True)
        ])
    diff_mad = recording_diff_mad(rereferenced[ok_mask])
    invalid = list(invalid_intervals or [])

    stream = np.asarray(envelope["stream"], dtype=float)
    frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    windows_all: list = []
    qc_summary: dict = {"dropped_windows": 0, "reasons": {}}
    for block_number, (start, end) in enumerate(move_bounds):
        end = min(end, ecog.shape[1])
        n_bins = int(np.count_nonzero(block_ids == block_number))
        if n_bins == 0:
            continue
        rows, qc_records, kept_windows = _block_window_features(
            rereferenced[:, start:end], fs, bands, top_edge, window_s,
            update_s, lags, diff_mad, n_bins, ok_mask,
            [(a - start, b - start) for a, b in invalid],
            capture_windows=capture_windows,
        )
        for record in qc_records:
            if not record["kept"]:
                qc_summary["dropped_windows"] += 1
                for reason in record["reasons"]:
                    qc_summary["reasons"][reason] = (
                        qc_summary["reasons"].get(reason, 0) + 1
                    )
        for row_index, row in enumerate(rows):
            target_bin = row.pop("target_bin")
            frames.append(pd.DataFrame([row]))
            if capture_windows and kept_windows is not None:
                windows_all.append(kept_windows[row_index])
            meta_rows.append({
                "subject_position": subject_position,
                "recording_position": recording_position,
                "block": block_number,
                "target_bin": target_bin,
                "target_time_s": (target_bin + 0.5) * update_s,
                "sample_in_recording": start + int(
                    round((target_bin + 0.5) * update_s * fs)
                ),
            })
    if not frames:
        raise DataValidationError(
            "no usable windows survived block-boundary and QC filtering"
        )
    features = pd.concat(frames, ignore_index=True).astype(float)
    target = _targets_for(meta_rows, block_ids, stream)
    groups = np.asarray([m["subject_position"] for m in meta_rows])
    return WindowDataset(
        features=features, target=target, groups=groups,
        metadata=pd.DataFrame(meta_rows), qc=qc_summary,
        windows=windows_all if capture_windows else None,
    )


def _targets_for(meta_rows: list[dict], block_ids: np.ndarray,
                 stream: np.ndarray) -> np.ndarray:
    """Envelope value for each kept window: the stream is concatenated
    over MOVE blocks in order, so block b's bins start at the cumulative
    bin count of the blocks before b."""
    starts: dict[int, int] = {}
    cursor = 0
    for block_number in sorted(set(block_ids.tolist())):
        starts[block_number] = cursor
        cursor += int(np.count_nonzero(block_ids == block_number))
    indices = [starts[m["block"]] + m["target_bin"] for m in meta_rows]
    if max(indices) >= stream.size:
        raise DataValidationError("envelope stream shorter than window grid")
    return stream[np.asarray(indices, dtype=int)]


# ---------------------------------------------------------------------------
# Decoders and evaluation
# ---------------------------------------------------------------------------

MOVE_MODELS: tuple[str, ...] = (
    "lagged_ridge_primary", "svr_linear_move", "xgboost_shallow_move",
)


def _grid_for_move(model_name: str) -> list[dict]:
    spec = protocol.MODELS_MOVE.get(model_name)
    if spec is None:
        raise DataValidationError(
            f"unknown MOVE model '{model_name}'; frozen: "
            f"{sorted(protocol.MODELS_MOVE)}"
        )
    grid = spec.get("grid", {})
    if not grid:
        return [{}]
    combos = [{}]
    for param, values in grid.items():
        combos = [dict(combo, **{param: value}) for combo in combos
                  for value in values]
    return combos


def _build_move_estimator(model_name: str, params: dict, seed: int = 42):
    if model_name == "lagged_ridge_primary":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(params["alpha"]), random_state=seed)
    if model_name == "svr_linear_move":
        from sklearn.svm import SVR

        return SVR(kernel="linear", C=float(params["C"]))
    if model_name == "xgboost_shallow_move":
        from .evaluation import xgboost_model

        return xgboost_model()
    raise DataValidationError(f"unknown MOVE model '{model_name}'")


@dataclass
class MoveCVResult:
    """Subject-held-out MOVE decoding results for one (block, model).

    ``per_subject`` is the first-class table: one row per held-out
    subject position with window-level Pearson r and MAE. ``summary``
    carries median/IQR across subjects plus the pooled-window secondary
    (explicitly flagged non-independent). ``baselines`` holds the three
    frozen references on the identical folds.
    """

    model_name: str
    block: str
    per_subject: pd.DataFrame
    summary: dict
    baselines: dict = field(default_factory=dict)
    selected_params: list = field(default_factory=list)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _select_move_params(model_name: str, X: pd.DataFrame, y: np.ndarray,
                        groups: np.ndarray, seed: int = 42) -> dict:
    """Inner subject-grouped selection on outer-training windows only
    (inner criterion: mean MAE; ties → smaller frozen grid index)."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.pipeline import Pipeline

    from .evaluation import make_preprocessing

    combos = _grid_for_move(model_name)
    if combos == [{}]:
        return {}
    n_subjects = len(np.unique(groups))
    splitter = (
        LeaveOneGroupOut()
        if n_subjects < protocol.INNER_LOSO_MIN_OUTER_TRAIN_SUBJECTS
        else GroupKFold(n_splits=protocol.INNER_SPLITS)
    )
    splits = list(splitter.split(X, y, groups))
    best_params, best_mae = combos[0], float("inf")
    for params in combos:
        maes = []
        for train_idx, val_idx in splits:
            if val_idx.size == 0:
                continue
            steps = [(name, clone(step)) for name, step in make_preprocessing()]
            pipe = Pipeline([*steps, ("model", _build_move_estimator(
                model_name, params, seed=seed))])
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[val_idx])
            maes.append(float(np.mean(np.abs(pred - y[val_idx]))))
        if maes and float(np.mean(maes)) < best_mae:
            best_mae = float(np.mean(maes))
            best_params = params
    return best_params


def run_move_cv(
    dataset: WindowDataset,
    model_names: tuple[str, ...] = MOVE_MODELS,
    block: str = "multiband_power",
    lags: tuple[float, ...] = protocol.MOVE_LAGS_S,
    seed: int = protocol.PERMUTATION_SEED,
    feature_frame: pd.DataFrame | None = None,
) -> list[MoveCVResult]:
    """Subject-held-out decoding over one prespecified band block and a
    lag-stack subset, with the three frozen baselines on the identical
    folds. Outer folds are computed once here from the subject positions,
    so every call with the same dataset shares identical folds.

    ``feature_frame`` (used by the DMD representation comparison,
    ``dmd_representation``) replaces the band-block column selection with
    an explicit, caller-built feature matrix aligned row-for-row with the
    dataset — the folds, selection, and metrics are untouched."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from .evaluation import _check_group_integrity, _default_splitter, make_preprocessing

    X = (
        dataset.lag_subset(dataset.block_subset(block), lags)
        if feature_frame is None
        else feature_frame.reset_index(drop=True).astype(float)
    )
    y = dataset.target
    groups = dataset.groups
    n_subjects = len(np.unique(groups))
    splits = list(_default_splitter(n_subjects).split(X, y, groups))
    _check_group_integrity(groups, splits)

    results: list[MoveCVResult] = []
    for model_name in model_names:
        if model_name == "xgboost_shallow_move":
            try:
                _build_move_estimator(model_name, {})
            except ImportError:
                continue  # optional DD-003 comparator, skipped cleanly
        oof = np.full(y.size, np.nan)
        selected: list = []
        for _fold, (train_idx, test_idx) in enumerate(splits):
            params = _select_move_params(model_name, X.iloc[train_idx],
                                         y[train_idx], groups[train_idx],
                                         seed=seed)
            selected.append(dict(params))
            steps = [(name, clone(step)) for name, step in make_preprocessing()]
            pipe = Pipeline([*steps, ("model", _build_move_estimator(
                model_name, params, seed=seed))])
            pipe.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict(X.iloc[test_idx])
        if np.isnan(oof).any():
            raise DataValidationError("splitter left windows unpredicted")
        per_subject_rows = []
        for subject in sorted(set(groups.tolist())):
            mask = groups == subject
            per_subject_rows.append({
                "subject_position": subject,
                "n_windows": int(mask.sum()),
                "pearson_r": _pearson(y[mask], oof[mask]),
                "mae": float(np.mean(np.abs(oof[mask] - y[mask]))),
            })
        per_subject = pd.DataFrame(per_subject_rows)
        valid_r = per_subject["pearson_r"].dropna().to_numpy()
        summary = {
            "median_pearson_r": float(np.median(valid_r)) if valid_r.size else None,
            "iqr_pearson_r": (
                float(np.percentile(valid_r, 75) - np.percentile(valid_r, 25))
                if valid_r.size >= 4 else None
            ),
            "median_mae": float(np.median(per_subject["mae"])),
            "iqr_mae": float(
                np.percentile(per_subject["mae"], 75)
                - np.percentile(per_subject["mae"], 25)
            ),
            "pooled_mae_secondary_non_independent": float(
                np.mean(np.abs(oof - y))
            ),
            "n_subjects": int(n_subjects),
            "n_windows": int(y.size),
        }
        baseline_rows: dict[str, list[float]] = {
            name: [] for name in ("fold_global_mean", "block_mean_naive",
                                  "persistence")
        }
        baseline_mae_rows: dict[str, list[float]] = {
            name: [] for name in baseline_rows
        }
        metadata = dataset.metadata
        for train_idx, test_idx in splits:
            preds = _baselines_for_fold(test_idx, y, metadata, train_idx)
            for name, prediction in preds.items():
                for subject in sorted(set(groups[test_idx].tolist())):
                    mask = groups[test_idx] == subject
                    r = _pearson(y[test_idx][mask], prediction[mask])
                    baseline_rows[name].append(r if r is not None else 0.0)
                    baseline_mae_rows[name].append(float(np.mean(
                        np.abs(prediction[mask] - y[test_idx][mask])
                    )))
        baselines = {
            name: {
                "median_pearson_r": float(np.median(values)) if values else None,
                "median_mae": (
                    float(np.median(baseline_mae_rows[name]))
                    if baseline_mae_rows[name] else None
                ),
                "iqr_mae": (
                    float(np.percentile(baseline_mae_rows[name], 75)
                          - np.percentile(baseline_mae_rows[name], 25))
                    if len(baseline_mae_rows[name]) >= 4 else None
                ),
                "definition": {
                    "fold_global_mean":
                        "training subjects' mean target (fold-local)",
                    "block_mean_naive": (
                        "held-out recording's own per-block target mean — "
                        "naive-target reference, never a model claim"
                    ),
                    "persistence": "target at t − 0.1 s (one-update lag)",
                }[name],
            }
            for name, values in baseline_rows.items()
        }
        results.append(MoveCVResult(
            model_name=model_name, block=block, per_subject=per_subject,
            summary=summary, baselines=baselines, selected_params=selected,
        ))
    return results


def _baselines_for_fold(test_idx: np.ndarray, y: np.ndarray,
                        metadata: pd.DataFrame,
                        train_idx: np.ndarray) -> dict[str, np.ndarray]:
    """The three frozen naive references for one outer fold."""
    fold_global_mean = np.full(test_idx.size, float(np.mean(y[train_idx])))
    block_mean_naive = np.empty(test_idx.size)
    persistence = np.empty(test_idx.size)
    for position, row_index in enumerate(test_idx):
        row = metadata.loc[row_index]
        same = metadata[
            (metadata["block"] == row["block"])
            & (metadata["recording_position"] == row["recording_position"])
        ]
        block_mean_naive[position] = float(
            np.mean(y[same.index.to_numpy()])
        )
        previous = same[same["target_bin"] == row["target_bin"] - 1]
        persistence[position] = (
            y[previous.index[0]] if previous.index.size
            else fold_global_mean[position]
        )
    return {
        "fold_global_mean": fold_global_mean,
        "block_mean_naive": block_mean_naive,
        "persistence": persistence,
    }


def move_transfer_criterion(result: MoveCVResult) -> dict:
    """Evaluate the frozen §12 movement-state-transfer criterion IN CODE
    from one (primary-model) result: (a) median across-subject Pearson
    r > 0 (the accompanying subject-cluster bootstrap CI is computed
    separately by :func:`bootstrap_median_r` and supplied by the caller),
    and (b) median across-subject MAE better (lower) than BOTH the
    persistence and block-mean baselines on the identical folds.

    Returns every component alongside the Boolean so the verdict is
    never asserted in prose without its parts."""
    baselines = result.baselines
    median_r = result.summary.get("median_pearson_r")
    median_mae = result.summary.get("median_mae")
    persistence_mae = baselines.get("persistence", {}).get("median_mae")
    block_mean_mae = baselines.get("block_mean_naive", {}).get("median_mae")
    criterion_a = median_r is not None and median_r > 0
    beats_persistence = (
        median_mae is not None and persistence_mae is not None
        and median_mae < persistence_mae
    )
    beats_block_mean = (
        median_mae is not None and block_mean_mae is not None
        and median_mae < block_mean_mae
    )
    return {
        "criterion_a_median_r_positive": bool(criterion_a),
        "criterion_b_beats_persistence_mae": bool(beats_persistence),
        "criterion_b_beats_block_mean_mae": bool(beats_block_mean),
        "criterion_b_met": bool(beats_persistence and beats_block_mean),
        "transfer_success": bool(criterion_a and beats_persistence
                                 and beats_block_mean),
        "components": {
            "median_pearson_r": median_r,
            "median_mae_model": median_mae,
            "median_mae_persistence": persistence_mae,
            "median_mae_block_mean": block_mean_mae,
        },
    }


def bootstrap_median_r(per_subject_r: np.ndarray,
                       n_resamples: int = protocol.BOOTSTRAP_N_RESAMPLES,
                       seed: int = protocol.BOOTSTRAP_SEED,
                       level: float = protocol.CI_LEVEL) -> dict:
    """Subject-cluster bootstrap of the median across-subject Pearson r:
    subjects (one r value each) resampled with replacement; the statistic
    is the median of the resampled values. Returns the observed statistic,
    percentile endpoints, the full draw list (auditability — the MOVE run
    saves this artifact verbatim), seed, level, and resample count."""
    values = np.asarray(per_subject_r, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise DataValidationError(
            "bootstrap_median_r needs at least 2 subjects with a finite r"
        )
    rng = np.random.default_rng(seed)
    alpha = (1.0 - level) / 2.0
    draws = np.empty(n_resamples)
    for index in range(n_resamples):
        sample = values[rng.integers(0, values.size, values.size)]
        draws[index] = float(np.median(sample))
    return {
        "statistic": float(np.median(values)),
        "ci_low": float(np.percentile(draws, 100.0 * alpha)),
        "ci_high": float(np.percentile(draws, 100.0 * (1.0 - alpha))),
        "ci_level": float(level),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n_subjects": int(values.size),
        "draws": [float(v) for v in draws],
    }


# ---------------------------------------------------------------------------
# Autocorrelation-preserving nulls
# ---------------------------------------------------------------------------


def _sample_circular_shift(span: int, min_bins: int, segment_bins: int,
                           blockwise: bool, rng: np.random.Generator,
) -> int | None:
    """One valid circular shift for a block of ``span`` target bins, or
    ``None`` when the block is too short to guarantee the minimum
    EFFECTIVE displacement.

    The guarantee is ``min(shift, span − shift) ≥ min_bins``: a nominal
    shift of ``span − 1`` is circularly a one-bin shift in the opposite
    direction, so the plain null samples from
    ``[min_bins, span − min_bins]`` and the blockwise null samples
    segment multiples under the same bound. Blocks shorter than twice
    the minimum displacement cannot satisfy it and return ``None`` —
    the caller counts the exclusion instead of shifting silently.
    """
    if span < 2 * min_bins:
        return None
    if blockwise:
        k_min = max(1, -(-min_bins // segment_bins))  # ceil division
        k_max = (span - min_bins) // segment_bins
        if k_max < k_min:
            return None
        return int(rng.integers(k_min, k_max + 1)) * segment_bins
    return int(rng.integers(min_bins, span - min_bins + 1))


def move_circular_shift_null(
    dataset: WindowDataset,
    model_name: str = "lagged_ridge_primary",
    block: str = "multiband_power",
    lags: tuple[float, ...] = protocol.MOVE_LAGS_S,
    n_shifts: int = protocol.MOVE_NULL_SHIFTS,
    seed: int = protocol.PERMUTATION_SEED,
    min_shift_s: float = protocol.MOVE_CIRCULAR_SHIFT_MIN_S,
    blockwise: bool = False,
) -> dict:
    """Autocorrelation-preserving null: circular shifts of the glove
    target WITHIN each recording block (neural windows unchanged), each
    shift's EFFECTIVE circular displacement ``min(shift, span − shift)``
    ≥ ``min_shift_s`` — breaks local neural–behavior alignment while
    preserving the target's autocorrelation and spectrum — with the
    decoder RETRAINED per shift through its frozen selection.

    The effective-displacement guarantee is what makes "each shift ≥ 2 s"
    true: a nominal shift of ``span − 1`` bins is circularly a one-bin
    shift in the opposite direction, so the sampling range is
    ``[min_bins, span − min_bins]`` and blocks shorter than twice the
    minimum displacement are EXCLUDED (counted in ``skipped_blocks``),
    never silently shifted.

    ``blockwise=True`` restricts shifts to multiples of 2 s segments
    under the same effective-displacement guarantee (the secondary null).
    Independent row shuffling is never used. Returns the null
    distribution of the headline metric (median across-subject Pearson
    r) and its conservative p-value against the observed run.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from .bootstrap import permutation_p_value
    from .evaluation import _default_splitter, make_preprocessing

    X = dataset.lag_subset(dataset.block_subset(block), lags)
    y = dataset.target
    groups = dataset.groups
    metadata = dataset.metadata
    observed_run = run_move_cv(dataset, model_names=(model_name,), block=block,
                               lags=lags)
    observed = observed_run[0].summary["median_pearson_r"]
    splits = list(_default_splitter(len(np.unique(groups))).split(X, y, groups))
    rng = np.random.default_rng(seed)
    min_bins = max(int(round(min_shift_s / protocol.MOVE_UPDATE_S)), 1)
    segment_bins = max(int(round(2.0 / protocol.MOVE_UPDATE_S)), 1)
    null_values: list[float] = []
    skipped_blocks = 0
    shifted_blocks = 0
    for _ in range(n_shifts):
        shifted = y.copy()
        for recording in sorted(set(metadata["recording_position"])):
            rows = metadata[metadata["recording_position"] == recording]
            for block_number in sorted(set(rows["block"])):
                indices = rows[rows["block"] == block_number].index.to_numpy()
                order = np.argsort(metadata.loc[indices, "target_bin"].to_numpy())
                indices = indices[order]
                span = indices.size
                shift = _sample_circular_shift(span, min_bins, segment_bins,
                                               blockwise, rng)
                if shift is None:
                    # Effective displacement cannot clear the minimum —
                    # exclude (counted), never shift silently.
                    skipped_blocks += 1
                    continue
                assert min(shift, span - shift) >= min_bins
                shifted[indices] = y[np.roll(indices, shift)]
                shifted_blocks += 1
        per_subject_r: list[float] = []
        for train_idx, test_idx in splits:
            params = _select_move_params(model_name, X.iloc[train_idx],
                                         shifted[train_idx], groups[train_idx],
                                         seed=seed)
            steps = [(name, clone(step)) for name, step in make_preprocessing()]
            pipe = Pipeline([*steps, ("model", _build_move_estimator(
                model_name, params, seed=seed))])
            pipe.fit(X.iloc[train_idx], shifted[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            for subject in sorted(set(groups[test_idx].tolist())):
                mask = groups[test_idx] == subject
                r = _pearson(shifted[test_idx][mask], pred[mask])
                if r is not None:
                    per_subject_r.append(r)
        null_values.append(
            float(np.median(per_subject_r)) if per_subject_r else float("nan")
        )
    null = np.asarray([v for v in null_values if np.isfinite(v)])
    observed_value = float(observed) if observed is not None else float("nan")
    return {
        "observed_median_pearson_r": observed_value,
        "null_median_pearson_r": null,
        "p_value": permutation_p_value(observed_value, null, "higher",
                                       null.size),
        "n_shifts": int(null.size),
        "min_shift_s": min_shift_s,
        "min_effective_shift_s": min_shift_s,
        "effective_displacement_rule": (
            "min(shift, span − shift) ≥ min_shift_s — sampled from "
            "[min_bins, span − min_bins]; blocks shorter than twice the "
            "minimum displacement are excluded and counted"
        ),
        "blocks_shifted_per_draw": shifted_blocks // max(n_shifts, 1),
        "blocks_skipped_per_draw": skipped_blocks // max(n_shifts, 1),
        "blockwise": blockwise,
    }


# ---------------------------------------------------------------------------
# Sanitized aggregates and plots
# ---------------------------------------------------------------------------

# Validated reference-palette slots (dataviz skill reference instance):
# categorical slots 1/2 in fixed order on the light surface; text in ink.
SERIES_COLOR_1 = "#2a78d6"   # observed / model (slot 1)
SERIES_COLOR_2 = "#eb6834"   # null / baseline (slot 2)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


def plot_move_results(
    results: list[MoveCVResult],
    null: dict | None = None,
    out_dir: str | Path | None = None,
    lag_ablation: list[MoveCVResult] | None = None,
    block_ablation: list[MoveCVResult] | None = None,
) -> dict[str, str]:
    """Sanitized aggregate plots (PNG, gitignored outputs).

    Panels: (1) per-subject Pearson-r distribution per model (strip,
    subject positions on the axis — no identifiers); (2) observed vs
    null (histogram + observed line); (3) lag ablation; (4) band-block
    ablation. Validated categorical slots in fixed order; one axis per
    panel; legend whenever two series share a panel; recessive axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        raise DataValidationError("an explicit output directory is required")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _style(ax, title, ylabel):
        ax.set_title(title, color=INK_PRIMARY, fontsize=10)
        ax.set_ylabel(ylabel, color=INK_PRIMARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # (1) per-subject distributions.
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    for slot, result in enumerate(results):
        values = result.per_subject["pearson_r"].dropna().to_numpy()
        color = SERIES_COLOR_1 if slot == 0 else SERIES_COLOR_2
        jitter = (np.arange(values.size) % 7) * 0.006
        ax.scatter(np.full(values.size, slot) + jitter, values, s=14,
                   color=color, alpha=0.75, edgecolors=SURFACE,
                   linewidths=0.5, label=result.model_name)
        ax.hlines(float(np.median(values)), slot - 0.25, slot + 0.25,
                  color=INK_PRIMARY, linewidth=1.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r.model_name for r in results], fontsize=8)
    if len(results) >= 2:
        ax.legend(frameon=False, fontsize=8)
    _style(ax, "Per-subject decoding (positions, no IDs)",
           "window-level Pearson r")
    fig.tight_layout()
    path = out_path / "per_subject_distribution.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths["per_subject_distribution"] = str(path)

    # (2) observed vs null.
    if null is not None:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        values = np.asarray(null["null_median_pearson_r"], dtype=float)
        ax.hist(values, bins=20, color=SERIES_COLOR_2, alpha=0.85,
                edgecolor=SURFACE, linewidth=0.5, label="circular-shift null")
        observed = null["observed_median_pearson_r"]
        ax.axvline(observed, color=SERIES_COLOR_1, linewidth=2,
                   label=f"observed = {observed:.3f}")
        ax.set_xlabel("median across-subject Pearson r", color=INK_PRIMARY)
        ax.legend(frameon=False, fontsize=8)
        _style(ax, f"Observed vs shift null (p = {null['p_value']:.4f}, "
                   f"n = {null['n_shifts']})", "null draws")
        fig.tight_layout()
        path = out_path / "observed_vs_null.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths["observed_vs_null"] = str(path)

    # (3) lag ablation.
    if lag_ablation:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        names = [name for name in lag_ablation]
        values = [lag_ablation[name].summary["median_pearson_r"] or 0.0
                  for name in lag_ablation]
        ax.bar(names, values, color=SERIES_COLOR_1, width=0.5)
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1)
        _style(ax, "Lag ablation (primary decoder)", "median Pearson r")
        fig.tight_layout()
        path = out_path / "lag_ablation.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths["lag_ablation"] = str(path)

    # (4) band-block ablation.
    if block_ablation:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        names = list(block_ablation)
        values = [block_ablation[name].summary["median_pearson_r"] or 0.0
                  for name in block_ablation]
        ax.bar(names, values, color=SERIES_COLOR_1, width=0.5)
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1)
        _style(ax, "Band-block ablation (primary decoder)",
               "median Pearson r")
        fig.tight_layout()
        path = out_path / "band_block_ablation.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths["band_block_ablation"] = str(path)
    return paths

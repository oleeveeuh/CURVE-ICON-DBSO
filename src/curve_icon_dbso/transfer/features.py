"""Grid window features with causal normalization (v2 contract §6).

Per target update bin, per grid offset (anchor-relative strip-1
position), per frozen band: log10 relative band power (Welch, 0.5 s
Hamming segments, 50 % overlap — the v1 estimator), plus per-offset
aperiodic exponent and the frozen missingness columns. The causal
trailing z-score (past 5 s of updates only, clip ±5) is applied in a
single chronological pass per recording — it never reads the current
bin or any future sample, so zero-shot use is the explicitly allowed
past-unlabeled-neural normalization. Windows never cross block,
calibration, purge, or test boundaries (the caller passes one span's
blocks; the builder drops any window that leaves the span with a
machine-readable reason).

**Frozen 1 s boundary guard (enforced in machinery since v2.1.0).**
Every retained row's full read range — the earliest sample any lagged
window starts through the lag-0 window end — must lie at least
``contract.BOUNDARY_GUARD_S`` inside BOTH edges of its (span-clipped)
block. Rows in the guard zone are dropped with a machine-readable
reason; the enforced distances are pinned by test, not by the
constant's value. Because callers pass span-clipped blocks, the same
guard keeps rows away from calibration/purge/test boundaries too.
Each meta row records ``window_earliest_sample`` AND
``window_latest_sample`` — the exact sample range read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..errors import DataValidationError
from ..neural_features import spectral_features, welch_psd
from ..signal_qc import (
    _median_abs_deviation,
    common_median_reference,
    detect_line_frequency,
    notch_line_noise,
    qc_channel_block,
)
from . import contract
from .spatial import build_grid_map, missingness_columns, ok_channel_mask


@dataclass
class FeatureTable:
    """Window × feature matrix with position-labeled metadata."""

    features: pd.DataFrame
    meta: pd.DataFrame          # block, target_bin, bin_start, validity counts
    target: np.ndarray | None   # caller-attached target values per kept row
    label: np.ndarray | None    # caller-attached detection labels per kept row
    qc: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)


def prepare_strip(
    strip_ecog: np.ndarray, fs: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rereference + line-noise treatment for one recording's strip.

    A whole-recording basic-QC mask gates the common-median reference
    (target-independent) AND the line-noise notch: the detected line
    frequency (+ harmonics) is notched on QC-OK channels only. Failed
    channels keep their raw values — never imputed, never notched
    (they are excluded downstream by the same mask; real recordings can
    carry entirely non-finite contacts, e.g. a dead P7).
    """
    strip_ecog = np.asarray(strip_ecog, dtype=float)
    ok = ok_channel_mask(strip_ecog)
    referenced = common_median_reference(strip_ecog, ok_mask=ok)
    line = detect_line_frequency(np.median(referenced[ok], axis=0), fs)
    prepared = referenced
    if line["line_freq_hz"] is not None:
        prepared = np.array(referenced, dtype=float, copy=True)
        for channel in np.flatnonzero(ok):
            prepared[channel] = notch_line_noise(
                referenced[channel], fs, line["line_freq_hz"])
    qc = {
        "n_channels": int(strip_ecog.shape[0]),
        "n_ok_channels_reference": int(ok.sum()),
        "line_freq_hz": line["line_freq_hz"],
        "line_freq_ambiguous": bool(line["ambiguous"]),
    }
    return prepared, ok, qc


def causal_trailing_z(values: np.ndarray, trailing_bins: int,
                      clip: float) -> np.ndarray:
    """Strictly causal trailing z: bin i is standardized against bins
    [max(0, i−trailing), i) — never bin i, never the future. Fewer than
    3 past bins (warm-up) returns NaN; the caller drops those rows with
    a recorded reason. A zero-variance past window scores 0.0."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for i in range(values.size):
        past = values[max(0, i - trailing_bins):i]
        if past.size < 3:
            continue
        std = float(np.std(past))
        if not np.isfinite(std) or std <= 0:
            out[i] = 0.0
            continue
        out[i] = float(np.clip((values[i] - float(np.mean(past))) / std, -clip, clip))
    return out


def _spectral_row(window: np.ndarray, fs: float,
                  bands: dict[str, tuple[float, float]]) -> dict[str, float] | None:
    """One offset-window's spectral features (single lag), or None."""
    if window.size < int(round(contract.WELCH_SEGMENT_S * fs)):
        return None
    if not np.isfinite(window).all():
        return None
    freqs, psd = welch_psd(window, fs, segment_s=contract.WELCH_SEGMENT_S)
    features = spectral_features(freqs, psd, {"bands_active": dict(bands)},
                                 contract.BAND_TOP_HZ_MAX)
    row: dict[str, float] = {}
    for band in bands:
        value = features[f"{band}_log_relative_power"]
        if value is None or not np.isfinite(value):
            return None
        row[band] = float(value)
    exponent = float(features["aperiodic_exponent"])
    if not np.isfinite(exponent):
        return None
    row["aperiodic_exponent"] = exponent
    return row


def _row_key(channel: int, lag_start: int, lag_end: int) -> tuple[int, int, int]:
    return (int(channel), int(lag_start), int(lag_end))


def _normalized_columns(columns: list[str]) -> list[str]:
    """Spectral feature columns (the only ones causally normalized)."""
    return [c for c in columns
            if c.startswith("off") and not c.endswith("_missing")
            and c != "coverage_frac"]


def build_grid_features(
    strip_ecog: np.ndarray,
    fs: float,
    anchor_position: int,
    blocks: list[tuple[int, int, str]],
    window_s: float = contract.WINDOW_S_PRIMARY,
    with_lags: bool = True,
) -> FeatureTable:
    """Feature rows for every target bin of the given blocks (one
    recording, one span), chronological. Per-window QC (v1.0.1 §4:
    flatline / saturation / transients at the recording's diff-MAD)
    drops bins with machine-readable reasons; a bin keeps only offsets
    whose lag-0 window passed QC and whose every lag produced valid
    spectral features — a bin with no valid offsets is dropped.
    """
    grid_map = build_grid_map(strip_ecog.shape[0], anchor_position)
    prepared, ok_mask, prep_qc = prepare_strip(strip_ecog, fs)
    diff_mad = float(_median_abs_deviation(
        np.diff(prepared[ok_mask], axis=1))) if ok_mask.sum() > 1 else 0.0
    static_missing = missingness_columns(grid_map)
    trailing = int(round(contract.CAUSAL_NORM_TRAILING_S / contract.UPDATE_S))

    lags = contract.LAG_STACK_S if with_lags else contract.DETECTION_LAGS_S
    update = max(int(round(contract.UPDATE_S * fs)), 1)
    guard = int(round(contract.BOUNDARY_GUARD_S * fs))
    max_lag_shift = int(round(max(lags) * fs))
    rows: list[dict] = []
    meta_rows: list[dict] = []
    drops: dict[str, int] = {}
    # Per-build spectral memo (pure performance, identical output): the
    # lag stack makes windows overlap exactly — the lag-k window of bin
    # b IS the lag-0 window of bin b−k — so each unique (channel,
    # window-span) spectral row is computed once per build.
    spectral_memo: dict[tuple[int, int, int], dict[str, float] | None] = {}

    for block_number, (start, end, _kind) in enumerate(blocks):
        block_end = min(end, prepared.shape[1])
        n_bins = max(0, block_end - start) // update
        for target_bin in range(n_bins):
            target_time_s = (target_bin + 0.5) * contract.UPDATE_S
            end0 = int(round(target_time_s * fs))
            start0 = end0 - int(round(window_s * fs))
            if start0 < 0 or end0 > (block_end - start):
                drops["window_outside_block"] = drops.get("window_outside_block", 0) + 1
                continue
            # Frozen 1 s boundary guard, enforced in sample space: the
            # row's full read range (earliest lagged start .. lag-0 end)
            # must sit ≥ guard samples inside both clipped block edges.
            earliest_possible = start0 - max_lag_shift
            if earliest_possible < guard or end0 > (block_end - start) - guard:
                drops["boundary_guard_1s"] = drops.get("boundary_guard_1s", 0) + 1
                continue
            per_offset: dict[int, dict[str, float]] = {}
            offset_failed = 0
            earliest_lag_start = None
            for offset, channel in grid_map.offset_to_channel.items():
                if not ok_mask[channel]:
                    offset_failed += 1
                    continue
                window = prepared[channel, start + start0:start + end0]
                if window.size != int(round(window_s * fs)):
                    offset_failed += 1
                    continue
                qc = qc_channel_block(window, diff_mad)
                if qc["qc_status"] != "ok":
                    offset_failed += 1
                    continue
                per_lag: dict[str, float] = {}
                lag_ok = True
                for lag in lags:
                    shift = int(round(lag * fs))
                    lag_start, lag_end = start0 - shift, end0 - shift
                    if lag_start < 0 or lag_end <= lag_start or lag_end > (block_end - start):
                        lag_ok = False
                        break
                    if earliest_lag_start is None or lag_start < earliest_lag_start:
                        earliest_lag_start = lag_start
                    memo_key = _row_key(channel, start + lag_start,
                                        start + lag_end)
                    if memo_key not in spectral_memo:
                        spectral_memo[memo_key] = _spectral_row(
                            prepared[channel, start + lag_start:start + lag_end],
                            fs, contract.BANDS_HZ,
                        )
                    features = spectral_memo[memo_key]
                    if features is None:
                        lag_ok = False
                        break
                    suffix = f"_{lag:.1f}s" if len(lags) > 1 else ""
                    for key, value in features.items():
                        per_lag[f"off{offset:+d}_{key}{suffix}"] = value
                if lag_ok:
                    per_offset[offset] = per_lag
                else:
                    offset_failed += 1
            if not per_offset:
                drops["no_valid_offsets"] = drops.get("no_valid_offsets", 0) + 1
                continue
            row: dict[str, float] = {}
            for per_lag in per_offset.values():
                row.update(per_lag)
            row.update(static_missing)
            row["coverage_frac"] = len(per_offset) / len(contract.GRID_OFFSETS)
            rows.append(row)
            meta_rows.append({
                "block": block_number,
                "block_kind": _kind,
                "target_bin": target_bin,
                "bin_start_sample": start + target_bin * update,
                # Earliest/latest samples any window of this bin reads —
                # spans use them to forbid windows crossing a span/purge
                # boundary and to pin the enforced boundary-guard
                # distances (frozen leakage rule, v2.1.0).
                "window_earliest_sample": start + (earliest_lag_start if earliest_lag_start is not None else start0),
                "window_latest_sample": start + end0 - 1,
                "n_offsets_valid": len(per_offset),
                "n_offsets_failed": offset_failed,
            })

    meta = pd.DataFrame(meta_rows)
    if not rows:
        first_band = next(iter(contract.BANDS_HZ))
        empty = pd.DataFrame(columns=[f"off+0_{first_band}"])
        return FeatureTable(features=empty, meta=meta, target=None, label=None,
                            qc={"prepared": prep_qc, "drops": drops,
                                "grid": grid_map.coverage_summary()})
    frame = pd.DataFrame(rows)
    spectral = _normalized_columns(list(frame.columns))
    for column in spectral:
        frame[column] = causal_trailing_z(
            frame[column].to_numpy(dtype=float), trailing, contract.CAUSAL_CLIP)
    warmup_rows = frame[spectral].isna().any(axis=1)
    n_warmup = int(warmup_rows.sum())
    if n_warmup:
        drops["causal_warmup"] = n_warmup
        frame = frame.loc[~warmup_rows].reset_index(drop=True)
        meta = meta.loc[~warmup_rows.to_numpy()].reset_index(drop=True)
    return FeatureTable(features=frame, meta=meta, target=None, label=None,
                        qc={"prepared": prep_qc, "drops": drops,
                            "grid": grid_map.coverage_summary()})


def build_bipolar_features(
    strip_ecog: np.ndarray,
    fs: float,
    blocks: list[tuple[int, int, str]],
    window_s: float = contract.WINDOW_S_PRIMARY,
    with_lags: bool = True,
) -> FeatureTable:
    """The single prespecified spatial ablation: adjacent-contact bipolar
    pairs on the anchored strip (7 pairs), UNANCHORED pair-index
    topology — the contrast against the anchored referential grid tests
    what anchor alignment contributes. Pair features die with either
    parent contact (never imputed); no grid offsets, no missingness
    borrow from other pairs."""
    from ..signal_qc import bipolar_rereference

    strip_ecog = np.asarray(strip_ecog, dtype=float)
    if strip_ecog.shape[0] < 8:
        raise DataValidationError("bipolar ablation needs a full 8-contact strip")
    pairs = [(i, i + 1) for i in range(7)]
    bipolar, labels = bipolar_rereference(strip_ecog[:8], pairs)
    ok = np.isfinite(bipolar).all(axis=1)
    for ch in range(bipolar.shape[0]):
        if ok[ch]:
            ok[ch] = (float(np.max(bipolar[ch]) - np.min(bipolar[ch])) > 1e-10)
    referenced = bipolar - np.median(bipolar[ok], axis=0)
    line = detect_line_frequency(np.median(referenced[ok], axis=0), fs)
    prepared = referenced
    if line["line_freq_hz"] is not None:
        # Notch QC-OK pair channels only — a pair that died with a
        # non-finite parent keeps its raw values (never imputed; it is
        # excluded by the same mask downstream).
        prepared = np.array(referenced, dtype=float, copy=True)
        for channel in np.flatnonzero(ok):
            prepared[channel] = notch_line_noise(
                referenced[channel], fs, line["line_freq_hz"])
    diff_mad = float(_median_abs_deviation(
        np.diff(prepared[ok], axis=1))) if ok.sum() > 1 else 0.0
    trailing = int(round(contract.CAUSAL_NORM_TRAILING_S / contract.UPDATE_S))
    lags = contract.LAG_STACK_S if with_lags else contract.DETECTION_LAGS_S
    update = max(int(round(contract.UPDATE_S * fs)), 1)
    guard = int(round(contract.BOUNDARY_GUARD_S * fs))
    max_lag_shift = int(round(max(lags) * fs))
    rows: list[dict] = []
    meta_rows: list[dict] = []
    drops: dict[str, int] = {}
    spectral_memo: dict[tuple[int, int, int], dict[str, float] | None] = {}
    for block_number, (start, end, _kind) in enumerate(blocks):
        block_end = min(end, prepared.shape[1])
        n_bins = max(0, block_end - start) // update
        for target_bin in range(n_bins):
            target_time_s = (target_bin + 0.5) * contract.UPDATE_S
            end0 = int(round(target_time_s * fs))
            start0 = end0 - int(round(window_s * fs))
            if start0 < 0 or end0 > (block_end - start):
                drops["window_outside_block"] = drops.get("window_outside_block", 0) + 1
                continue
            # Frozen 1 s boundary guard in sample space (mirrors the grid
            # builder; v2.1.0).
            if start0 - max_lag_shift < guard \
                    or end0 > (block_end - start) - guard:
                drops["boundary_guard_1s"] = drops.get("boundary_guard_1s", 0) + 1
                continue
            per_pair: dict[int, dict[str, float]] = {}
            for pair_index, channel in enumerate(range(prepared.shape[0])):
                if not ok[channel]:
                    continue
                window = prepared[channel, start + start0:start + end0]
                if window.size != int(round(window_s * fs)):
                    continue
                if qc_channel_block(window, diff_mad)["qc_status"] != "ok":
                    continue
                per_lag: dict[str, float] = {}
                lag_ok = True
                for lag in lags:
                    shift = int(round(lag * fs))
                    lag_start, lag_end = start0 - shift, end0 - shift
                    if lag_start < 0 or lag_end <= lag_start or lag_end > (block_end - start):
                        lag_ok = False
                        break
                    memo_key = _row_key(channel, start + lag_start,
                                        start + lag_end)
                    if memo_key not in spectral_memo:
                        spectral_memo[memo_key] = _spectral_row(
                            prepared[channel, start + lag_start:start + lag_end],
                            fs, contract.BANDS_HZ)
                    feats = spectral_memo[memo_key]
                    if feats is None:
                        lag_ok = False
                        break
                    suffix = f"_{lag:.1f}s" if len(lags) > 1 else ""
                    for key, value in feats.items():
                        per_lag[f"pair{pair_index}_{key}{suffix}"] = value
                if lag_ok:
                    per_pair[pair_index] = per_lag
            if not per_pair:
                drops["no_valid_pairs"] = drops.get("no_valid_pairs", 0) + 1
                continue
            row: dict[str, float] = {}
            for per_lag in per_pair.values():
                row.update(per_lag)
            row["coverage_frac"] = len(per_pair) / len(pairs)
            rows.append(row)
            meta_rows.append({
                "block": block_number,
                "block_kind": _kind,
                "target_bin": target_bin,
                "bin_start_sample": start + target_bin * update,
                "window_earliest_sample": start + start0,
                "window_latest_sample": start + end0 - 1,
                "n_offsets_valid": len(per_pair),
                "n_offsets_failed": len(pairs) - len(per_pair),
            })
    meta = pd.DataFrame(meta_rows)
    if not rows:
        first_band = next(iter(contract.BANDS_HZ))
        empty = pd.DataFrame(columns=[f"pair0_{first_band}"])
        return FeatureTable(features=empty, meta=meta, target=None, label=None,
                            qc={"drops": drops, "bipolar_labels": labels})
    frame = pd.DataFrame(rows)
    spectral = [c for c in frame.columns
                if c.startswith("pair") and c != "coverage_frac"]
    for column in spectral:
        frame[column] = causal_trailing_z(
            frame[column].to_numpy(dtype=float), trailing, contract.CAUSAL_CLIP)
    warmup_rows = frame[spectral].isna().any(axis=1)
    if int(warmup_rows.sum()):
        drops["causal_warmup"] = int(warmup_rows.sum())
        frame = frame.loc[~warmup_rows].reset_index(drop=True)
        meta = meta.loc[~warmup_rows.to_numpy()].reset_index(drop=True)
    return FeatureTable(features=frame, meta=meta, target=None, label=None,
                        qc={"drops": drops, "bipolar_labels": labels})

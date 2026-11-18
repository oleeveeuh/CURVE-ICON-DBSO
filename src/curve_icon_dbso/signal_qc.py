"""Signal QC, rereferencing, and line-noise handling for the follow-on
protocol (frozen contract v1.0.1, ``curve_icon_dbso.protocol``).

Implements the frozen artifact/QC rules of the contract as **pure
functions on arrays** — verification uses synthetic fixtures only, and
no function here ever reads targets, fold membership, or identity:

* validated bipolar rereferencing (requires explicit electrode-adjacency
  pairs) with the contract's documented fallback — a common median
  reference across QC-OK channels — when adjacency is unavailable;
* line-noise detection (larger PSD maximum within ±2 Hz of 50 or 60 Hz;
  within 3 dB → ``line_freq_ambiguous`` note) and zero-phase notch
  filtering at the detected frequency plus harmonics up to Nyquist;
* explicit, machine-readable rejection of flatlines, saturation (rail
  detection), rapid transients, and implausible band power — failed
  windows/channels are counted with reasons and never imputed,
  zero-filled, or silently dropped;
* a block-interior window grid (windows never cross REST/MOVE block
  boundaries or invalid intervals).

Numeric thresholds here are the concrete readings of the frozen
``protocol.QC_RULES`` prose; ``tests/test_signal_qc.py`` cross-checks
that the two stay in lockstep.

Units: signals are in raw sensor units (uncalibrated); band powers are
(sensor units)² (Welch PSD integrated over Hz); all QC quantities are
dimensionless or in sensor units as documented per function.

Acquisition note (recorded, not guessed): the rail-detection reading of
the frozen saturation rule is per channel-block — "the recording's max
or min sample value" is applied within each channel's own block, which
is strictly more sensitive than cohort-wide rails and cannot mask a
pinned channel. The frozen 5 % fraction is unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

from . import protocol
from .errors import DataValidationError, SamplingRateError

# --- Frozen QC thresholds (mirror protocol.QC_RULES prose; cross-checked
# --- by tests/test_signal_qc.py::test_thresholds_match_frozen_contract).
FLATLINE_TOL: float = 1e-12
SATURATION_FRACTION: float = 0.05
TRANSIENT_MAD_MULT: float = 20.0
TRANSIENT_FRACTION: float = 0.005
IMPLAUSIBLE_POWER_Z: float = 10.0
LINE_FREQ_CANDIDATES_HZ: tuple[float, float] = (50.0, 60.0)
LINE_FREQ_TOLERANCE_HZ: float = 2.0
LINE_FREQ_AMBIGUITY_DB: float = 3.0
NOTCH_WIDTH_HZ: float = 2.0  # notch half-width: line_freq ± 2 Hz
WELCH_SEGMENT_SECONDS: float = 1.0


def validate_fs(fs: float, top_edge_hz: float) -> None:
    """Validate a sampling rate against the topmost active band edge.

    Raises:
        SamplingRateError: On non-finite/non-positive fs, or when the top
            edge does not clear ``fs/2 - protocol.NYQUIST_MARGIN_HZ``.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise SamplingRateError(f"Sampling rate must be a positive finite number, got {fs!r}")
    ceiling = fs / 2.0 - protocol.NYQUIST_MARGIN_HZ
    if top_edge_hz > ceiling:
        raise SamplingRateError(
            f"Top band edge {top_edge_hz} Hz does not clear fs/2 − "
            f"{protocol.NYQUIST_MARGIN_HZ} Hz margin (fs={fs} Hz allows up to "
            f"{ceiling} Hz). Resolve the cohort's band set with "
            "protocol.resolve_spectral_bands(min cohort fs) first."
        )


def active_top_edge(bands_resolved: dict) -> float:
    """Topmost active band edge for a ``resolve_spectral_bands`` result.

    gamma_high when active, else the highest standard band's upper edge.
    The HFO block is NOT part of the band ladder (separate block, gated
    separately by ``hfo_block_feasible`` plus the artifact QC gate).
    """
    edges = [limits[1] for limits in bands_resolved["bands_active"].values()]
    if bands_resolved.get("gamma_high") is not None:
        edges.append(bands_resolved["gamma_high"][1])
    if not edges:
        raise DataValidationError(
            "No band is active at this sampling rate; the recording cannot "
            "produce the frozen feature set."
        )
    return float(max(edges))


# ---------------------------------------------------------------------------
# Rereferencing
# ---------------------------------------------------------------------------


def bipolar_rereference(
    ecog: np.ndarray, pairs: list[tuple[int, int]]
) -> tuple[np.ndarray, list[str]]:
    """Validated bipolar rereferencing over explicit electrode pairs.

    ``pairs`` holds ``(anode, cathode)`` channel indices; each output
    channel is ``ecog[anode] − ecog[cathode]``. Pair labels are
    ``"{anode}-{cathode}"`` positions — never participant identifiers.

    Raises:
        DataValidationError: On empty pairs, out-of-range or degenerate
            (anode == cathode) pairs, or a non-2-D array.
    """
    ecog = np.asarray(ecog, dtype=float)
    if ecog.ndim != 2:
        raise DataValidationError(f"ecog must be 2-D (channels × samples); got {ecog.shape}")
    if not pairs:
        raise DataValidationError(
            "Bipolar rereferencing requires an explicit adjacency list of "
            "(anode, cathode) pairs; without one use common_median_reference."
        )
    n_channels = ecog.shape[0]
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for anode, cathode in pairs:
        for idx in (anode, cathode):
            if not (0 <= idx < n_channels):
                raise DataValidationError(
                    f"adjacency pair ({anode}, {cathode}) indexes channel {idx} "
                    f"but the recording has {n_channels} channels"
                )
        if anode == cathode:
            raise DataValidationError(f"degenerate bipolar pair ({anode}, {cathode})")
        rows.append(ecog[anode] - ecog[cathode])
        labels.append(f"{anode}-{cathode}")
    return np.vstack(rows), labels


def common_median_reference(ecog: np.ndarray, ok_mask: np.ndarray | None = None) -> np.ndarray:
    """Contract fallback reference: subtract the per-sample median across
    QC-OK channels (target-independent, per recording).

    A channel whose samples are non-finite is excluded from the median
    (and flagged upstream by the caller's QC) — never imputed.
    """
    ecog = np.asarray(ecog, dtype=float)
    if ecog.ndim != 2:
        raise DataValidationError(f"ecog must be 2-D (channels × samples); got {ecog.shape}")
    if ok_mask is None:
        ok_mask = np.isfinite(ecog).all(axis=1)
    ok_mask = np.asarray(ok_mask, dtype=bool)
    if ok_mask.shape[0] != ecog.shape[0]:
        raise DataValidationError("ok_mask must have one entry per channel")
    if not ok_mask.any():
        raise DataValidationError("common median reference needs at least one OK channel")
    reference = np.median(ecog[ok_mask], axis=0)
    return ecog - reference


# ---------------------------------------------------------------------------
# Line noise
# ---------------------------------------------------------------------------


def detect_line_frequency(signal: np.ndarray, fs: float) -> dict:
    """Detect the line frequency per the frozen rule.

    The line frequency is the larger of the Welch-PSD maxima within ±2 Hz
    of 50 Hz and of 60 Hz; if the two maxima are within 3 dB the record
    carries ``line_freq_ambiguous`` and the larger wins. Candidates above
    the usable ceiling (fs/2 − margin − tolerance) are ignored.

    Returns ``{"line_freq_hz": float | None, "ambiguous": bool,
    "psd_db_50": float | None, "psd_db_60": float | None}``.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if not np.isfinite(signal).all():
        raise DataValidationError("line-frequency detection needs finite samples")
    if not np.isfinite(fs) or fs <= 0:
        raise SamplingRateError(f"Sampling rate must be a positive finite number, got {fs!r}")
    ceiling = fs / 2.0 - protocol.NYQUIST_MARGIN_HZ - LINE_FREQ_TOLERANCE_HZ
    nperseg = min(int(WELCH_SEGMENT_SECONDS * fs), signal.size)
    if nperseg < 8:
        raise DataValidationError("signal too short for line-frequency detection")
    freqs, psd = welch(signal, fs=fs, window="hamming", nperseg=nperseg,
                       noverlap=nperseg // 2)
    result: dict = {"line_freq_hz": None, "ambiguous": False,
                    "psd_db_50": None, "psd_db_60": None}
    band_powers_db: list[float | None] = []
    for candidate in LINE_FREQ_CANDIDATES_HZ:
        if candidate > ceiling:
            band_powers_db.append(None)
            continue
        near = (freqs >= candidate - LINE_FREQ_TOLERANCE_HZ) & (
            freqs <= candidate + LINE_FREQ_TOLERANCE_HZ
        )
        if not near.any():
            band_powers_db.append(None)
            continue
        peak = float(np.max(psd[near]))
        band_powers_db.append(10.0 * np.log10(peak) if peak > 0 else None)
    valid = [(candidate, value) for candidate, value in
             zip(LINE_FREQ_CANDIDATES_HZ, band_powers_db, strict=True)
             if value is not None]
    if not valid:
        return result
    result["psd_db_50"], result["psd_db_60"] = band_powers_db
    best_freq, best_db = max(valid, key=lambda item: item[1])
    if len(valid) == 2 and (best_db - min(v for _, v in valid)) < LINE_FREQ_AMBIGUITY_DB:
        result["ambiguous"] = True
    result["line_freq_hz"] = float(best_freq)
    return result


def notch_line_noise(signal: np.ndarray, fs: float, line_freq_hz: float) -> np.ndarray:
    """Zero-phase notch at ``line_freq_hz`` and harmonics up to Nyquist.

    Each notch is a 2nd-order IIR notch (scipy ``iirnotch``) with a
    ±2 Hz half-width (Q = f0 / 4), applied with ``filtfilt`` so the line
    removal introduces no phase distortion. Harmonics at or above the
    usable ceiling (fs/2 − margin) are skipped.

    Raises:
        DataValidationError: On non-finite input or a non-positive line
            frequency.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if not np.isfinite(signal).all():
        raise DataValidationError("notch filtering needs finite samples")
    if not (np.isfinite(line_freq_hz) and line_freq_hz > 0):
        raise DataValidationError(f"invalid line frequency: {line_freq_hz!r}")
    ceiling = fs / 2.0 - protocol.NYQUIST_MARGIN_HZ
    filtered = signal
    harmonic = line_freq_hz
    while harmonic < ceiling:
        b, a = iirnotch(harmonic, Q=harmonic / (2.0 * NOTCH_WIDTH_HZ), fs=fs)
        filtered = filtfilt(b, a, filtered)
        harmonic += line_freq_hz
    return filtered


# ---------------------------------------------------------------------------
# Channel-block / window QC
# ---------------------------------------------------------------------------


def _median_abs_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def recording_diff_mad(ecog: np.ndarray) -> float:
    """Per-recording reference scale: MAD of the sample-to-sample
    differences over all finite channels (the frozen transient rule's
    reference distribution; target-independent)."""
    ecog = np.asarray(ecog, dtype=float)
    if ecog.ndim != 2:
        raise DataValidationError(f"ecog must be 2-D (channels × samples); got {ecog.shape}")
    finite = ecog[np.isfinite(ecog).all(axis=1)]
    if finite.size == 0:
        raise DataValidationError("recording_diff_mad needs at least one finite channel")
    diffs = np.concatenate([np.diff(channel) for channel in finite])
    mad = _median_abs_deviation(diffs)
    if mad <= 0.0:
        # Degenerate (e.g., constant recording): fall back to the diffs'
        # own scale so the transient rule stays defined; the flatline rule
        # will catch constant blocks anyway.
        return float(np.std(diffs)) or float(np.finfo(float).eps)
    return mad


def qc_channel_block(
    block: np.ndarray, diff_mad: float, reference_band_power_z: dict[str, float] | None = None
) -> dict:
    """QC one channel-block (one channel, one REST/MOVE block) per the
    frozen rules.

    Checks, in order: non-finite samples; flatline (max − min ≤ 1e-12);
    saturation (> 5 % of samples at the block's own max or min — rail
    detection); rapid transients (fraction of |diff| samples exceeding
    20 × the recording's diff-MAD above 0.005); implausible band power
    (caller-supplied per-band z-scores vs the recording's REST-window
    distribution, any |z| > 10 fails).

    Returns ``{"qc_status", "qc_reason", "n_samples"}`` — a failed block
    is a record, never a repair.
    """
    block = np.asarray(block, dtype=float).ravel()
    record = {"qc_status": "ok", "qc_reason": "", "n_samples": int(block.size)}
    if block.size == 0:
        record.update({"qc_status": "failed", "qc_reason": "empty block"})
        return record
    if not np.isfinite(block).all():
        record.update({
            "qc_status": "failed",
            "qc_reason": f"non-finite samples ({int(np.count_nonzero(~np.isfinite(block)))})",
        })
        return record
    value_range = float(block.max() - block.min())
    if value_range <= FLATLINE_TOL:
        record.update({"qc_status": "failed", "qc_reason": "flatline"})
        return record
    rail_count = int(np.count_nonzero((block == block.max()) | (block == block.min())))
    if rail_count / block.size > SATURATION_FRACTION:
        record.update({
            "qc_status": "failed",
            "qc_reason": f"saturation ({rail_count}/{block.size} samples at rails)",
        })
        return record
    if diff_mad > 0:
        diffs = np.abs(np.diff(block))
        transient_fraction = float(np.count_nonzero(diffs > TRANSIENT_MAD_MULT * diff_mad)) / max(diffs.size, 1)
        if transient_fraction > TRANSIENT_FRACTION:
            record.update({
                "qc_status": "failed",
                "qc_reason": f"rapid transients (fraction {transient_fraction:.4f})",
            })
            return record
    if reference_band_power_z:
        for band, z in reference_band_power_z.items():
            if np.isfinite(z) and abs(float(z)) > IMPLAUSIBLE_POWER_Z:
                record.update({
                    "qc_status": "failed",
                    "qc_reason": f"implausible_band_power ({band}: z={z:.1f})",
                })
                return record
    return record


def window_grid(
    block_start: int, block_end: int, fs: float,
    window_s: float = protocol.MOVE_WINDOW_S,
    update_s: float = protocol.MOVE_UPDATE_S,
) -> list[tuple[int, int]]:
    """Contract MOVE window grid inside one block's sample bounds.

    Windows are ``[t − window_s, t)`` ending at the target sample (the
    causal convention), stepped by ``update_s``, and never extend past
    the block bounds — a window that would cross a block boundary simply
    does not exist. Returns ``(start, end)`` half-open sample pairs.

    Raises:
        DataValidationError: On non-positive fs/window/update or bounds.
    """
    for name, value in (("fs", fs), ("window_s", window_s), ("update_s", update_s)):
        if not np.isfinite(value) or value <= 0:
            raise DataValidationError(f"{name} must be a positive finite number, got {value!r}")
    if block_end <= block_start or block_start < 0:
        raise DataValidationError(
            f"invalid block bounds: start={block_start}, end={block_end}"
        )
    window_samples = int(round(window_s * fs))
    update_samples = max(int(round(update_s * fs)), 1)
    grid: list[tuple[int, int]] = []
    # First window ends one window-length into the block.
    end = block_start + window_samples
    while end <= block_end:
        grid.append((end - window_samples, end))
        end += update_samples
    return grid


def qc_windows(
    block: np.ndarray,
    fs: float,
    diff_mad: float,
    window_s: float = protocol.MOVE_WINDOW_S,
    update_s: float = protocol.MOVE_UPDATE_S,
    reference_band_power_z: dict[str, list[float]] | None = None,
) -> list[dict]:
    """Per-window QC records over one channel-block (MOVE-arm stream).

    ``reference_band_power_z`` optionally maps band name → per-window
    z-scores against the recording's own REST-window distribution; any
    |z| > 10 fails that window per band. Records carry machine-readable
    ``reasons`` lists (a window can fail several checks) and never
    modify the underlying data.
    """
    grid = window_grid(0, block.shape[-1], fs, window_s, update_s)
    block = np.asarray(block, dtype=float).ravel()
    records: list[dict] = []
    for start, end in grid:
        window = block[start:end]
        reasons: list[str] = []
        if window.size == 0 or not np.isfinite(window).all():
            reasons.append("non_finite_samples")
        else:
            if float(window.max() - window.min()) <= FLATLINE_TOL:
                reasons.append("flatline")
            rails = int(np.count_nonzero((window == window.max()) | (window == window.min())))
            if rails / window.size > SATURATION_FRACTION:
                reasons.append("saturation")
            if diff_mad > 0:
                diffs = np.abs(np.diff(window))
                fraction = float(np.count_nonzero(diffs > TRANSIENT_MAD_MULT * diff_mad)) / max(diffs.size, 1)
                if fraction > TRANSIENT_FRACTION:
                    reasons.append("rapid_transients")
            if reference_band_power_z:
                for band, z_list in reference_band_power_z.items():
                    z = float(z_list[len(records)]) if len(records) < len(z_list) else float("nan")
                    if np.isfinite(z) and abs(z) > IMPLAUSIBLE_POWER_Z:
                        reasons.append(f"implausible_band_power:{band}")
        records.append({
            "start": start,
            "end": end,
            "qc_status": "ok" if not reasons else "failed",
            "reasons": reasons,
        })
    return records


def qc_summary(records: list[dict]) -> dict:
    """Aggregate QC records into counts + reason frequencies.

    The summary carries **no participant identifiers** — only counts and
    reason strings (channel/block positions stay in the per-record list,
    which callers keep in gitignored local outputs).
    """
    n_ok = sum(1 for record in records if record.get("qc_status") == "ok")
    n_failed = len(records) - n_ok
    reasons: dict[str, int] = {}
    for record in records:
        if record.get("qc_status") != "ok":
            for reason in record.get("reasons", [record.get("qc_reason", "unspecified")]):
                key = reason.split(" (")[0]
                reasons[key] = reasons.get(key, 0) + 1
    return {"n_ok": n_ok, "n_failed": n_failed, "reasons": reasons}


# ---------------------------------------------------------------------------
# Band-pass helper shared by the feature layer and PAC
# ---------------------------------------------------------------------------


def bandpass(signal: np.ndarray, fs: float, low_hz: float, high_hz: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (order 4 default), Nyquist-
    validated. Raises :class:`SamplingRateError` when the band does not
    clear Nyquist with the frozen margin.

    Implemented with second-order sections (``sosfiltfilt``) — the same
    frozen filter in its numerically stable factorization; the raw
    ``(b, a)`` form of a narrow normalized band-pass is so ill-conditioned
    that its output can vary with input scale at the 1e-2 level, which
    would corrupt amplitude-invariance properties."""
    signal = np.asarray(signal, dtype=float).ravel()
    if not np.isfinite(signal).all():
        raise DataValidationError("band-pass filtering needs finite samples")
    if not (0.0 < low_hz < high_hz):
        raise DataValidationError(f"invalid band [{low_hz}, {high_hz}]")
    ceiling = fs / 2.0 - protocol.NYQUIST_MARGIN_HZ
    if high_hz > ceiling:
        raise SamplingRateError(
            f"band upper edge {high_hz} Hz does not clear fs/2 − "
            f"{protocol.NYQUIST_MARGIN_HZ} Hz margin (fs={fs} Hz)"
        )
    nyquist = fs / 2.0
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band",
                 output="sos")
    return sosfiltfilt(sos, signal)

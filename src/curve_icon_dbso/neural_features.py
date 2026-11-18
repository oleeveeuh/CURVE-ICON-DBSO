"""Neural feature extraction for the follow-on protocol (frozen contract
v1.0.1, ``curve_icon_dbso.protocol``): contract-approved Welch PSD,
frozen band boundaries, relative/log/aperiodic-adjusted power, dynamics
features, beta bursts, REST / MOVE / MOVE-minus-REST feature blocks, and
region-aware robust aggregation.

Pure functions on arrays; every quantity is target-independent and
fold-independent. Nothing here imputes failed blocks or zero-fills
failed channels: a failed channel-block keeps its ``None`` features and
is counted in the returned QC summaries (reason strings only — no
participant identifiers).

Conventions fixed by the contract and implemented here:

* Welch PSD: 1 s Hamming segments, 50 % overlap (the same estimator the
  frozen ``hfo``/``kinematics`` modules use). Multitaper is deliberately
  NOT added — the contract approves Welch.
* Band membership: half-open ``[low, high)`` over Welch bin centers,
  with the topmost active edge inclusive (``active_top_edge``).
* ``{band}_power``: Welch PSD integral over the band — units (sensor
  units)². ``relative`` divides by total power over the active range
  (dimensionless); ``log_relative`` is log10 of that (dimensionless);
  ``aperiodic_adjusted`` divides mean in-band PSD by the aperiodic fit's
  predicted PSD at the band center (dimensionless periodic surplus).
* ``aperiodic_exponent``: χ in PSD ∝ 1/f^χ — computed as minus the OLS
  slope of log10(PSD) vs log10(f) over the active range (a pure 1/f
  background gives χ ≈ 1); ``aperiodic_offset`` is the fit intercept.
* ``hjorth_mobility`` = sqrt(var(x')/var(x)) with x' ≈ diff(x)·fs
  (radians/second; for a sine of frequency f it equals 2πf);
  ``hjorth_complexity`` = mobility of the derivative ÷ mobility of the
  signal (dimensionless); ``line_length_per_s`` =
  Σ|Δx| / duration (sensor units / s); ``spectral_entropy`` = normalized
  Shannon entropy of the PSD over the active range (dimensionless,
  0 ≈ pure tone, → 1 ≈ flat).
* Beta bursts: envelope of the beta_high band-passed signal; threshold =
  the recording's own REST-block envelope 75th percentile (reference
  data only — no target, no fold information); occupancy = fraction of
  evaluated samples above threshold; bursts are above-threshold runs of
  at least 50 ms; ``beta_burst_n`` counts them so the duration/median
  ratio of a burst-free block reads as 0-with-count-0, not a measured 0.
* DELTA blocks: power features as log10(MOVE) − log10(REST) within the
  same recording (block-averaged first); dynamics features as signed
  native-unit differences.
* Aggregation: median over a channel's OK blocks, then (region map
  present) median over each region's OK channels, else median over all
  OK channels plus an ``n_ok`` count covariate.

Feature naming matches the contract's ablation blocks (§11 of
``docs/followon_protocol.md``): ``{band}_log_relative_power``,
``{band}_power_aperiodic_adjusted``, ``aperiodic_exponent``,
``aperiodic_offset``, ``{band}_log_ratio_power`` (DELTA), dynamics
names per the contract table, HFO as ``hfo_power``/``hfo_peak_freq``
when the cohort-uniform HFO gate permits.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch

from . import protocol
from .errors import DataValidationError
from .hfo import extract_channel_features
from .kinematics import parse_restmove_blocks
from .signal_qc import (
    active_top_edge,
    bandpass,
    bipolar_rereference,
    common_median_reference,
    detect_line_frequency,
    notch_line_noise,
    qc_channel_block,
    qc_summary,
    qc_windows,
    recording_diff_mad,
    validate_fs,
)

WELCH_SEGMENT_SECONDS: float = 1.0


# ---------------------------------------------------------------------------
# PSD and spectral transforms
# ---------------------------------------------------------------------------


def welch_psd(signal: np.ndarray, fs: float, segment_s: float = WELCH_SEGMENT_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Contract-approved Welch PSD (1 s Hamming segments, 50 % overlap).

    Returns ``(freqs, psd)`` with the PSD in sensor units²/Hz.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    nperseg = min(int(segment_s * fs), signal.size)
    if nperseg < 8:
        raise DataValidationError(
            f"signal too short for Welch PSD ({signal.size} samples at fs={fs})"
        )
    return welch(signal, fs=fs, window="hamming", nperseg=nperseg, noverlap=nperseg // 2)


def band_mask(freqs: np.ndarray, low: float, high: float, top_edge: float) -> np.ndarray:
    """Frozen band-membership convention: half-open [low, high) over bin
    centers, with the topmost active edge inclusive."""
    freqs = np.asarray(freqs, dtype=float)
    mask = (freqs >= low) & (freqs < high)
    if high == top_edge:
        mask |= freqs == top_edge
    return mask


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float,
               top_edge: float) -> float:
    """Welch PSD integral over the band — units (sensor units)².

    Raises:
        DataValidationError: When no PSD bin falls inside the band.
    """
    mask = band_mask(freqs, low, high, top_edge)
    if not mask.any():
        raise DataValidationError(f"no PSD bins inside band [{low}, {high})")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def aperiodic_fit(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float,
                  top_edge: float) -> dict:
    """Dependency-free aperiodic 1/f estimate over the active range.

    OLS of log10(PSD) vs log10(f) over the bins in [f_low, f_high];
    ``exponent`` = −slope (so PSD ∝ 1/f^χ with χ ≈ 1 for a pure 1/f
    background), ``offset`` = intercept. ``predict(f)`` returns the
    fitted PSD at frequency ``f`` (sensor units²/Hz).
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if int(mask.sum()) < 2:
        raise DataValidationError("aperiodic fit needs at least 2 PSD bins in range")
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, intercept = np.polyfit(log_f, log_p, 1)
    exponent = float(-slope)

    def predict(f: float) -> float:
        return float(10.0 ** (intercept - exponent * np.log10(f)))

    return {"exponent": exponent, "offset": float(intercept), "predict": predict}


def aperiodic_adjusted_band_power(freqs: np.ndarray, psd: np.ndarray, low: float,
                                  high: float, top_edge: float, aperiodic: dict) -> float:
    """Mean in-band PSD ÷ aperiodic-predicted PSD at the band center —
    dimensionless periodic surplus above the 1/f background."""
    mask = band_mask(freqs, low, high, top_edge)
    if not mask.any():
        raise DataValidationError(f"no PSD bins inside band [{low}, {high})")
    center = float(np.mean(freqs[mask]))
    predicted = aperiodic["predict"](center)
    if predicted <= 0:
        raise DataValidationError("aperiodic fit predicted non-positive PSD")
    return float(np.mean(psd[mask]) / predicted)


def spectral_features(freqs: np.ndarray, psd: np.ndarray, bands_resolved: dict,
                      top_edge: float) -> dict:
    """Per-band spectral features for one channel-block, per the frozen
    transforms, plus the recording-level aperiodic pair.

    Keys: ``{band}_power``, ``{band}_relative_power``,
    ``{band}_log_relative_power``, ``{band}_power_aperiodic_adjusted``
    for every active band (standard bands + gamma_high when active), and
    ``aperiodic_exponent`` / ``aperiodic_offset``.
    """
    bands: dict[str, tuple[float, float]] = {
        **bands_resolved["bands_active"],
        **({"gamma_high": tuple(bands_resolved["gamma_high"])}
           if bands_resolved.get("gamma_high") else {}),
    }
    active_range = (4.0, top_edge)
    total = band_power(freqs, psd, active_range[0], active_range[1], top_edge)
    if total <= 0:
        raise DataValidationError("total in-range PSD power is non-positive")
    aperiodic = aperiodic_fit(freqs, psd, active_range[0], active_range[1], top_edge)
    features: dict = {
        "aperiodic_exponent": aperiodic["exponent"],
        "aperiodic_offset": aperiodic["offset"],
    }
    for name, (low, high) in bands.items():
        power = band_power(freqs, psd, low, high, top_edge)
        relative = power / total
        features[f"{name}_power"] = power
        features[f"{name}_relative_power"] = relative
        features[f"{name}_log_relative_power"] = float(np.log10(relative)) if relative > 0 else None
        features[f"{name}_power_aperiodic_adjusted"] = aperiodic_adjusted_band_power(
            freqs, psd, low, high, top_edge, aperiodic
        )
    return features


# ---------------------------------------------------------------------------
# Dynamics features
# ---------------------------------------------------------------------------


def hjorth_parameters(signal: np.ndarray, fs: float) -> dict:
    """Hjorth mobility and complexity.

    mobility = sqrt(var(x')/var(x)) with x' the time derivative estimated
    as diff(x)·fs — radians per SECOND (a pure sine of f Hz gives 2πf);
    complexity = mobility(x')/mobility(x) — dimensionless (≈ 1 for a
    sinusoid; the fs scale cancels in the ratio). A constant signal has
    zero variance: mobility/complexity are defined as 0.0 there (the
    flatline QC rule flags such blocks anyway — the 0 is a defined
    convention, never a silent repair of real data).
    """
    if not np.isfinite(fs) or fs <= 0:
        raise DataValidationError(f"fs must be a positive finite number, got {fs!r}")
    x = np.asarray(signal, dtype=float).ravel()
    dx = np.diff(x)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(np.diff(dx)))
    mobility = float(np.sqrt(var_dx / var_x) * fs) if var_x > 0 else 0.0
    mobility_dx = float(np.sqrt(var_ddx / var_dx) * fs) if var_dx > 0 else 0.0
    complexity = (mobility_dx / mobility) if mobility > 0 else 0.0
    return {"hjorth_mobility": mobility, "hjorth_complexity": complexity}


def line_length_per_s(signal: np.ndarray, fs: float) -> float:
    """Line length normalized per second — Σ|Δx| / duration, sensor
    units per second."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        raise DataValidationError("line length needs at least 2 samples")
    duration_s = x.size / fs
    return float(np.sum(np.abs(np.diff(x))) / duration_s)


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray, f_low: float,
                     f_high: float, top_edge: float) -> float:
    """Normalized Shannon entropy of the PSD over the active range —
    dimensionless in [0, 1] (0 ≈ single-line spectrum, 1 ≈ flat)."""
    mask = (freqs >= max(f_low, 0.0)) & (freqs <= min(f_high, top_edge))
    weights = psd[mask]
    total = float(weights.sum())
    if total <= 0 or weights.size < 2:
        raise DataValidationError("spectral entropy needs positive power over >= 2 bins")
    p = weights / total
    entropy = float(-np.sum(p * np.log(p)))
    return float(entropy / np.log(p.size))


def beta_burst_features(
    eval_signal: np.ndarray,
    reference_signal: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
    percentile: float = protocol.BURST_THRESHOLD_PERCENTILE,
    min_duration_ms: float = protocol.BURST_MIN_DURATION_MS,
) -> dict:
    """Beta-burst occupancy, duration, and amplitude per the frozen rule.

    The threshold is the 75th percentile of the Hilbert envelope of the
    *reference* signal (the recording's own REST blocks — reference data
    only: no target information, no fold membership, no held-out
    influence). Occupancy is the fraction of evaluated samples above the
    threshold; bursts are above-threshold runs of at least
    ``min_duration_ms``; the amplitude summary is the median
    envelope/threshold ratio during bursts (dimensionless).

    ``beta_burst_n`` lets consumers distinguish "no bursts detected"
    (n=0) from measured values.

    Raises:
        DataValidationError: When the reference envelope is degenerate
            (threshold not strictly positive) — never silently repaired.
    """
    band = band or protocol.STANDARD_BANDS_HZ["beta_high"]
    ref_env = np.abs(hilbert(bandpass(reference_signal, fs, band[0], band[1])))
    eval_env = np.abs(hilbert(bandpass(eval_signal, fs, band[0], band[1])))
    threshold = float(np.percentile(ref_env, percentile))
    if not np.isfinite(threshold) or threshold <= 0:
        raise DataValidationError(
            "beta-burst threshold is degenerate (reference envelope has no "
            "positive variation); refusing to fabricate burst statistics"
        )
    above = eval_env > threshold
    occupancy = float(np.mean(above))
    # Contiguous above-threshold runs, measured in samples.
    run_bounds = np.flatnonzero(np.diff(np.concatenate(([False], above, [False])).astype(int)))
    starts, ends = run_bounds[0::2], run_bounds[1::2]
    min_samples = int(round(min_duration_ms / 1000.0 * fs))
    bursts = [(s, e) for s, e in zip(starts, ends, strict=True) if (e - s) >= min_samples]
    if bursts:
        durations_ms = [(e - s) / fs * 1000.0 for s, e in bursts]
        excess = np.concatenate([eval_env[s:e] / threshold for s, e in bursts])
        mean_duration = float(np.mean(durations_ms))
        median_ratio = float(np.median(excess))
    else:
        mean_duration = 0.0
        median_ratio = 0.0
    return {
        "beta_burst_occupancy": occupancy,
        "beta_burst_mean_duration_ms": mean_duration,
        "beta_burst_median_amplitude_ratio": median_ratio,
        "beta_burst_n": len(bursts),
        "burst_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Channel-block feature extraction
# ---------------------------------------------------------------------------


def extract_block_features(
    block: np.ndarray, fs: float, bands_resolved: dict, top_edge: float,
    burst_reference: np.ndarray | None = None,
) -> dict:
    """All frozen features for one QC-OK channel-block (a single channel
    over a single REST or MOVE block).

    Returns spectral features (:func:`spectral_features`), dynamics
    (:func:`hjorth_parameters`, ``line_length_per_s``,
    ``spectral_entropy``), beta bursts (threshold from
    ``burst_reference`` — the REST signal — when given), and the legacy
    HFO block when ``bands_resolved['hfo_block_feasible']`` is true and
    the block passes its extraction QC (the HFO record keeps its own
    ``hfo_qc_*`` fields; a failed HFO extraction nulls the two HFO
    values without touching the other features).
    """
    features = spectral_features(*welch_psd(block, fs), bands_resolved, top_edge)
    features.update(hjorth_parameters(block, fs))
    features["line_length_per_s"] = line_length_per_s(block, fs)
    features["spectral_entropy"] = spectral_entropy(
        *welch_psd(block, fs), 4.0, top_edge, top_edge
    )
    if burst_reference is not None:
        features.update(beta_burst_features(block, burst_reference, fs))
    if bands_resolved.get("hfo_block_feasible"):
        from .config import HFOConfig

        hfo = extract_channel_features(block, fs, HFOConfig(
            low_hz=protocol.HFO_BAND_HZ[0], high_hz=protocol.HFO_BAND_HZ[1]
        ))
        features["hfo_power"] = hfo["hfo_power"]
        features["hfo_peak_freq"] = hfo["hfo_peak_freq"]
        features["hfo_qc_status"] = hfo["qc_status"]
        features["hfo_qc_reason"] = hfo["qc_reason"]
    return features




def _median_of_numeric(values: list) -> float | None:
    """Median over the finite numeric entries of ``values`` — None when
    there are none (never a fabricated zero, never a string median)."""
    numbers = [float(v) for v in values
               if isinstance(v, (int, float, np.integer, np.floating))
               and not isinstance(v, bool) and np.isfinite(v)]
    return float(np.median(numbers)) if numbers else None


def build_subject_level_features(
    ecog: np.ndarray,
    fs: float,
    restmove_segs: np.ndarray,
    bands_resolved: dict,
    adjacency: list[tuple[int, int]] | None = None,
    region_map: dict | None = None,
    include_move_windows: bool = False,
) -> dict:
    """REST / MOVE / DELTA feature blocks for one recording (subject-level
    arms), with QC counts and reasons — no participant identifiers.

    Pipeline, per the frozen contract:

    1. resolve the block layout (0-based half-open, kind 0/1);
    2. validate fs against the topmost active band edge;
    3. per-channel QC (non-finite → failed, counted, excluded — never
       zero-filled);
    4. rereference: bipolar over the explicit adjacency pairs when
       given, else the contract fallback (common median across OK
       channels);
    5. detect the line frequency per recording and notch it (+ harmonics)
       on every channel;
    6. per OK channel: REST features with the burst threshold from the
       concatenated REST signal; MOVE features against the same
       threshold; per-block QC (flatline/saturation/transients) fails a
       block — its features stay ``None`` and are counted;
    7. aggregate per channel (median over OK blocks), then per region
       (median over the region's OK channels when ``region_map`` is
       given) else over all OK channels + ``n_ok`` counts;
    8. DELTA = log10-ratio for power features, signed difference for
       dynamics.

    Returns ``{"rest": {...}, "move": {...}, "delta": {...},
    "qc": {...}, "meta": {...}}``. Feature values are floats or ``None``
    (never NaN, never fabricated zeros); ``delta`` is ``None`` unless
    both REST and MOVE values exist for a feature.

    Raises:
        DataValidationError: On malformed inputs (shape, blocks, fs
            below the active band ceiling).
    """
    ecog = np.asarray(ecog, dtype=float)
    if ecog.ndim != 2:
        raise DataValidationError(f"ecog must be 2-D (channels × samples); got {ecog.shape}")
    if not bands_resolved.get("bands_active") and not bands_resolved.get("gamma_high"):
        raise DataValidationError("bands_resolved has no active bands for this fs")
    top_edge = active_top_edge(bands_resolved)
    validate_fs(fs, top_edge)
    blocks = parse_restmove_blocks(restmove_segs)
    rest_bounds = [(s, e) for s, e, kind in blocks if kind == "rest"]
    move_bounds = [(s, e) for s, e, kind in blocks if kind == "move"]
    if not rest_bounds:
        raise DataValidationError("no REST blocks in restmove_segs — the REST/DELTA arms "
                                  "require rest; refusing to fabricate features")
    if not move_bounds:
        raise DataValidationError("no MOVE blocks in restmove_segs — the DELTA arm requires "
                                  "move; refusing to fabricate features")
    end_clipped = [(s, min(e, ecog.shape[1])) for s, e in rest_bounds + move_bounds]
    if any(e <= s for s, e in end_clipped):
        raise DataValidationError("a block is empty after clipping to the recording length")

    ok_mask = np.isfinite(ecog).all(axis=1)
    if not ok_mask.any():
        raise DataValidationError("every channel is non-finite; nothing to extract")

    # Rereference (bipolar when adjacency is provided; contract fallback
    # otherwise) and notch the detected line frequency per recording.
    if adjacency:
        rereferenced, channel_labels = bipolar_rereference(ecog, adjacency)
        # A bipolar pair is OK only if both source channels are OK.
        pair_ok = np.array([
            bool(ok_mask[a] and ok_mask[b]) for a, b in adjacency
        ])
        ok_after_ref = pair_ok
    else:
        rereferenced = common_median_reference(ecog, ok_mask)
        channel_labels = [str(i) for i in range(ecog.shape[0])]
        ok_after_ref = ok_mask
    if not ok_after_ref.any():
        raise DataValidationError(
            "no OK channel (or bipolar pair) survives rereferencing; refusing "
            "to fabricate features from an empty channel set"
        )
    line = detect_line_frequency(
        np.mean(rereferenced[ok_after_ref], axis=0), fs
    )
    if line["line_freq_hz"] is not None:
        rereferenced = np.vstack([
            notch_line_noise(channel, fs, line["line_freq_hz"]) if ok else channel
            for channel, ok in zip(rereferenced, ok_after_ref, strict=True)
        ])

    diff_mad = recording_diff_mad(rereferenced[ok_after_ref])

    def block_signal(start: int, end: int) -> np.ndarray:
        return rereferenced[:, start:end]

    # REST burst reference per channel: concatenated REST blocks.
    rest_concat = np.concatenate([block_signal(s, e) for s, e in rest_bounds], axis=1)

    def extract_state(bounds: list[tuple[int, int]], state: str) -> tuple[dict, dict, list]:
        """Per-channel features for one state. Returns (per_channel median
        features, per_channel QC, window records for MOVE)."""
        per_channel: dict[str, dict] = {}
        channel_qc: dict[str, dict] = {}
        window_records: list = []
        for idx, label in enumerate(channel_labels):
            if not ok_after_ref[idx]:
                channel_qc[label] = {"qc_status": "failed",
                                     "qc_reason": "source channel non-finite"}
                continue
            block_records = []
            block_features: list[dict] = []
            for start, end in bounds:
                block = rereferenced[idx, start:end]
                record = qc_channel_block(block, diff_mad)
                if record["qc_status"] == "ok":
                    try:
                        block_features.append(extract_block_features(
                            block, fs, bands_resolved, top_edge,
                            # Burst threshold always comes from the
                            # recording's own REST blocks (frozen rule):
                            # REST occupancy is REST-vs-REST, MOVE is
                            # MOVE against the REST threshold.
                            burst_reference=rest_concat[idx],
                        ))
                    except DataValidationError:
                        record = {"qc_status": "failed",
                                  "qc_reason": "feature extraction failed"}
                block_records.append(record)
            channel_qc[label] = qc_summary(block_records)
            if state == "move" and include_move_windows:
                bands_for_qc = {
                    **bands_resolved["bands_active"],
                    **({"gamma_high": tuple(bands_resolved["gamma_high"])}
                       if bands_resolved.get("gamma_high") else {}),
                }
                for start, end in bounds:
                    move_block = rereferenced[idx, start:end]
                    z_scores = implausible_power_z(
                        window_band_powers(move_block, fs, bands_for_qc, top_edge),
                        window_band_powers(rest_concat[idx], fs, bands_for_qc, top_edge),
                    )
                    window_records.extend(
                        {**record, "channel": label}
                        for record in qc_windows(
                            move_block, fs, diff_mad,
                            reference_band_power_z=z_scores,
                        )
                    )
            if block_features:
                keys = [k for k in block_features[0].keys()
                        if k not in ("hfo_qc_status", "hfo_qc_reason")]
                per_channel[label] = {
                    key: _median_of_numeric(
                        [f.get(key) for f in block_features]
                    )
                    for key in keys
                }
        return per_channel, channel_qc, window_records

    rest_per_channel, rest_qc, _ = extract_state(rest_bounds, "rest")
    move_per_channel, move_qc, window_records = extract_state(move_bounds, "move")

    rest = aggregate_channel_values(rest_per_channel, region_map)
    move = aggregate_channel_values(move_per_channel, region_map)

    # DELTA per the frozen rules: band power as the log10-ratio built
    # from the log-relative features (a plain difference of logs — the
    # log10 of an already-logged value would be a double log);
    # aperiodic-adjusted surplus and hfo power as log10 changes of the
    # positive values; dynamics as signed native-unit differences.
    delta: dict = {}
    signed_diff_names = (
        "aperiodic_exponent", "aperiodic_offset", "hjorth_mobility",
        "hjorth_complexity", "line_length_per_s", "spectral_entropy",
        "beta_burst_occupancy", "beta_burst_mean_duration_ms",
        "beta_burst_median_amplitude_ratio", "hfo_peak_freq",
    )
    log_ratio_names = ("_power_aperiodic_adjusted",)
    for name, rest_value in rest["features"].items():
        move_value = move["features"].get(name)
        if name.endswith("_log_relative_power"):
            delta[name.replace(
                "_log_relative_power", "_log_ratio_power"
            )] = (
                None if rest_value is None or move_value is None
                else float(move_value - rest_value)
            )
        elif name.endswith(log_ratio_names) or name == "hfo_power":
            delta[f"{name}_log_ratio" if name != "hfo_power"
                  else "hfo_power_log_ratio"] = (
                None if rest_value is None or move_value is None
                or rest_value <= 0 or move_value <= 0
                else float(np.log10(move_value) - np.log10(rest_value))
            )
        elif name in signed_diff_names:
            delta[f"{name}_delta"] = (
                None if rest_value is None or move_value is None
                else float(move_value - rest_value)
            )
    qc = {
        "rest_channel_qc": rest_qc,
        "move_channel_qc": move_qc,
        "n_channels_ok_after_reference": int(np.count_nonzero(ok_after_ref)),
        "line_noise": line,
        "region_map_present": region_map is not None,
        "bipolar": adjacency is not None,
    }
    meta = {
        "fs": float(fs),
        "top_edge_hz": top_edge,
        "bands_active": sorted(bands_resolved["bands_active"]),
        "gamma_high": bands_resolved.get("gamma_high"),
        "hfo_block_feasible": bool(bands_resolved.get("hfo_block_feasible")),
        "n_rest_blocks": len(rest_bounds),
        "n_move_blocks": len(move_bounds),
    }
    result = {"rest": rest, "move": move, "delta": delta, "qc": qc, "meta": meta}
    if include_move_windows:
        result["move_window_qc"] = qc_summary(window_records)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def window_band_powers(
    block: np.ndarray, fs: float, bands: dict[str, tuple[float, float]],
    top_edge: float,
    window_s: float = protocol.MOVE_WINDOW_S,
    update_s: float = protocol.MOVE_UPDATE_S,
) -> dict[str, list[float]]:
    """Per-window band powers over one channel-block (MOVE-window QC
    reference stream) — band power per window in (sensor units)²."""
    from .signal_qc import window_grid

    block = np.asarray(block, dtype=float).ravel()
    grid = window_grid(0, block.size, fs, window_s, update_s)
    powers: dict[str, list[float]] = {name: [] for name in bands}
    for start, end in grid:
        window = block[start:end]
        if window.size < 8 or not np.isfinite(window).all():
            for name in bands:
                powers[name].append(float("nan"))
            continue
        freqs, psd = welch_psd(window, fs, segment_s=window_s)
        for name, (low, high) in bands.items():
            try:
                powers[name].append(band_power(freqs, psd, low, high, top_edge))
            except DataValidationError:
                powers[name].append(float("nan"))
    return powers


def implausible_power_z(move_powers: dict[str, list[float]],
                        rest_powers: dict[str, list[float]]) -> dict[str, list[float]]:
    """Per-window z-scores of MOVE band power against the recording's own
    REST-window distribution (frozen implausible-band-power rule) —
    target-independent reference data only."""
    z: dict[str, list[float]] = {}
    for band, move_values in move_powers.items():
        rest_values = np.asarray(rest_powers.get(band, []), dtype=float)
        rest_finite = rest_values[np.isfinite(rest_values)]
        mean = float(np.mean(rest_finite)) if rest_finite.size else 0.0
        std = float(np.std(rest_finite)) if rest_finite.size else 0.0
        band_z: list[float] = []
        for value in move_values:
            if not np.isfinite(value) or std <= 0:
                band_z.append(float("nan"))
            else:
                band_z.append(float((value - mean) / std))
        z[band] = band_z
    return z


def aggregate_channel_values(
    per_channel: dict[str, dict], region_map: dict | None = None
) -> dict:
    """Robust aggregation per the frozen contract.

    Per feature: median over channels with a value (channels with all
    ``None`` features are excluded and visible in ``n_channels_with_value``
    vs ``n_channels_present``). With ``region_map`` (channel label → one
    of ``protocol.REGION_LABELS``): one median per region under
    ``regions[region][feature]``; the overall fallback median is still
    returned under ``features``.
    """
    labels = sorted(per_channel)
    feature_keys: set[str] = set()
    for values in per_channel.values():
        feature_keys.update(values)
    aggregated: dict = {"features": {}, "n_channels_present": len(labels),
                        "n_channels_with_value": 0}
    if region_map:
        aggregated["regions"] = {}
    for key in sorted(feature_keys):
        values = [per_channel[label][key] for label in labels
                  if per_channel[label].get(key) is not None]
        finite = [v for v in values if np.isfinite(v)]
        aggregated["features"][key] = float(np.median(finite)) if finite else None
    aggregated["n_channels_with_value"] = sum(
        1 for label in labels if any(
            per_channel[label].get(key) is not None for key in feature_keys
        )
    )
    if region_map:
        unknown = sorted({label for label in labels if label not in region_map})
        if unknown:
            raise DataValidationError(
                f"region_map is missing channels: {unknown[:5]}"
                f"{'…' if len(unknown) > 5 else ''}"
            )
        for region in protocol.REGION_LABELS:
            region_labels = [label for label in labels
                             if region_map.get(label) == region]
            region_features: dict = {}
            for key in sorted(feature_keys):
                values = [per_channel[label][key] for label in region_labels
                          if per_channel[label].get(key) is not None]
                region_features[key] = (
                    float(np.median([v for v in values if np.isfinite(v)]))
                    if values else None
                )
            aggregated["regions"][region] = {
                "features": region_features,
                "n_channels": len(region_labels),
            }
    return aggregated

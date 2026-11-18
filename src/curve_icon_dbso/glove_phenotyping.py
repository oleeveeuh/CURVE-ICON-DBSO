"""Prespecified glove phenotyping for the follow-on protocol (frozen
contract v1.0.1): secondary kinematic targets derived from MOVE blocks of
individual glove channels, with the primary ``tap_rate_mean`` passed
through from the frozen ``curve_icon_dbso.kinematics`` implementation —
UNCHANGED, and never re-fit or re-scored here.

Frozen definitions (contract §2, ``curve_icon_dbso.protocol``):

* **Detection.** Per MOVE block × glove channel: the channel-block
  contributes only if its dominant 0.5–4.0 Hz Welch peak carries
  ≥ 0.20 of in-band power (the frozen prominence rule — reused from
  ``kinematics._channel_tap_rate``, the single source of truth, so the
  secondary targets can never drift from the primary's contribution
  rule). Cycle detection runs on the 0.5–4 Hz zero-phase Butterworth
  (order 4, ``protocol.TAP_DETECTION_FILTER_ORDER``) filtered channel;
  peaks are local maxima separated by at least 0.15 s
  (``protocol.TAP_MIN_PEAK_SEPARATION_S``); orientation (peaks vs
  troughs — sensor inversion) is chosen by the larger peak count, ties
  upright. All settings are frozen constants — nothing here is
  optimized against ECoG features, model performance, or targets.
* **`tap_interval_cv`** (secondary): per contributing channel-block with
  ≥ 5 intervals — SD/mean of the successive inter-cycle intervals in
  seconds; dimensionless; recording value = median across contributing
  channel-blocks; ≥ 2 contributing channel-blocks required.
* **`tap_vigor_index`** (secondary): per contributing channel-block,
  ``RMS(diff(filtered)) / (2π · f_tap · RMS(filtered))`` — dimensionless
  and amplitude-independent (a pure sinusoid scores 1.0 by construction
  regardless of its amplitude); median across contributing channel-blocks.
* **`tap_sequence_slope`** (secondary): per contributing channel-block
  with ≥ 5 cycles — OLS slope of per-cycle peak amplitude vs cycle
  index, normalized by the mean amplitude — dimensionless per-cycle
  fractional change (negative = decrement); median across contributing
  channel-blocks.
* **`tap_amplitude`** (within-subject-only): median RMS of the filtered
  channel-blocks — sensor units; recorded for QC only because glove
  sensors are uncalibrated across subjects; excluded from all
  cross-subject modeling.
* **`glove_vigor_envelope_z`** (MOVE-arm continuous target): per frozen
  definition — per QC-OK channel: band-pass 0.5–4 Hz, Hilbert |envelope|
  (inversion-invariant by construction), average across OK channels,
  100 ms moving-mean smooth, mean within each 100 ms update bin, then
  per-recording z-score across MOVE samples (dimensionless; the
  standardization uses only this recording's glove data). Computed per
  MOVE block; windows never cross block boundaries.

Purity and leakage guarantees, structurally enforced: every function
here takes **glove arrays only** — no ECoG argument exists to read, no
future neural data can enter; no subject identity or evaluation-fold
membership is accepted or returned. Targets may legitimately summarize
a whole block (they are the prediction targets, not model inputs); the
causality requirement applies to the neural windows that decode them
(contract §6).

Failed channels/blocks are counted with machine-readable reasons and
never imputed or zero-filled; a recording whose contributing
channel-block count is below the frozen floor of 2 gets ``None``
targets with an explicit reason.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert

from . import protocol
from .errors import DataValidationError
from .kinematics import _channel_tap_rate, parse_restmove_blocks
from .signal_qc import bandpass

#: Frozen floors (mirrors of ``protocol`` constants; pinned by tests).
DETECTION_BAND_HZ = protocol.TAP_DETECTION_BAND_HZ
FILTER_ORDER = protocol.TAP_DETECTION_FILTER_ORDER
MIN_PEAK_SEPARATION_S = protocol.TAP_MIN_PEAK_SEPARATION_S
MIN_CYCLES_PER_BLOCK = protocol.TAP_MIN_CYCLES_PER_BLOCK
MIN_CONTRIBUTING_BLOCKS = protocol.TAP_MIN_CHANNEL_BLOCKS
ENVELOPE_UPDATE_S = protocol.MOVE_UPDATE_S

_ENDPOINT_PRIMARY = "primary"
_ENDPOINT_SECONDARY = "secondary"
_ENDPOINT_WITHIN_SUBJECT = "within_subject_only"


# ---------------------------------------------------------------------------
# Per-channel-block extraction
# ---------------------------------------------------------------------------


def _cycles(block_filtered: np.ndarray, fs: float) -> tuple[np.ndarray, str]:
    """Cycle peak indices on the filtered block, honoring the frozen
    minimum separation and the inversion rule (orientation = larger peak
    count; ties upright)."""
    distance = max(int(round(MIN_PEAK_SEPARATION_S * fs)), 1)
    upright = find_peaks(block_filtered, distance=distance)[0]
    inverted = find_peaks(-block_filtered, distance=distance)[0]
    if inverted.size > upright.size:
        return inverted, "inverted"
    return upright, "upright"


def channel_block_kinematics(
    block: np.ndarray, fs: float
) -> dict:
    """All frozen kinematic quantities for one glove channel-block.

    Returns a record with the frozen prominence verdict (reused from the
    primary's rule), the detected cycles, and the secondary per-block
    values (``None`` where the block does not meet a target's cycle
    floor). Never imputes: exclusions carry reasons.
    """
    block = np.asarray(block, dtype=float).ravel()
    record: dict = {
        "n_samples": int(block.size),
        "contributing": False,
        "reasons": [],
        "f_tap_hz": None,
        "n_cycles": None,
        "orientation": None,
        "tap_interval_cv": None,
        "tap_vigor_index": None,
        "tap_sequence_slope": None,
        "tap_amplitude_rms": None,
    }
    if block.size == 0 or not np.isfinite(block).all():
        record["reasons"].append("non_finite_or_empty")
        return record
    prominence = _channel_tap_rate(block, fs)
    if prominence is None:
        record["reasons"].append("below_prominence_floor")
        return record
    f_tap, _share = prominence
    record["f_tap_hz"] = f_tap
    record["contributing"] = True

    if block.size <= 3 * (FILTER_ORDER + 1):
        record["reasons"].append("too_short_for_bandpass")
        return record
    filtered = bandpass(block, fs, DETECTION_BAND_HZ[0], DETECTION_BAND_HZ[1],
                        order=FILTER_ORDER)
    peaks, orientation = _cycles(filtered, fs)
    record["orientation"] = orientation
    record["n_cycles"] = int(peaks.size)

    record["tap_amplitude_rms"] = float(np.sqrt(np.mean(filtered ** 2)))
    omega = 2.0 * np.pi * f_tap
    rms_filtered = float(np.sqrt(np.mean(filtered ** 2)))
    # diff is per sample; scale to per-second velocity units so a pure
    # sinusoid scores vigor ≈ 1.0 regardless of amplitude (frozen
    # dimensionless definition).
    rms_diff_per_s = float(np.sqrt(np.mean(np.diff(filtered) ** 2))) * fs
    if rms_filtered > 0 and omega > 0:
        record["tap_vigor_index"] = float(
            rms_diff_per_s / (omega * rms_filtered)
        )

    if peaks.size >= MIN_CYCLES_PER_BLOCK:
        peak_times = peaks / fs
        intervals = np.diff(peak_times)
        mean_interval = float(np.mean(intervals))
        if mean_interval > 0:
            record["tap_interval_cv"] = float(np.std(intervals) / mean_interval)
        amplitudes = filtered[peaks]
        cycle_index = np.arange(peaks.size, dtype=float)
        mean_amp = float(np.mean(amplitudes))
        if mean_amp != 0 and peaks.size >= 2:
            slope = float(np.polyfit(cycle_index, amplitudes, 1)[0])
            record["tap_sequence_slope"] = float(slope / mean_amp)
    else:
        record["reasons"].append(
            f"cycle_floor_not_met ({peaks.size} < {MIN_CYCLES_PER_BLOCK})"
        )
    return record


# ---------------------------------------------------------------------------
# Recording-level targets
# ---------------------------------------------------------------------------


def _median_or_none(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(numbers)) if numbers else None


def glove_move_envelope(
    glove: np.ndarray, fs: float, restmove_segs: np.ndarray,
) -> dict:
    """The frozen ``glove_vigor_envelope_z`` MOVE-arm target stream.

    Per QC-OK (all-finite) channel: band-pass, Hilbert |envelope|,
    average across OK channels, 100 ms moving-mean smooth (edge-
    renormalized), mean within each 100 ms update bin, per-recording
    z-score across MOVE samples. Computed per MOVE block; the returned
    stream is the concatenation over blocks in order, with ``block_ids``
    marking block membership. A recording with no finite channel or a
    degenerate (zero-variance) envelope yields ``stream=None`` plus the
    machine-readable reason — never a zero-filled stream.
    """
    glove = np.asarray(glove, dtype=float)
    blocks = parse_restmove_blocks(restmove_segs)
    move_bounds = [(s, e) for s, e, kind in blocks if kind == "move"]
    result: dict = {"stream": None, "block_ids": None, "update_s": ENVELOPE_UPDATE_S,
                    "n_samples": None, "qc_status": "failed", "qc_reason": ""}
    if not move_bounds:
        result["qc_reason"] = "no MOVE blocks"
        return result
    finite = glove[np.isfinite(glove).all(axis=1)]
    if finite.size == 0:
        result["qc_reason"] = "no finite glove channels"
        return result
    per_block: list[np.ndarray] = []
    block_ids: list[int] = []
    for block_number, (start, end) in enumerate(move_bounds):
        end = min(end, glove.shape[1])
        segment = glove[:, start:end]
        ok = segment[np.isfinite(segment).all(axis=1)]
        if ok.shape[0] == 0:
            result["qc_reason"] = f"move block {block_number} has no finite channels"
            return result
        envelopes = []
        for channel in ok:
            filtered = bandpass(channel, fs, DETECTION_BAND_HZ[0],
                                DETECTION_BAND_HZ[1], order=FILTER_ORDER)
            envelopes.append(np.abs(hilbert(filtered)))
        envelope = np.mean(envelopes, axis=0)
        window = max(int(round(ENVELOPE_UPDATE_S * fs)), 1)
        kernel = np.ones(window) / window
        smoothed = np.convolve(envelope, kernel, mode="same")
        # Edge renormalization: divide by the per-sample kernel mass.
        mass = np.convolve(np.ones_like(envelope), kernel, mode="same")
        smoothed = smoothed / np.maximum(mass, 1e-12)
        update = max(int(round(ENVELOPE_UPDATE_S * fs)), 1)
        n_bins = smoothed.size // update
        binned = smoothed[: n_bins * update].reshape(n_bins, update).mean(axis=1)
        per_block.append(binned)
        block_ids.extend([block_number] * n_bins)
    stream = np.concatenate(per_block)
    std = float(np.std(stream))
    if not np.isfinite(std) or std <= 0:
        result["qc_reason"] = "degenerate envelope (zero variance)"
        return result
    result.update({
        "stream": (stream - float(np.mean(stream))) / std,
        "block_ids": np.asarray(block_ids, dtype=int),
        "n_samples": int(stream.size),
        "qc_status": "ok",
    })
    return result


def extract_glove_targets(
    glove: np.ndarray, fs_glove: float, restmove_segs: np.ndarray,
) -> dict:
    """Primary + secondary + within-subject-only glove endpoints for one
    recording, with QC counts — the recording-level view of the frozen
    contract §2.

    The primary section is the frozen ``kinematics.tap_rate_for_recording``
    result, passed through verbatim (this module never modifies it). The
    secondary/within-subject sections aggregate the contributing
    channel-blocks; a recording below the frozen floor of 2 contributing
    channel-blocks keeps its primary verdict and gets ``None`` secondary
    targets with the explicit reason. All outputs are keyed by position
    (block/channel indices) — no subject identity anywhere.

    Raises:
        DataValidationError: On a non-finite or non-positive fs, a glove
            rate band outside Nyquist, or malformed segments (inherited
            from the frozen parsers).
    """
    from .kinematics import tap_rate_for_recording

    if not np.isfinite(fs_glove) or fs_glove <= 0:
        raise DataValidationError(
            f"Glove sampling rate must be positive and finite, got {fs_glove!r}"
        )
    glove = np.asarray(glove, dtype=float)
    primary = tap_rate_for_recording(glove, fs_glove, restmove_segs)
    if glove.ndim == 2 and glove.shape[0] > 0 and np.isfinite(glove).all():
        blocks = parse_restmove_blocks(restmove_segs)
        move_bounds = [(s, e) for s, e, kind in blocks if kind == "move"]
        records: list[dict] = []
        for block_number, (start, end) in enumerate(move_bounds):
            end = min(end, glove.shape[1])
            for channel_idx, channel in enumerate(glove[:, start:end]):
                record = channel_block_kinematics(channel, fs_glove)
                record["block"] = block_number
                record["channel"] = int(channel_idx)
                records.append(record)
    else:
        records = []

    contributing = [r for r in records if r["contributing"]]
    secondary: dict = {}
    for target in ("tap_interval_cv", "tap_vigor_index", "tap_sequence_slope"):
        eligible = [r for r in contributing if r[target] is not None]
        n_eligible = len(eligible)
        if n_eligible < MIN_CONTRIBUTING_BLOCKS:
            secondary[target] = {
                "value": None,
                "endpoint": _ENDPOINT_SECONDARY,
                "n_blocks": n_eligible,
                "qc_status": "failed",
                "qc_reason": (
                    f"only {n_eligible} contributing channel-block(s) met the "
                    f"{MIN_CYCLES_PER_BLOCK}-cycle floor; "
                    f"{MIN_CONTRIBUTING_BLOCKS} required"
                ),
            }
        else:
            secondary[target] = {
                "value": _median_or_none([r[target] for r in eligible]),
                "endpoint": _ENDPOINT_SECONDARY,
                "n_blocks": n_eligible,
                "qc_status": "ok",
                "qc_reason": "",
            }

    n_contributing = len(contributing)
    if n_contributing < MIN_CONTRIBUTING_BLOCKS:
        amplitude_block: dict = {
            "value": None, "endpoint": _ENDPOINT_WITHIN_SUBJECT,
            "n_blocks": n_contributing, "qc_status": "failed",
            "qc_reason": f"only {n_contributing} contributing channel-block(s)",
        }
    else:
        amplitude_block = {
            "value": _median_or_none([r["tap_amplitude_rms"] for r in contributing]),
            "endpoint": _ENDPOINT_WITHIN_SUBJECT,
            "n_blocks": n_contributing,
            "qc_status": "ok",
            "qc_reason": "",
        }

    cycle_eligible = [r for r in contributing
                      if r["n_cycles"] is not None
                      and r["n_cycles"] >= MIN_CYCLES_PER_BLOCK]
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record["reasons"]:
            key = reason.split(" (")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1

    return {
        "primary": {
            "tap_rate_mean": primary["tap_rate_mean"],
            "endpoint": _ENDPOINT_PRIMARY,
            "n_contributing": primary["n_tap_contributing"],
            "qc_status": primary["tap_qc_status"],
            "qc_reason": primary["tap_qc_reason"],
            "definition": "frozen kinematics implementation — unchanged",
        },
        "secondary": secondary,
        "within_subject_only": {"tap_amplitude": amplitude_block},
        "move_continuous": glove_move_envelope(glove, fs_glove, restmove_segs),
        "qc": {
            "n_channel_blocks": len(records),
            "n_contributing": n_contributing,
            "n_cycle_eligible": len(cycle_eligible),
            "reliability": (
                float(len(cycle_eligible) / n_contributing)
                if n_contributing else 0.0
            ),
            "reasons": reason_counts,
            "orientations": {
                str(o): sum(1 for r in records if r["orientation"] == o)
                for o in ("upright", "inverted")
            },
        },
    }


# ---------------------------------------------------------------------------
# Sanitized aggregate QC report (template)
# ---------------------------------------------------------------------------


def format_glove_qc_report(outputs: list[dict]) -> str:
    """Sanitized aggregate QC report across recordings.

    Accepts the per-recording dicts from :func:`extract_glove_targets` —
    which carry no subject identity — and prints aggregate inclusion
    counts, per-target availability, and reliability only. The function
    structurally cannot leak participant values: it receives neither
    labels nor per-subject target values (availability is a count; the
    reported medians are aggregate statistics over availability-eligible
    recordings). Cohort medians are findings, not data — publication
    stays governed by the recorded terms (docs/data_access.md).
    """
    n = len(outputs)
    primary_ok = sum(1 for o in outputs if o["primary"]["qc_status"] == "ok")
    lines = [
        "GLOVE PHENOTYPING — SANITIZED AGGREGATE QC (no subject codes, "
        "no per-subject values)",
        f"Recordings evaluated            : {n}",
        f"Primary tap-rate QC passes      : {primary_ok}/{n}",
        "Endpoint separation             : primary | secondary | "
        "within-subject-only (amplitude never modeled cross-subject)",
    ]
    for target in ("tap_interval_cv", "tap_vigor_index", "tap_sequence_slope"):
        available = sum(1 for o in outputs
                        if o["secondary"][target]["qc_status"] == "ok")
        median_value = _median_or_none(
            [o["secondary"][target]["value"] for o in outputs
             if o["secondary"][target]["qc_status"] == "ok"]
        )
        lines.append(
            f"{target:<22}: available {available}/{n}"
            + (f"; cohort median {median_value:.4f}" if median_value is not None else "")
        )
    amplitude_available = sum(
        1 for o in outputs if o["within_subject_only"]["tap_amplitude"]["qc_status"] == "ok"
    )
    lines.append(
        f"{'tap_amplitude (W-S only)':<22}: recorded {amplitude_available}/{n}"
        " (never modeled cross-subject)"
    )
    envelope_ok = sum(1 for o in outputs
                      if o["move_continuous"]["qc_status"] == "ok")
    lines.append(f"{'move_vigor_envelope':<22}: available {envelope_ok}/{n}")
    reliabilities = [o["qc"]["reliability"] for o in outputs]
    median_reliability = _median_or_none(reliabilities)
    if median_reliability is not None:
        lines.append(f"Cohort median cycle reliability : {median_reliability:.3f}")
    total_failures = sum(
        sum(o["qc"]["reasons"].values()) for o in outputs
    )
    lines.append(f"Channel-block QC exclusions     : {total_failures} total")
    return "\n".join(lines)

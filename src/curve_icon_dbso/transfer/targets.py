"""Glove target streams and tap-event extraction (v2 contract §4, §10).

The continuous target mirrors the v1-frozen envelope construction
verbatim (per QC-OK glove channel: 0.5–4 Hz band-pass, Hilbert
envelope, channel average, 100 ms moving-mean smooth with edge
renormalization, mean per 100 ms update bin) — but exposes the RAW
standardized-agnostic stream so v2 can standardize it with
span-appropriate constants (calibration-segment-only in few-shot;
never the final test interval — the frozen leakage rule).

**Span-local construction (v2.1.0, offline retrospective; owner
decision 2026-11-19).** The envelope is computed per MOVE block
CLIPPED to the caller's span: every span's targets are filtered from
that span's raw glove samples only, so no future test sample can
alter a calibration target and vice versa. Zero-phase filtering
remains allowed WITHIN a span and is documented as retrospective.
Both sides of every clipped segment edge are excluded by
``contract.TARGET_EDGE_EXCLUSION_S`` (covers the 0.5 Hz-band-edge
Hilbert transient, the band-pass startup, and the smoothing kernel),
so each kept bin's value depends only on samples strictly interior
to its own span; excluded bins are counted, never silently filled.

Tap events reuse the frozen detection rule (same band, order, minimum
separation, orientation rule) with per-cycle outcomes for Track D.
Physical-unit cross-subject claims are unsupported (recorded UNKNOWN):
amplitude/velocity outcomes are within-person z only, ITI outcomes are
scale-free.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ..errors import DataValidationError
from ..glove_phenotyping import _cycles
from ..kinematics import parse_restmove_blocks
from ..signal_qc import bandpass
from . import contract

#: Frozen tap-detection band (same values as the v1 phenotyping rule).
TAP_BAND_LOW, TAP_BAND_HIGH = 0.5, 4.0
TAP_FILTER_ORDER = 4


def move_envelope_stream(
    glove: np.ndarray, fs: float, blocks: list[tuple[int, int, str]],
    clip: tuple[int, int] | None = None,
) -> dict:
    """Raw (unstandardized) movement-envelope stream on the 100 ms grid.

    ``clip=(lo, hi)`` restricts construction to absolute samples
    ``[lo, hi)`` — the span-local rule (v2.1.0): each MOVE block is
    clipped to the span, filtered from the clipped samples only, and
    bins within ``TARGET_EDGE_EXCLUSION_S`` of a clipped segment edge
    are excluded (counted, never filled). ``clip=None`` means the whole
    recording (used only where no span boundary exists).

    Returns ``{"stream", "block_ids", "bin_starts", "qc_status",
    "qc_reason", "edge_excluded_bins"}`` where ``bin_starts`` are the
    absolute sample index of each update bin's start and ``block_ids``
    identify the MOVE block of each bin (unique within one call).
    Computed per clipped MOVE block; blocks contribute in chronological
    order. A recording with no finite glove channel or a degenerate
    envelope yields ``stream=None`` plus the reason — never a
    zero-filled stream.
    """
    glove = np.asarray(glove, dtype=float)
    lo, hi = (0, glove.shape[1]) if clip is None else (int(clip[0]), int(clip[1]))
    move_bounds = [(s, e) for s, e, kind in blocks if kind == "move"]
    result: dict = {"stream": None, "block_ids": None, "bin_starts": None,
                    "update_s": contract.UPDATE_S, "qc_status": "failed",
                    "qc_reason": "", "edge_excluded_bins": 0, "clip": (lo, hi)}
    if not move_bounds:
        result["qc_reason"] = "no MOVE blocks"
        return result
    finite = glove[np.isfinite(glove).all(axis=1)]
    if finite.size == 0:
        result["qc_reason"] = "no finite glove channels"
        return result
    update = max(int(round(contract.UPDATE_S * fs)), 1)
    margin = int(round(contract.TARGET_EDGE_EXCLUSION_S * fs))
    per_block: list[np.ndarray] = []
    block_ids: list[int] = []
    bin_starts: list[int] = []
    for block_number, (start, end) in enumerate(move_bounds):
        seg_start, seg_end = max(start, lo), min(end, hi)
        if seg_end <= seg_start:
            continue
        segment = glove[:, seg_start:seg_end]
        ok = segment[np.isfinite(segment).all(axis=1)]
        if ok.shape[0] == 0:
            result["qc_reason"] = f"move block {block_number} has no finite channels"
            return result
        envelopes = []
        for channel in ok:
            filtered = bandpass(channel, fs, TAP_BAND_LOW, TAP_BAND_HIGH,
                                order=TAP_FILTER_ORDER)
            envelopes.append(np.abs(hilbert(filtered)))
        envelope = np.mean(envelopes, axis=0)
        window = max(int(round(contract.UPDATE_S * fs)), 1)
        kernel = np.ones(window) / window
        smoothed = np.convolve(envelope, kernel, mode="same")
        mass = np.convolve(np.ones_like(envelope), kernel, mode="same")
        smoothed = smoothed / np.maximum(mass, 1e-12)
        n_bins = smoothed.size // update
        if n_bins == 0:
            result["qc_reason"] = f"move block {block_number} shorter than one update bin"
            return result
        binned = smoothed[: n_bins * update].reshape(n_bins, update).mean(axis=1)
        # Edge exclusion (span-local rule): keep only bins FULLY interior
        # to the clipped segment by the frozen margin on both sides.
        bin_start_samples = seg_start + np.arange(n_bins) * update
        bin_end_samples = bin_start_samples + update
        interior = (bin_start_samples >= seg_start + margin) \
            & (bin_end_samples <= seg_end - margin)
        result["edge_excluded_bins"] += int((~interior).sum())
        per_block.append(binned[interior])
        block_ids.extend([block_number] * int(interior.sum()))
        bin_starts.extend(bin_start_samples[interior].tolist())
    if not per_block or not any(len(b) for b in per_block):
        result["qc_reason"] = "all bins excluded by the span-edge margin"
        return result
    stream = np.concatenate([b for b in per_block if b.size])
    result.update({
        "stream": stream,
        "block_ids": np.asarray(block_ids, dtype=int),
        "bin_starts": np.asarray(bin_starts, dtype=int),
        "qc_status": "ok",
    })
    return result


def standardize_with_constants(stream: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply explicit standardization constants (calibration-fitted in
    few-shot; the constants are NEVER fitted on a final test interval —
    the caller owns that rule and the leakage tests pin it)."""
    if not np.isfinite(std) or std <= 0:
        raise DataValidationError(f"degenerate standardization constant std={std!r}")
    return (np.asarray(stream, dtype=float) - float(mean)) / float(std)


# ---------------------------------------------------------------------------
# Tap events (Track D)
# ---------------------------------------------------------------------------


def extract_tap_events(
    glove: np.ndarray, fs: float, blocks: list[tuple[int, int, str]],
) -> list[dict]:
    """Per-channel-block tap-event records under the frozen rule.

    Each record: ``{"block", "channel", "peak_samples", "iti_s",
    "log_iti_s", "amplitude_raw", "peak_velocity_raw", "n_cycles",
    "orientation"}``. Amplitude/velocity are raw sensor values — callers
    z them within recording (within-person only). A channel-block below
    the frozen prominence floor contributes nothing (recorded upstream
    by the phenotyping QC, not silently here).
    """
    from ..glove_phenotyping import (  # frozen prominence + cycle floors (read-only reuse)
        MIN_CYCLES_PER_BLOCK,
        channel_block_kinematics,
    )

    glove = np.asarray(glove, dtype=float)
    events: list[dict] = []
    for block_number, (start, end, kind) in enumerate(blocks):
        if kind != "move":
            continue
        end = min(end, glove.shape[1])
        for channel_idx in range(glove.shape[0]):
            segment = glove[channel_idx, start:end]
            record = channel_block_kinematics(segment, fs)
            if not record["contributing"] or record["n_cycles"] is None \
                    or record["n_cycles"] < MIN_CYCLES_PER_BLOCK:
                continue
            filtered = bandpass(segment, fs, TAP_BAND_LOW, TAP_BAND_HIGH,
                                order=TAP_FILTER_ORDER)
            peaks, orientation = _cycles(filtered, fs)
            if peaks.size < 5:
                continue
            peak_samples = peaks + start
            iti = np.diff(peak_samples) / fs
            amplitudes = filtered[peaks]
            # Peak velocity: max |diff| within one half-cycle either side.
            half = max(int(round(0.5 * fs / max(record["f_tap_hz"], 1e-6))), 1)
            velocities = []
            for p in peaks:
                lo, hi = max(p - half, 0), min(p + half, filtered.size)
                segment_diff = np.abs(np.diff(filtered[lo:hi]))
                velocities.append(float(np.max(segment_diff)) * fs if segment_diff.size else np.nan)
            events.append({
                "block": block_number,
                "channel": int(channel_idx),
                "peak_samples": peak_samples,
                "iti_s": iti,
                "log_iti_s": np.log10(np.maximum(iti, 1e-6)),
                "amplitude_raw": amplitudes,
                "peak_velocity_raw": np.asarray(velocities, dtype=float),
                "n_cycles": int(peaks.size),
                "orientation": orientation,
                "f_tap_hz": record["f_tap_hz"],
            })
    return events


def event_outcome_z(values: np.ndarray) -> np.ndarray:
    """Within-recording z of an event outcome (within-person only)."""
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0:
        return np.full(values.shape, np.nan)
    return (values - float(np.mean(values))) / std


def parse_blocks(segs: np.ndarray) -> list[tuple[int, int, str]]:
    """Passthrough block parser (kept here so callers need one import)."""
    return parse_restmove_blocks(segs)

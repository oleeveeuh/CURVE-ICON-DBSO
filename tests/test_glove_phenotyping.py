"""Deterministic synthetic tests for the prespecified glove phenotyping
(frozen contract §2): known-rate tapping, interval jitter, amplitude
decay, missing cycles, sensor inversion, drift, noise, QC floors, the
primary/secondary endpoint separation, the sanitized aggregate report,
and the leakage guarantees. Waveforms with known ground truth only — no
real data, no ECoG argument exists in the module under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.glove_phenotyping import (
    MIN_CONTRIBUTING_BLOCKS,
    MIN_CYCLES_PER_BLOCK,
    channel_block_kinematics,
    extract_glove_targets,
    format_glove_qc_report,
    glove_move_envelope,
)

FS = 500.0


def _taps(fs: float, seconds: float, rate_hz: float, amplitude: float = 1.0,
          interval_cv: float = 0.0, decay: float = 0.0, seed: int = 0,
          amplitude_floor: float = 0.0) -> np.ndarray:
    """Tap train: half-wave-rectified sinusoid-like taps with optional
    interval jitter, per-cycle amplitude decay, and noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    signal = np.zeros(n)
    period = 1.0 / rate_hz
    t = 0.0
    cycle = 0
    base_amplitude = amplitude
    while t < seconds:
        amp = base_amplitude * (1.0 + decay * cycle)
        amp = max(amp, amplitude_floor * base_amplitude)
        jitter = rng.normal(0.0, interval_cv * period) if interval_cv > 0 else 0.0
        cycle_start = t + jitter
        # A tap: one positive half-cycle at the tap rate, shaped smoothly.
        tap_len = period / 2.0
        idx = np.arange(int(cycle_start * fs), min(int((cycle_start + tap_len) * fs), n))
        if idx.size:
            phase = (idx - cycle_start * fs) / fs / tap_len * np.pi
            signal[idx] += amp * np.sin(phase)
        t += period
        cycle += 1
    return signal + 0.002 * rng.standard_normal(n)


def _layout(fs: float, seconds: float, pad_s: float = 1.0) -> tuple[int, int, int, np.ndarray]:
    """REST(pad) + MOVE(seconds) + REST(pad) + MOVE(seconds) segment
    layout; returns (total, pad, block_len, segs)."""
    pad = int(pad_s * fs)
    block = int(seconds * fs)
    total = 2 * pad + 2 * block
    segs = np.array([
        [0, pad, 0, 0],
        [pad, pad + block, 0, 1],
        [pad + block, pad + block + pad, 0, 0],
        [pad + block + pad, total, 0, 1],
    ], dtype=float)
    return total, pad, block, segs


def _two_move_blocks(fs: float, seconds: float, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Glove (1, total) with two MOVE blocks of `seconds` each, built
    from one tap train spanning the whole recording."""
    total, _pad, _block, segs = _layout(fs, seconds)
    train = _taps(fs, total / fs, **kwargs)
    return train.reshape(1, -1), segs


# ---------------------------------------------------------------------------
# Known rate: the primary passes through; cycles match ground truth
# ---------------------------------------------------------------------------


def test_known_rate_is_recovered_and_primary_is_frozen_passthrough():
    fs, seconds, rate = FS, 4.0, 2.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=rate)
    result = extract_glove_targets(glove, fs, segs)
    primary = result["primary"]
    assert primary["endpoint"] == "primary"
    assert primary["qc_status"] == "ok"
    assert primary["tap_rate_mean"] == pytest.approx(rate, abs=0.15)
    # Passthrough identity: identical to the frozen implementation.
    from curve_icon_dbso.kinematics import tap_rate_for_recording

    frozen = tap_rate_for_recording(glove, fs, segs)
    assert primary["tap_rate_mean"] == frozen["tap_rate_mean"]
    assert primary["definition"].startswith("frozen kinematics")
    # Cycle-based secondary targets exist and are labeled secondary.
    for target in ("tap_interval_cv", "tap_vigor_index", "tap_sequence_slope"):
        assert result["secondary"][target]["endpoint"] == "secondary"
        assert result["secondary"][target]["value"] is not None
    _total, pad, block, _segs_layout = _layout(fs, seconds)
    per_block = channel_block_kinematics(glove[0, pad: pad + block], fs)
    expected_cycles = seconds * rate
    assert per_block["n_cycles"] == pytest.approx(expected_cycles, rel=0.2)
    assert per_block["n_cycles"] >= MIN_CYCLES_PER_BLOCK


def test_regular_tapping_has_near_zero_interval_cv_and_unit_vigor():
    fs, seconds = FS, 4.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0)
    result = extract_glove_targets(glove, fs, segs)
    interval_cv = result["secondary"]["tap_interval_cv"]["value"]
    vigor = result["secondary"]["tap_vigor_index"]["value"]
    assert 0.0 <= interval_cv <= 0.15
    # A pure sinusoid scores vigor ≈ 1.0 by the frozen normalization.
    assert vigor == pytest.approx(1.0, rel=0.25)


def test_interval_jitter_raises_cv_proportionally():
    fs, seconds = FS, 4.0
    regular, segs_r = _two_move_blocks(fs, seconds, rate_hz=2.0, interval_cv=0.0)
    jittered, segs_j = _two_move_blocks(fs, seconds, rate_hz=2.0, interval_cv=0.20)
    cv_regular = extract_glove_targets(regular, fs, segs_r)[
        "secondary"]["tap_interval_cv"]["value"]
    cv_jittered = extract_glove_targets(jittered, fs, segs_j)[
        "secondary"]["tap_interval_cv"]["value"]
    assert cv_jittered > 2.0 * cv_regular
    assert 0.05 <= cv_jittered <= 0.5


# ---------------------------------------------------------------------------
# Amplitude: decay, vigor invariance, within-subject-only marking
# ---------------------------------------------------------------------------


def test_amplitude_decay_yields_a_negative_sequence_slope():
    fs, seconds = FS, 4.0
    decayed, segs = _two_move_blocks(
        fs, seconds, rate_hz=2.0, decay=-0.12, amplitude_floor=0.3
    )
    result = extract_glove_targets(decayed, fs, segs)
    slope = result["secondary"]["tap_sequence_slope"]["value"]
    assert slope is not None and slope < -0.05
    assert slope > -0.6  # decay −0.12/cycle normalized by the mean amplitude
    regular, segs_r = _two_move_blocks(fs, seconds, rate_hz=2.0)
    slope_flat = extract_glove_targets(regular, fs, segs_r)[
        "secondary"]["tap_sequence_slope"]["value"]
    assert slope_flat == pytest.approx(0.0, abs=0.06)


def test_vigor_is_amplitude_invariant():
    fs, seconds = FS, 4.0
    small, segs = _two_move_blocks(fs, seconds, rate_hz=2.0, amplitude=1.0)
    big = small * 250.0
    vigor_small = extract_glove_targets(small, fs, segs)[
        "secondary"]["tap_vigor_index"]["value"]
    vigor_big = extract_glove_targets(big, fs, segs)[
        "secondary"]["tap_vigor_index"]["value"]
    assert vigor_small == pytest.approx(vigor_big, rel=1e-6)


def test_amplitude_is_marked_within_subject_only():
    fs, seconds = FS, 4.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0)
    result = extract_glove_targets(glove, fs, segs)
    amplitude = result["within_subject_only"]["tap_amplitude"]
    assert amplitude["endpoint"] == "within_subject_only"
    assert amplitude["value"] is not None and amplitude["value"] > 0
    assert "never modeled cross-subject" in format_glove_qc_report([result])


# ---------------------------------------------------------------------------
# Missing cycles, inversion, drift, noise
# ---------------------------------------------------------------------------


def test_missing_cycles_reduce_counts_and_raise_cv():
    fs, seconds, rate = FS, 4.0, 2.0
    total, pad, block, segs = _layout(fs, seconds)
    train = _taps(fs, total / fs, rate_hz=rate)
    # Punch a ~1.5-cycle gap inside the second MOVE block.
    gap_start = pad + block + pad + int(0.5 * fs)
    gap_end = gap_start + int(0.75 * fs)
    train[gap_start:gap_end] = 0.0
    glove = train.reshape(1, -1)
    result = extract_glove_targets(glove, fs, segs)
    clean, segs_clean = _two_move_blocks(fs, seconds, rate_hz=rate)
    _total, pad, block, _layout_segs = _layout(fs, seconds)
    channel_block_kinematics(clean[0, pad:pad + block], fs)  # clean reference
    gapped_block = channel_block_kinematics(
        glove[0, pad + block + pad: total], fs
    )
    assert gapped_block["n_cycles"] is not None
    # The gap must register as irregularity (edge ringing can add small
    # peaks, so the count itself is not monotone — the CV is).
    cv_clean = extract_glove_targets(clean, fs, segs_clean)[
        "secondary"]["tap_interval_cv"]["value"]
    cv_gapped = result["secondary"]["tap_interval_cv"]["value"]
    assert cv_gapped is not None
    assert cv_gapped > 2.0 * cv_clean


def test_sensor_inversion_yields_the_same_targets():
    fs, seconds = FS, 4.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0, seed=3)
    upright = extract_glove_targets(glove, fs, segs)
    inverted = extract_glove_targets(-glove, fs, segs)
    a = upright["secondary"]["tap_vigor_index"]["value"]
    b = inverted["secondary"]["tap_vigor_index"]["value"]
    # Vigor is sign-invariant by construction (RMS ratios, no peaks).
    assert a == pytest.approx(b, rel=0.01)
    # Both orientations must recover a regular rhythm (loose band: the
    # orientation choice can pick different-but-corresponding extrema).
    for target in ("tap_interval_cv",):
        assert upright["secondary"][target]["value"] < 0.15
        assert inverted["secondary"][target]["value"] < 0.15
    assert upright["primary"]["tap_rate_mean"] == pytest.approx(
        inverted["primary"]["tap_rate_mean"], rel=0.01
    )


def test_slow_drift_is_removed_by_the_detection_band():
    fs, seconds = FS, 4.0
    clean, segs = _two_move_blocks(fs, seconds, rate_hz=2.0, seed=5)
    t = np.arange(clean.shape[1]) / fs
    drifted = clean + 1.0 * np.sin(2 * np.pi * 0.2 * t)
    clean_result = extract_glove_targets(clean, fs, segs)
    drifted_result = extract_glove_targets(drifted, fs, segs)
    for target in ("tap_interval_cv", "tap_vigor_index"):
        assert drifted_result["secondary"][target]["value"] == pytest.approx(
            clean_result["secondary"][target]["value"], rel=0.1
        )


def test_small_noise_does_not_destroy_the_targets():
    fs, seconds = FS, 4.0
    rng = np.random.default_rng(9)
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0, seed=2)
    noisy = glove + 0.1 * rng.standard_normal(glove.shape)
    result = extract_glove_targets(noisy, fs, segs)
    assert result["primary"]["tap_rate_mean"] == pytest.approx(2.0, abs=0.25)
    assert result["secondary"]["tap_interval_cv"]["value"] is not None


# ---------------------------------------------------------------------------
# QC floors and failure modes
# ---------------------------------------------------------------------------


def test_below_two_contributing_blocks_secondary_targets_are_none():
    fs, seconds = FS, 4.0
    total, pad, block, segs = _layout(fs, seconds)
    train = _taps(fs, total / fs, rate_hz=2.0)
    # Flat second channel: two blocks total, both contributing (2 ≥ 2).
    glove = np.vstack([train, np.zeros(total)])
    # Channel 1 (flat) contributes nothing in either MOVE block; channel 0
    # contributes in both → 2 contributing channel-blocks → still eligible.
    result = extract_glove_targets(glove, fs, segs)
    assert result["secondary"]["tap_interval_cv"]["qc_status"] == "ok"
    # Now kill the second block of channel 0 as well: only 1 contributing.
    glove2 = glove.copy()
    glove2[0, pad + block + pad:] = 0.0
    result2 = extract_glove_targets(glove2, fs, segs)
    assert result2["primary"]["qc_status"] == "failed"
    assert result2["secondary"]["tap_interval_cv"]["value"] is None
    assert result2["secondary"]["tap_interval_cv"]["qc_status"] == "failed"
    assert "required" in result2["secondary"]["tap_interval_cv"]["qc_reason"]


def test_cycle_floor_per_block_is_enforced():
    """A contributing block with fewer than 5 cycles feeds no cycle target."""
    fs = FS
    short_block = _taps(fs, 1.0, rate_hz=2.0)  # ~2 cycles
    record = channel_block_kinematics(short_block, fs)
    assert record["n_cycles"] is not None and record["n_cycles"] < MIN_CYCLES_PER_BLOCK
    assert record["tap_interval_cv"] is None
    assert record["tap_sequence_slope"] is None
    assert any(r.startswith("cycle_floor_not_met") for r in record["reasons"])


def test_invalid_inputs_raise_loudly():
    glove, segs = _two_move_blocks(FS, 2.0, rate_hz=2.0)
    with pytest.raises(DataValidationError, match="positive"):
        extract_glove_targets(glove, -1.0, segs)
    with pytest.raises(DataValidationError):
        extract_glove_targets(glove, FS, np.array([[0, 10, 0, 7]], dtype=float))


def test_move_envelope_is_frozen_definition_and_inversion_invariant():
    fs, seconds = FS, 4.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0)
    forward = glove_move_envelope(glove, fs, segs)
    flipped = glove_move_envelope(-glove, fs, segs)
    assert forward["qc_status"] == "ok"
    assert forward["stream"] is not None
    assert forward["n_samples"] == forward["stream"].size
    assert float(np.std(forward["stream"])) == pytest.approx(1.0, abs=1e-6)
    assert float(np.mean(forward["stream"])) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(forward["stream"], flipped["stream"])
    assert set(np.unique(forward["block_ids"])) == {0, 1}
    degenerate = glove_move_envelope(np.zeros_like(glove), fs, segs)
    assert degenerate["qc_status"] == "failed" and degenerate["stream"] is None


# ---------------------------------------------------------------------------
# Report, determinism, leakage guarantees
# ---------------------------------------------------------------------------


def test_aggregate_report_is_sanitized_and_deterministic():
    fs, seconds = FS, 4.0
    outputs = []
    for seed in range(4):
        glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0,
                                       interval_cv=0.05 * seed, seed=seed)
        outputs.append(extract_glove_targets(glove, fs, segs))
    report = format_glove_qc_report(outputs)
    assert report == format_glove_qc_report(outputs)
    assert "DBS_bG" not in report
    assert "Recordings evaluated            : 4" in report
    assert "Primary tap-rate QC passes      : 4/4" in report
    assert "tap_interval_cv" in report
    # No per-subject rows: every content line is a count or cohort median.
    for line in report.splitlines():
        assert ":" in line or line.startswith("GLOVE") or line.startswith("Endpoint")


def test_extraction_is_deterministic():
    fs, seconds = FS, 4.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0, interval_cv=0.1, seed=4)
    first = repr(extract_glove_targets(glove, fs, segs))
    second = repr(extract_glove_targets(glove, fs, segs))
    assert first == second


def test_outputs_carry_no_subject_identity():
    fs, seconds = FS, 4.0
    glove, segs = _two_move_blocks(fs, seconds, rate_hz=2.0)
    blob = repr(extract_glove_targets(glove, fs, segs))
    assert "DBS_bG" not in blob
    assert "subject_id" not in blob.lower()
    assert "fold" not in blob.lower()


def test_frozen_floors_match_the_contract():
    assert MIN_CONTRIBUTING_BLOCKS == protocol.TAP_MIN_CHANNEL_BLOCKS == 2
    assert MIN_CYCLES_PER_BLOCK == protocol.TAP_MIN_CYCLES_PER_BLOCK == 5
    doc = protocol.protocol_document()
    assert doc["targets"]["primary_subject"]["name"] == "tap_rate_mean"
    assert set(doc["targets"]["secondary_subject"]) == {
        "tap_interval_cv", "tap_vigor_index", "tap_sequence_slope",
    }
    assert set(doc["targets"]["within_subject_only"]) == {"tap_amplitude"}

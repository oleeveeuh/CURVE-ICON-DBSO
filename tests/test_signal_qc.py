"""Deterministic synthetic tests for the signal-QC layer (frozen follow-on
contract): line-noise handling, rereferencing, channel/window QC with
machine-readable reasons, and the QC-summary contract (counts + reasons,
no participant identifiers). Known waveforms only — no real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.errors import DataValidationError, SamplingRateError
from curve_icon_dbso.neural_features import window_band_powers
from curve_icon_dbso.signal_qc import (
    FLATLINE_TOL,
    IMPLAUSIBLE_POWER_Z,
    LINE_FREQ_AMBIGUITY_DB,
    SATURATION_FRACTION,
    TRANSIENT_FRACTION,
    TRANSIENT_MAD_MULT,
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
    window_grid,
)


def _sine(freq_hz: float, fs: float, seconds: float, amplitude: float = 1.0,
          phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)


def _tone_plus_noise(freq_hz: float, fs: float, seconds: float,
                     amplitude: float, noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _sine(freq_hz, fs, seconds, amplitude) + noise * rng.standard_normal(
        int(seconds * fs)
    )


# ---------------------------------------------------------------------------
# Frozen thresholds stay in lockstep with the contract prose
# ---------------------------------------------------------------------------


def test_thresholds_match_frozen_contract_prose():
    qc_rules = protocol.QC_RULES
    assert "1e-12" in qc_rules["flatline"] and FLATLINE_TOL == 1e-12
    assert "5%" in qc_rules["saturation"] and SATURATION_FRACTION == 0.05
    assert "20 × MAD" in qc_rules["rapid_transients"] and TRANSIENT_MAD_MULT == 20.0
    assert "0.005" in qc_rules["rapid_transients"] and TRANSIENT_FRACTION == 0.005
    assert "exceeds 10" in qc_rules["implausible_band_power"]
    assert IMPLAUSIBLE_POWER_Z == 10.0
    assert "± 2 Hz" in qc_rules["line_noise"] and "3 dB" in qc_rules["line_noise"]
    assert LINE_FREQ_AMBIGUITY_DB == 3.0


# ---------------------------------------------------------------------------
# Line noise
# ---------------------------------------------------------------------------


def test_line_frequency_detected_at_50hz():
    fs = 1000.0
    signal = _tone_plus_noise(50.0, fs, 4.0, 0.5, 0.05, seed=1)
    result = detect_line_frequency(signal, fs)
    assert result["line_freq_hz"] == 50.0
    assert result["ambiguous"] is False


def test_equal_50_and_60_peaks_are_flagged_ambiguous_and_larger_wins():
    fs = 1000.0
    signal = _sine(50.0, fs, 4.0, 1.0) + _sine(60.0, fs, 4.0, 1.0)
    result = detect_line_frequency(signal, fs)
    assert result["ambiguous"] is True
    assert result["line_freq_hz"] in (50.0, 60.0)


def test_candidate_above_ceiling_is_ignored():
    fs = 120.0  # ceiling = 60 − 10 − 2 = 48 → 50/60 both infeasible
    result = detect_line_frequency(_sine(50.0, fs, 4.0, 1.0), fs)
    assert result["line_freq_hz"] is None


def test_notch_removes_the_line_and_its_harmonics():
    fs = 2000.0
    signal = (
        _tone_plus_noise(50.0, fs, 4.0, 0.5, 0.02, seed=2)
        + _sine(150.0, fs, 4.0, 0.2)          # third harmonic
        + _sine(10.0, fs, 4.0, 0.5)           # protected content
    )

    def band_power_db(x: np.ndarray, low: float, high: float) -> float:
        from scipy.signal import welch

        freqs, psd = welch(x, fs=fs, nperseg=int(fs))
        mask = (freqs >= low) & (freqs <= high)
        return 10.0 * float(np.log10(np.max(psd[mask])))

    cleaned = notch_line_noise(signal, fs, 50.0)
    assert band_power_db(cleaned, 49, 51) <= band_power_db(signal, 49, 51) - 20.0
    assert band_power_db(cleaned, 149, 151) <= band_power_db(signal, 149, 151) - 20.0
    # Non-line content survives.
    assert band_power_db(cleaned, 9, 11) >= band_power_db(signal, 9, 11) - 1.0


# ---------------------------------------------------------------------------
# Rereferencing
# ---------------------------------------------------------------------------


def test_bipolar_rereference_is_anode_minus_cathode():
    fs = 500.0
    a = _sine(10.0, fs, 1.0, 2.0)
    b = _sine(20.0, fs, 1.0, 1.0)
    ecog = np.vstack([a, b, _sine(30.0, fs, 1.0, 1.0)])
    rereferenced, labels = bipolar_rereference(ecog, [(0, 1)])
    np.testing.assert_allclose(rereferenced[0], a - b)
    assert labels == ["0-1"]


def test_bipolar_rereference_rejects_bad_adjacency():
    ecog = np.zeros((3, 10))
    with pytest.raises(DataValidationError, match="explicit adjacency"):
        bipolar_rereference(ecog, [])
    with pytest.raises(DataValidationError, match="indexes channel"):
        bipolar_rereference(ecog, [(0, 5)])
    with pytest.raises(DataValidationError, match="degenerate"):
        bipolar_rereference(ecog, [(1, 1)])


def test_common_median_reference_subtracts_the_ok_channel_median():
    t = np.arange(100)
    ecog = np.vstack([t * 1.0, t * 2.0, t * 4.0])
    referenced = common_median_reference(ecog)
    np.testing.assert_allclose(referenced[0], -(t * 1.0))  # median = 2t
    np.testing.assert_allclose(referenced[1], 0.0)
    # Non-finite channels are excluded from the median, never imputed.
    broken = ecog.copy()
    broken[1, 5] = np.nan
    ok_mask = np.isfinite(broken).all(axis=1)
    referenced = common_median_reference(broken, ok_mask)
    # OK channels are 0 (t) and 2 (4t); their median is 2.5t.
    np.testing.assert_allclose(referenced[0][:5], -(t[:5] * 1.5))
    with pytest.raises(DataValidationError, match="at least one OK channel"):
        common_median_reference(ecog, np.array([False, False, False]))


# ---------------------------------------------------------------------------
# Channel-block QC
# ---------------------------------------------------------------------------


def test_qc_flags_nonfinite_flatline_saturation_transients():
    fs = 1000.0
    diff_mad = 0.01
    nan_block = _sine(10.0, fs, 1.0)
    nan_block[10] = np.nan
    record = qc_channel_block(nan_block, diff_mad)
    assert record["qc_status"] == "failed" and "non-finite" in record["qc_reason"]

    record = qc_channel_block(np.full(1000, 3.0), diff_mad)
    assert record["qc_status"] == "failed" and record["qc_reason"] == "flatline"

    # > 5 % of samples pinned at rails: 100 flat-top samples of 1000.
    saturated = np.concatenate([_sine(10.0, fs, 0.9), np.full(100, 5.0)])
    record = qc_channel_block(saturated, diff_mad)
    assert record["qc_status"] == "failed" and "saturation" in record["qc_reason"]

    # Rapid transients: a few huge jumps against a tiny diff-MAD.
    quiet = _sine(10.0, fs, 1.0, amplitude=0.001) * 0.001
    quiet[::200] += 50.0
    record = qc_channel_block(quiet, diff_mad=1e-4)
    assert record["qc_status"] == "failed" and "rapid transients" in record["qc_reason"]

    record = qc_channel_block(_sine(10.0, fs, 1.0, 0.5), diff_mad)
    assert record["qc_status"] == "ok"


def test_qc_flags_implausible_band_power_from_reference_z():
    fs = 1000.0
    z_scores = {"gamma_high": IMPLAUSIBLE_POWER_Z + 1.0}
    record = qc_channel_block(_sine(10.0, fs, 1.0), diff_mad=0.01,
                              reference_band_power_z=z_scores)
    assert record["qc_status"] == "failed"
    assert "implausible_band_power" in record["qc_reason"]
    record = qc_channel_block(_sine(10.0, fs, 1.0), diff_mad=0.01,
                              reference_band_power_z={"gamma_high": 2.0})
    assert record["qc_status"] == "ok"


# ---------------------------------------------------------------------------
# Window grid and window QC
# ---------------------------------------------------------------------------


def test_window_grid_is_causal_steps_and_never_crosses_bounds():
    fs = 100.0
    grid = window_grid(0, 200, fs)  # 2 s block, 0.5 s windows, 0.1 s step
    assert grid[0] == (0, 50)
    for (prev_start, prev_end), (start, end) in zip(grid, grid[1:], strict=False):
        assert start - prev_start == 10 and end - prev_end == 10
    for start, end in grid:
        assert 0 <= start and end <= 200 and end - start == 50
    assert grid[-1][1] <= 200
    with pytest.raises(DataValidationError):
        window_grid(10, 5, fs)
    with pytest.raises(DataValidationError):
        window_grid(0, 200, fs=-1.0)


def test_qc_windows_reports_machine_readable_reasons():
    fs = 100.0
    # One saturated stretch inside an otherwise quiet block.
    block = 0.1 * np.sin(2 * np.pi * 2 * np.arange(200) / fs)
    block[60:120] = 9.9  # a whole window pinned at a rail value
    records = qc_windows(block, fs, diff_mad=0.01)
    failed = [r for r in records if r["qc_status"] == "failed"]
    assert failed, "the saturated window must fail"
    flat_keys = {r["reasons"][0].split(":")[0] for r in failed}
    assert flat_keys <= {"saturation", "flatline", "rapid_transients",
                         "non_finite_samples", "implausible_band_power"}
    summary = qc_summary(records)
    assert summary["n_failed"] == len(failed)
    assert summary["n_ok"] == len(records) - len(failed)
    assert summary["reasons"] and all(isinstance(k, str) for k in summary["reasons"])


def test_qc_windows_implausible_power_uses_reference_z():
    fs = 1000.0
    block = _sine(70.0, fs, 2.0, 5.0)
    grid_len = len(window_grid(0, block.size, fs))  # 16 windows
    z_scores = {"gamma_high": [0.0] * (grid_len - 5) + [50.0] * 5}
    records = qc_windows(block, fs, diff_mad=5.0,
                         reference_band_power_z=z_scores)
    failed = [r for r in records if r["qc_status"] == "failed"]
    assert len(failed) == 5
    assert all("implausible_band_power:gamma_high" in r["reasons"] for r in failed)


# ---------------------------------------------------------------------------
# Recording-scale helpers
# ---------------------------------------------------------------------------


def test_recording_diff_mad_is_positive_and_deterministic():
    fs = 500.0
    ecog = np.vstack([_sine(10.0, fs, 1.0, 1.0), _sine(20.0, fs, 1.0, 2.0)])
    mad = recording_diff_mad(ecog)
    assert mad > 0 and mad == recording_diff_mad(ecog)
    with pytest.raises(DataValidationError):
        recording_diff_mad(np.full((2, 10), np.nan))


def test_validate_fs_and_active_top_edge():
    bands = protocol.resolve_spectral_bands(2000.0)
    assert active_top_edge(bands) == 100.0
    with pytest.raises(SamplingRateError):
        validate_fs(210.0, 100.0)  # ceiling = 105 − 10 = 95 Hz < 100 Hz
    with pytest.raises(SamplingRateError):
        validate_fs(float("nan"), 60.0)
    with pytest.raises(DataValidationError):
        active_top_edge({"bands_active": {}, "gamma_high": None})


def test_bandpass_validates_and_filters():
    fs = 1000.0
    signal = _sine(70.0, fs, 1.0) + _sine(5.0, fs, 1.0)
    filtered = bandpass(signal, fs, 60.0, 100.0)
    assert float(np.var(filtered)) > 0
    # The 5 Hz component is strongly attenuated.
    from scipy.signal import welch

    freqs, psd = welch(filtered, fs=fs, nperseg=int(fs))
    low_power = float(np.max(psd[(freqs > 4) & (freqs < 6)]))
    high_power = float(np.max(psd[(freqs > 65) & (freqs < 75)]))
    assert high_power > 100.0 * low_power
    with pytest.raises(SamplingRateError):
        bandpass(signal, fs, 60.0, 600.0)  # ceiling = 490 Hz


def test_window_band_powers_are_per_window_and_finite():
    fs = 1000.0
    bands = {**protocol.STANDARD_BANDS_HZ, "gamma_high": (60.0, 100.0)}
    block = _tone_plus_noise(70.0, fs, 2.0, 1.0, 0.1, seed=3)
    powers = window_band_powers(block, fs, bands, top_edge=100.0)
    assert set(powers) == set(bands)
    for values in powers.values():
        assert len(values) == 16  # 2 s at 0.5 s windows / 0.1 s step
        assert np.isfinite(values).all()
    # gamma_high carries the injected component.
    assert np.median(powers["gamma_high"]) > np.median(powers["theta"])

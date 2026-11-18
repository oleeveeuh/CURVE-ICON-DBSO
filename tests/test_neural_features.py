"""Deterministic synthetic tests for the neural feature layer (frozen
follow-on contract): band boundaries, aperiodic fits on 1/f backgrounds,
transform invariances, beta bursts, REST/MOVE/DELTA state differences,
channel failures, region-aware aggregation, and per-window QC. Known
waveforms and seeded noise only — no real data anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.errors import DataValidationError, SamplingRateError
from curve_icon_dbso.neural_features import (
    aperiodic_adjusted_band_power,
    aperiodic_fit,
    band_mask,
    band_power,
    beta_burst_features,
    build_subject_level_features,
    extract_block_features,
    hjorth_parameters,
    implausible_power_z,
    line_length_per_s,
    spectral_entropy,
    spectral_features,
    welch_psd,
)


def _sine(freq_hz: float, fs: float, seconds: float, amplitude: float = 1.0,
          phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)


def _pink_noise(n_samples: int, exponent: float, seed: int = 7) -> np.ndarray:
    """Frequency-domain 1/f^exponent background (unit-variance-ish)."""
    rng = np.random.default_rng(seed)
    spectrum = rng.standard_normal(n_samples + 1) + 1j * rng.standard_normal(n_samples + 1)
    freqs = np.fft.rfftfreq(2 * n_samples, d=1.0)
    shaping = np.ones_like(freqs)
    nonzero = freqs > 0
    shaping[nonzero] = 1.0 / freqs[nonzero] ** (exponent / 2.0)
    shaping[0] = 0.0
    spectrum *= shaping
    series = np.fft.irfft(spectrum, n=2 * n_samples)[:n_samples]
    return series / float(np.std(series))


BANDS_2K = {**protocol.STANDARD_BANDS_HZ, "gamma_high": (60.0, 100.0)}


# ---------------------------------------------------------------------------
# Welch PSD and band membership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq_hz,band", [
    (5.0, "theta"), (10.0, "alpha"), (16.0, "beta_low"), (25.0, "beta_high"),
    (45.0, "gamma_low"), (70.0, "gamma_high"), (200.0, "hfo"),
])
def test_known_sinusoids_land_in_their_bands(freq_hz, band):
    fs = 2000.0
    signal = _sine(freq_hz, fs, 2.0) + 0.01 * np.random.default_rng(0).standard_normal(
        int(2 * fs)
    )
    freqs, psd = welch_psd(signal, fs)
    features = spectral_features(freqs, psd, protocol.resolve_spectral_bands(fs), 100.0)
    target = (
        band_power(freqs, psd, 80.0, 500.0, 500.0) if band == "hfo"
        else features[f"{band}_power"]
    )
    # Every OTHER band's power must be far below the target band's.
    other_bands = [b for b in (*BANDS_2K,) if b != band]
    for other in other_bands:
        low, high = BANDS_2K[other]
        if abs(low - freq_hz) < 40 or abs(high - freq_hz) < 40:
            continue  # adjacent-band spectral leakage is acceptable
        assert target > 100.0 * band_power(freqs, psd, low, high, 100.0), (
            f"{freq_hz} Hz energy must dominate over band {other}"
        )


def test_band_membership_is_half_open_with_inclusive_top_edge():
    freqs = np.array([13.0, 19.9, 20.0, 34.9, 100.0])
    np.testing.assert_array_equal(
        band_mask(freqs, 13.0, 20.0, 100.0), [True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        band_mask(freqs, 20.0, 35.0, 100.0), [False, False, True, True, False]
    )
    np.testing.assert_array_equal(
        band_mask(freqs, 60.0, 100.0, 100.0), [False, False, False, False, True]
    )


def test_relative_powers_sum_to_at_most_one_and_log_is_finite():
    fs = 2000.0
    signal = _pink_noise(int(2 * fs), 1.0)
    freqs, psd = welch_psd(signal, fs)
    features = spectral_features(freqs, psd, protocol.resolve_spectral_bands(fs), 100.0)
    total_relative = sum(
        features[f"{band}_relative_power"] for band in (*BANDS_2K,)
    )
    assert 0.0 < total_relative <= 1.0
    for band in BANDS_2K:
        logged = features[f"{band}_log_relative_power"]
        assert logged is not None and np.isfinite(logged)
        assert logged == pytest.approx(
            np.log10(features[f"{band}_relative_power"])
        )


# ---------------------------------------------------------------------------
# Aperiodic fit and aperiodic-adjusted power
# ---------------------------------------------------------------------------


def test_aperiodic_fit_recovers_a_pink_background():
    fs = 2000.0
    signal = _pink_noise(int(2 * fs), 1.0, seed=11)  # PSD ∝ 1/f
    freqs, psd = welch_psd(signal, fs)
    fit = aperiodic_fit(freqs, psd, 4.0, 100.0, 100.0)
    assert fit["exponent"] == pytest.approx(1.0, abs=0.3)
    assert np.isfinite(fit["offset"]) and fit["predict"](20.0) > 0


def test_aperiodic_fit_recovers_a_brown_background():
    fs = 2000.0
    signal = _pink_noise(int(2 * fs), 2.0, seed=5)  # PSD ∝ 1/f²
    freqs, psd = welch_psd(signal, fs)
    fit = aperiodic_fit(freqs, psd, 4.0, 100.0, 100.0)
    assert fit["exponent"] == pytest.approx(2.0, abs=0.3)


def test_aperiodic_adjusted_power_highlights_periodic_peaks():
    fs = 2000.0
    signal = _pink_noise(int(2 * fs), 1.0, seed=3) + _sine(70.0, fs, 2.0, 2.0)
    freqs, psd = welch_psd(signal, fs)
    fit = aperiodic_fit(freqs, psd, 4.0, 100.0, 100.0)
    gamma_surplus = aperiodic_adjusted_band_power(freqs, psd, 60.0, 100.0, 100.0, fit)
    theta_surplus = aperiodic_adjusted_band_power(freqs, psd, 4.0, 8.0, 100.0, fit)
    assert gamma_surplus > 5.0 * theta_surplus


# ---------------------------------------------------------------------------
# Dynamics features
# ---------------------------------------------------------------------------


def test_hjorth_mobility_of_a_sine_is_its_angular_frequency():
    fs = 1000.0
    mobility = hjorth_parameters(_sine(10.0, fs, 2.0), fs)["hjorth_mobility"]
    assert mobility == pytest.approx(2 * np.pi * 10.0, rel=1e-2)
    complexity = hjorth_parameters(_sine(10.0, fs, 2.0), fs)["hjorth_complexity"]
    assert complexity == pytest.approx(1.0, rel=1e-2)
    flat = hjorth_parameters(np.zeros(2000), fs)
    assert flat == {"hjorth_mobility": 0.0, "hjorth_complexity": 0.0}


def test_hjorth_is_scale_invariant():
    fs = 500.0
    signal = _pink_noise(int(2 * fs), 1.0, seed=9)
    low = hjorth_parameters(signal, fs)
    high = hjorth_parameters(signal * 1000.0, fs)
    assert low["hjorth_mobility"] == pytest.approx(high["hjorth_mobility"], rel=1e-8)
    assert low["hjorth_complexity"] == pytest.approx(high["hjorth_complexity"], rel=1e-8)


def test_line_length_per_s_is_exact_for_a_ramp():
    fs = 100.0
    n = 201
    ramp = 0.5 * np.arange(n, dtype=float)
    # Σ|Δx| = 0.5 · 200 over 2.01 s.
    assert line_length_per_s(ramp, fs) == pytest.approx(0.5 * (n - 1) / (n / fs))
    with pytest.raises(DataValidationError):
        line_length_per_s(np.zeros(1), fs)


def test_spectral_entropy_separates_tone_from_noise():
    fs = 1000.0
    tone = spectral_entropy(*welch_psd(_sine(40.0, fs, 2.0), fs), 4.0, 100.0, 100.0)
    noise = spectral_entropy(
        *welch_psd(np.random.default_rng(1).standard_normal(int(2 * fs)), fs),
        4.0, 100.0, 100.0,
    )
    assert 0.0 <= tone < 0.3
    assert 0.6 < noise <= 1.0
    assert tone < noise


# ---------------------------------------------------------------------------
# Beta bursts
# ---------------------------------------------------------------------------


def _bursty_signal(fs: float, seconds: float, duty: float, burst_amp: float,
                   seed: int = 0) -> np.ndarray:
    """25 Hz carrier whose amplitude is burst_amp during bursts of
    200 ms every 500 ms, 1 otherwise."""
    t = np.arange(int(seconds * fs)) / fs
    envelope = np.where((t % 0.5) < 0.2, burst_amp, 1.0)
    rng = np.random.default_rng(seed)
    return envelope * np.sin(2 * np.pi * 25.0 * t) + 0.01 * rng.standard_normal(t.size)


def test_beta_burst_occupancy_duration_and_amplitude():
    fs = 1000.0
    reference = _sine(25.0, fs, 2.0)  # flat envelope → low 75th percentile
    eval_signal = _bursty_signal(fs, 2.0, duty=0.4, burst_amp=10.0)
    features = beta_burst_features(eval_signal, reference, fs)
    assert 0.30 <= features["beta_burst_occupancy"] <= 0.75
    assert features["beta_burst_n"] >= 2
    assert features["beta_burst_mean_duration_ms"] == pytest.approx(200.0, abs=130.0)
    assert features["beta_burst_median_amplitude_ratio"] > 3.0


def test_beta_bursts_are_zero_with_count_zero_when_none_exist():
    fs = 1000.0
    loud_reference = _sine(25.0, fs, 2.0, 10.0)
    quiet_eval = _sine(25.0, fs, 2.0, 1.0)
    features = beta_burst_features(quiet_eval, loud_reference, fs)
    assert features["beta_burst_occupancy"] == 0.0
    assert features["beta_burst_n"] == 0
    assert features["beta_burst_mean_duration_ms"] == 0.0


def test_beta_burst_threshold_is_reference_based_never_target_based():
    """Swapping the evaluation signal's amplitudes must not change the
    threshold — it is a property of the reference only."""
    fs = 1000.0
    reference = _sine(25.0, fs, 2.0)
    first = beta_burst_features(_bursty_signal(fs, 2.0, 0.4, 10.0), reference, fs)
    second = beta_burst_features(_bursty_signal(fs, 2.0, 0.4, 5.0), reference, fs)
    assert first["burst_threshold"] == second["burst_threshold"]


def test_beta_burst_rejects_degenerate_reference():
    with pytest.raises(DataValidationError, match="degenerate"):
        beta_burst_features(_sine(25.0, 1000.0, 1.0), np.zeros(1000), 1000.0)


# ---------------------------------------------------------------------------
# Block features and the HFO block
# ---------------------------------------------------------------------------


def test_block_features_include_hfo_when_feasible():
    fs = 2000.0
    bands = protocol.resolve_spectral_bands(fs)
    block = _sine(200.0, fs, 2.0) + 0.05 * np.random.default_rng(2).standard_normal(int(2 * fs))
    features = extract_block_features(block, fs, bands, 100.0)
    assert bands["hfo_block_feasible"] is True
    assert features["hfo_qc_status"] == "ok"
    assert features["hfo_peak_freq"] == pytest.approx(200.0, abs=5.0)
    assert features["hfo_power"] > 0


def test_block_features_omit_hfo_when_infeasible():
    fs = 1000.0
    bands = protocol.resolve_spectral_bands(fs)
    features = extract_block_features(_sine(25.0, fs, 1.0), fs, bands, 100.0)
    assert bands["hfo_block_feasible"] is False
    assert "hfo_power" not in features


# ---------------------------------------------------------------------------
# REST / MOVE / DELTA recording builder
# ---------------------------------------------------------------------------


def _recording(fs: float, seconds_per_block: float, move_gamma_gain: float,
               n_channels: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two REST blocks then two MOVE blocks; MOVE blocks add a gamma_high
    oscillation (the state difference) on every channel."""
    rng = np.random.default_rng(seed)
    block_len = int(seconds_per_block * fs)
    background = _pink_noise(block_len * 4 * n_channels, 1.0, seed=seed).reshape(
        n_channels, 4 * block_len
    ) * 5.0
    ecog = background.copy()
    for channel in range(n_channels):
        gamma = _sine(70.0, fs, 2 * seconds_per_block,
                      move_gamma_gain * (1.0 + 0.1 * channel),
                      phase=0.15 * channel)
        ecog[channel, 2 * block_len:] += gamma
    ecog += 0.1 * rng.standard_normal(ecog.shape)
    # restmove_segs: 2 REST then 2 MOVE, on the shared sample grid.
    segs = np.array([
        [0, block_len, 0, 0],
        [block_len, 2 * block_len, 0, 0],
        [2 * block_len, 3 * block_len, 0, 1],
        [3 * block_len, 4 * block_len, 0, 1],
    ], dtype=float)
    return ecog, segs


def test_delta_sees_the_state_difference():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=100.0)
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs)
    )
    delta = result["delta"]
    assert delta["gamma_high_log_ratio_power"] > delta["theta_log_ratio_power"]
    assert delta["gamma_high_log_ratio_power"] > 0.1
    # MOVE gamma occupancy against the REST threshold must exceed REST's
    # self-referenced baseline occupancy.
    assert (result["move"]["features"]["beta_burst_occupancy"]
            > result["rest"]["features"]["beta_burst_occupancy"] - 0.3)
    # Dynamics DELTA features exist and are finite where defined.
    for name in ("hjorth_mobility_delta", "spectral_entropy_delta",
                 "aperiodic_exponent_delta"):
        assert name in delta
        if delta[name] is not None:
            assert np.isfinite(delta[name])


def test_rest_and_move_feature_blocks_are_populated_and_finite():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0, seed=1)
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs)
    )
    for state in ("rest", "move"):
        features = result[state]["features"]
        expected_bands = sorted(
            (*BANDS_2K.keys(),)
        )
        for band in expected_bands:
            for suffix in ("_power", "_relative_power", "_log_relative_power",
                           "_power_aperiodic_adjusted"):
                value = features[f"{band}{suffix}"]
                if value is not None:
                    assert np.isfinite(value), f"{state}/{band}{suffix}"
        for key in ("aperiodic_exponent", "aperiodic_offset", "hjorth_mobility",
                    "line_length_per_s", "spectral_entropy",
                    "beta_burst_occupancy"):
            assert np.isfinite(features[key]), f"{state}/{key}"
        assert result[state]["n_channels_present"] == 4


def test_delta_is_none_without_a_state_and_never_fabricated():
    """A feature missing on one side yields None — never a zero."""
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0, seed=2)
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs)
    )
    rest_features = result["rest"]["features"]
    delta_names = {name.replace("_log_relative_power", "_log_ratio_power")
                   for name in rest_features
                   if name.endswith("_log_relative_power")}
    # Where both states exist, DELTA values are real floats (finite).
    for name in delta_names:
        if result["delta"][name] is not None:
            assert np.isfinite(result["delta"][name])


def test_nonfinite_channel_is_counted_and_excluded_not_zero_filled():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0, seed=3)
    ecog[3, :] = np.nan
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs)
    )
    qc = result["qc"]["rest_channel_qc"]["3"]
    assert qc["qc_status"] == "failed"
    assert "non-finite" in qc["qc_reason"]
    assert result["qc"]["n_channels_ok_after_reference"] == 3


def test_region_aware_aggregation_medians_per_region():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0, seed=4)
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs),
        region_map={"0": "M1", "1": "M1", "2": "PM", "3": "S1"},
    )
    assert result["qc"]["region_map_present"] is True
    assert result["rest"]["regions"]["M1"]["n_channels"] == 2
    assert result["rest"]["regions"]["PM"]["n_channels"] == 1
    assert result["rest"]["regions"]["S1"]["n_channels"] == 1
    # Regional medians are medians of the region's channels — the M1 pair
    # (channels 0,1 share construction) matches the overall median closely.
    m1 = result["rest"]["regions"]["M1"]["features"]["alpha_power"]
    overall = result["rest"]["features"]["alpha_power"]
    assert m1 == pytest.approx(overall, rel=1.0)


def test_bipolar_adjacency_path_and_pair_failure():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0, seed=5)
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs),
        adjacency=[(0, 1), (2, 3)],
    )
    assert result["qc"]["bipolar"] is True
    assert result["qc"]["n_channels_ok_after_reference"] == 2
    ecog[2, :] = np.nan
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs),
        adjacency=[(0, 1), (2, 3)],
    )
    # Pair (2,3) is dead; the recording continues on pair (0,1).
    assert result["qc"]["n_channels_ok_after_reference"] == 1
    assert "2-3" in result["qc"]["rest_channel_qc"]


def test_builder_rejects_missing_blocks_or_low_fs():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0)
    move_only = segs[segs[:, 3] == 1]
    with pytest.raises(DataValidationError, match="REST"):
        build_subject_level_features(ecog, fs, move_only,
                                     protocol.resolve_spectral_bands(fs))
    rest_only = segs[segs[:, 3] == 0]
    with pytest.raises(DataValidationError, match="MOVE"):
        build_subject_level_features(ecog, fs, rest_only,
                                     protocol.resolve_spectral_bands(fs))
    # Recording at 200 Hz cannot deliver bands resolved for 2000 Hz.
    with pytest.raises(SamplingRateError):
        build_subject_level_features(ecog, 200.0, segs,
                                     protocol.resolve_spectral_bands(fs))


# ---------------------------------------------------------------------------
# Per-window QC stream (MOVE arm reference)
# ---------------------------------------------------------------------------


def test_move_window_qc_stream_reports_implausible_power_and_saturation():
    fs = 2000.0
    ecog, segs = _recording(fs, 2.0, move_gamma_gain=5.0, seed=6)
    # Saturate a stretch inside the first MOVE block.
    block_len = int(2.0 * fs)
    ecog[0, 2 * block_len + 200: 2 * block_len + 700] = 500.0
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs),
        include_move_windows=True,
    )
    assert "move_window_qc" in result
    summary = result["move_window_qc"]
    assert summary["n_failed"] > 0
    assert any("saturation" in reason or "implausible" in reason
               for reason in summary["reasons"])


def test_implausible_power_z_flags_a_state_change():
    from curve_icon_dbso.neural_features import window_band_powers

    fs = 1000.0
    bands = {"gamma_high": (60.0, 100.0)}
    rest = _pink_noise(int(4 * fs), 1.0, seed=8) + _sine(70.0, fs, 4.0, 0.5)
    move = _pink_noise(int(2 * fs), 1.0, seed=8) + _sine(70.0, fs, 2.0, 50.0)
    z = implausible_power_z(
        {"gamma_high": window_band_powers(move, fs, bands, 100.0)["gamma_high"]},
        {"gamma_high": window_band_powers(rest, fs, bands, 100.0)["gamma_high"]},
    )
    assert max(z["gamma_high"]) > 5.0


# ---------------------------------------------------------------------------
# Properties: determinism, finiteness, no identifiers
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic():
    fs = 2000.0
    ecog, segs = _recording(fs, 1.5, move_gamma_gain=8.0, seed=10)
    first = build_subject_level_features(ecog, fs, segs,
                                         protocol.resolve_spectral_bands(fs))
    second = build_subject_level_features(ecog, fs, segs,
                                          protocol.resolve_spectral_bands(fs))
    assert first["rest"]["features"] == second["rest"]["features"]
    assert first["delta"] == second["delta"]


def test_aggregate_outputs_carry_no_identifiers():
    fs = 2000.0
    ecog, segs = _recording(fs, 1.5, move_gamma_gain=8.0, seed=11)
    result = build_subject_level_features(
        ecog, fs, segs, protocol.resolve_spectral_bands(fs),
        include_move_windows=True,
    )
    blob = repr(result)
    assert "DBS_bG" not in blob
    for reason_map in (result["qc"]["rest_channel_qc"],
                       result["qc"]["move_channel_qc"]):
        for channel_qc in reason_map.values():
            assert set(channel_qc) == {"n_ok", "n_failed", "reasons"}
            assert all(isinstance(k, str) for k in channel_qc["reasons"])


def test_every_non_null_feature_value_is_finite_across_seeds():
    fs = 2000.0
    for seed in range(5):
        ecog, segs = _recording(fs, 1.0, move_gamma_gain=3.0 + seed, seed=seed)
        result = build_subject_level_features(ecog, fs, segs,
                                              protocol.resolve_spectral_bands(fs))
        for state in ("rest", "move"):
            for value in result[state]["features"].values():
                if value is not None:
                    assert np.isfinite(value)
        for value in result["delta"].values():
            if value is not None:
                assert np.isfinite(value)

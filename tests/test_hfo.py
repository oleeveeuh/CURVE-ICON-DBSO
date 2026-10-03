"""HFO extraction: detection on synthetic signals, QC behavior, errors.

Integrity guarantees under test:

* A known spectral component is recovered.
* Invalid sampling rates / Nyquist violations raise clearly.
* Failed extraction is marked ``failed`` with ``None`` values — never
  zero, never randomly generated.
* Missing data raises ``MissingDataError``; there is no placeholder
  fallback anywhere in the module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.config import HFOConfig
from curve_icon_dbso.errors import DataValidationError, MissingDataError, SamplingRateError
from curve_icon_dbso.hfo import (
    aggregate_to_subject,
    extract_channel_features,
    extract_dataset,
    extract_recording_features,
    validate_sampling_rate,
)

FS = 2000.0
BAND = HFOConfig(low_hz=80.0, high_hz=500.0)


def _signal_with_tone(fs: float, seconds: float, tone_hz: float, tone_amp: float,
                      noise_sigma: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    signal = tone_amp * np.sin(2 * np.pi * tone_hz * t)
    signal += 0.3 * np.sin(2 * np.pi * 10.0 * t)  # background, far below band
    signal += rng.normal(0.0, noise_sigma, size=t.size)
    return signal


def test_known_tone_frequency_is_detected():
    signal = _signal_with_tone(FS, 5.0, tone_hz=250.0, tone_amp=1.0, noise_sigma=0.1)
    features = extract_channel_features(signal, FS, BAND)
    assert features["qc_status"] == "ok"
    assert features["qc_reason"] == ""
    assert features["hfo_power"] > 0.0
    assert abs(features["hfo_peak_freq"] - 250.0) <= 2.0


def test_power_reflects_tone_amplitude():
    quiet = extract_channel_features(
        _signal_with_tone(FS, 5.0, 250.0, tone_amp=0.5, noise_sigma=0.1), FS, BAND)
    loud = extract_channel_features(
        _signal_with_tone(FS, 5.0, 250.0, tone_amp=2.0, noise_sigma=0.1), FS, BAND)
    assert loud["hfo_power"] > quiet["hfo_power"]


def test_low_frequency_noise_gives_small_but_reported_power():
    """A signal with no in-band content still yields a real (small) PSD
    integral and an 'ok' status — a near-zero measurement is a
    measurement, and is distinguishable from a failure."""
    rng = np.random.default_rng(0)
    t = np.arange(int(FS * 5)) / FS
    signal = rng.normal(0.0, 0.05, size=t.size) + np.sin(2 * np.pi * 10.0 * t)
    features = extract_channel_features(signal, FS, BAND)
    assert features["qc_status"] == "ok"
    assert 0.0 < features["hfo_power"] < 0.01


def test_nyquist_violation_raises():
    with pytest.raises(SamplingRateError, match="Nyquist"):
        extract_channel_features(np.zeros(2000), fs=800.0, band=BAND)  # Nyquist=400 < 500


def test_invalid_sampling_rates_raise():
    for bad_fs in (0.0, -100.0, float("nan"), float("inf")):
        with pytest.raises(SamplingRateError):
            validate_sampling_rate(bad_fs, BAND)


def test_failed_channel_is_none_not_zero():
    """NaN-contaminated and empty channels must be recorded as failures
    with ``None`` features — distinguishable from a true zero."""
    for bad_signal in (np.array([np.nan] * 2000), np.array([], dtype=float)):
        features = extract_channel_features(bad_signal, FS, BAND)
        assert features["qc_status"] == "failed"
        assert features["hfo_power"] is None
        assert features["hfo_peak_freq"] is None
        assert features["qc_reason"]


def test_recording_features_keep_channel_rows():
    trial = np.vstack([
        _signal_with_tone(FS, 2.0, 250.0, 1.0, 0.1),          # good channel
        np.full((1, int(FS * 2)), np.nan),                    # failed channel
    ])
    frame = extract_recording_features(trial, FS, band=BAND, channel_labels=["good", "bad"])
    assert len(frame) == 2
    good = frame[frame["channel"] == "good"].iloc[0]
    bad = frame[frame["channel"] == "bad"].iloc[0]
    assert good["qc_status"] == "ok" and abs(good["hfo_peak_freq"] - 250.0) <= 2.0
    # In DataFrames failed features are NaN (pandas renders None as NaN):
    # still explicitly NOT zero.
    assert bad["qc_status"] == "failed"
    assert pd.isna(bad["hfo_power"])
    assert bad["hfo_power"] != 0


def test_missing_input_directory_raises_without_fallback(tmp_path: Path):
    """The historical bug: absent data silently triggered random
    placeholder features. Now it must raise, and no random generation
    may exist in the module."""
    with pytest.raises(MissingDataError, match="input directory not found"):
        extract_dataset(tmp_path / "does_not_exist")

    with pytest.raises(MissingDataError, match="No .mat files"):
        extract_dataset(tmp_path)  # exists but empty

    source = Path(__file__).resolve().parents[1] / "src" / "curve_icon_dbso" / "hfo.py"
    code = source.read_text()
    assert "np.random" not in code and "numpy.random" not in code, (
        "hfo.py must never generate random values (placeholder fallback regression)"
    )


def test_aggregate_separates_ok_from_failed(tmp_path: Path):
    channel_df = pd.DataFrame({
        "subject": ["S1", "S1", "S2"],
        "hfo_power": [1.0, 3.0, None],
        "hfo_peak_freq": [250.0, 270.0, None],
        "qc_status": ["ok", "ok", "failed"],
    })
    agg = aggregate_to_subject(channel_df).set_index("subject")
    assert agg.loc["S1", "hfo_power_mean"] == pytest.approx(2.0)
    assert agg.loc["S1", "n_failed_channels"] == 0
    assert agg.loc["S2", "n_ok_channels"] == 0
    assert agg.loc["S2", "n_failed_channels"] == 1
    assert pd.isna(agg.loc["S2", "hfo_power_mean"]), "failed subject must be NaN, not zero"


def test_mat_file_missing_raises(tmp_path: Path):
    from curve_icon_dbso.hfo import load_mat_file

    with pytest.raises(MissingDataError, match="not found"):
        load_mat_file(tmp_path / "absent.mat")


def test_unsupported_mat_structure_raises(tmp_path: Path):
    from scipy.io import savemat

    from curve_icon_dbso.hfo import load_mat_file

    path = tmp_path / "weird.mat"
    savemat(path, {"unrelated_field": np.zeros((2, 2))})
    with pytest.raises(DataValidationError, match="no supported data structure"):
        load_mat_file(path, fs_override=FS)

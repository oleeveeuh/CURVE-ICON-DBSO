"""Deterministic synthetic tests for the optional gated PAC module:
disabled-by-default behavior, the reliability gate, the surrogate gate on
coupled vs uncoupled signals, and determinism. Known waveforms only.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso.pac import (
    MIN_CYCLES,
    N_SURROGATES,
    SURROGATE_Z_MIN,
    compute_pac,
    mean_vector_length,
)

FS = 1000.0
SECONDS = 4.0


def _coupled_low(seconds: float = SECONDS, seed: int = 0) -> np.ndarray:
    """Low-band signal with a 25 Hz rhythm on broadband noise —
    non-periodic, so a circular time shift genuinely decorrelates it."""
    from curve_icon_dbso.neural_features import welch_psd  # noqa: F401 (import check)

    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    broadband = 0.3 * _pink_bg(int(seconds * FS), seed=seed)
    return broadband + 0.5 * np.sin(2 * np.pi * 25.0 * t) + 0.02 * rng.standard_normal(t.size)


def _pink_bg(n_samples: int, seed: int = 0) -> np.ndarray:
    """1/f background via FFT shaping (locally defined to keep the fixture
    self-contained)."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2 + 1
    spectrum = rng.standard_normal(half) + 1j * rng.standard_normal(half)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / FS)
    shaping = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9) ** 0.5, 0.0)
    series = np.fft.irfft(spectrum * shaping, n=n_samples)
    return series / (float(np.std(series)) or 1.0)


def _coupled(seconds: float = SECONDS, depth: float = 0.8,
             seed: int = 1) -> np.ndarray:
    """80 Hz carrier whose amplitude follows the PHASE of the coupled low
    signal (canonical beta-phase → gamma-amplitude coupling). The
    envelope is phase-locked but NOT periodic, so shift surrogates break
    the coupling instead of merely rotating the phasor."""
    from scipy.signal import hilbert

    from curve_icon_dbso.signal_qc import bandpass

    low = _coupled_low(seed=max(seed - 1, 0))
    t = np.arange(int(seconds * FS)) / FS
    beta_phase = np.angle(hilbert(bandpass(low, FS, 20.0, 35.0)))
    envelope = 1.0 + depth * np.cos(beta_phase)
    rng = np.random.default_rng(seed)
    return envelope * np.sin(2 * np.pi * 200.0 * t) + 0.05 * rng.standard_normal(t.size)


def _coupled_pair(seconds: float = SECONDS, depth: float = 0.8,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) sharing the SAME low signal — the high envelope is
    phase-locked to this exact low band."""
    low = _coupled_low(seconds=seconds, seed=seed)
    from scipy.signal import hilbert

    from curve_icon_dbso.signal_qc import bandpass

    t = np.arange(int(seconds * FS)) / FS
    beta_phase = np.angle(hilbert(bandpass(low, FS, 20.0, 35.0)))
    rng = np.random.default_rng(seed + 1)
    high = ((1.0 + depth * np.cos(beta_phase))
            * np.sin(2 * np.pi * 200.0 * t)
            + 0.05 * rng.standard_normal(t.size))
    return low, high


def _uncoupled(seconds: float = SECONDS, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Low phase at 25 Hz; high-band amplitude modulated by an
    INDEPENDENT 15 Hz rhythm — amplitude genuinely varies (so the
    surrogate z is not noise-dominated) but is unrelated to the phase."""
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    low = _coupled_low(seconds=seconds, seed=seed)
    envelope = 1.0 + 0.8 * np.sin(2 * np.pi * 15.0 * t)
    high = envelope * np.sin(2 * np.pi * 200.0 * t) + 0.05 * rng.standard_normal(t.size)
    return low, high


def test_pac_is_disabled_by_default_and_never_computes():
    result = compute_pac(*_coupled_pair(seed=0), FS)
    assert result["gate_pass"] is False
    assert result["enabled"] is False
    assert result["mvl"] is None
    assert result["reasons"] == ["pac_disabled_by_default"]


def test_gate_constants_are_strict():
    assert MIN_CYCLES >= 10
    assert N_SURROGATES >= 200
    assert SURROGATE_Z_MIN >= 2.0


def test_coupled_signal_passes_both_gates():
    low, high = _coupled_pair()
    result = compute_pac(low, high, FS, enabled=True,
                         high_band=(150.0, 250.0))
    assert result["enabled"] is True
    assert result["n_cycles"] >= MIN_CYCLES
    assert result["gate_pass"] is True, result["reasons"]
    assert result["surrogate_z"] >= SURROGATE_Z_MIN
    assert 0.0 < result["mvl"] <= 1.0


def test_uncoupled_signal_fails_the_surrogate_gate():
    low, high = _uncoupled()
    result = compute_pac(low, high, FS, enabled=True,
                         high_band=(150.0, 250.0))
    assert result["enabled"] is True
    assert result["gate_pass"] is False
    assert any("surrogate_gate_failed" in reason for reason in result["reasons"])
    assert result["surrogate_z"] < SURROGATE_Z_MIN


def test_short_signal_fails_the_reliability_gate():
    low, high = _coupled_pair(seconds=0.2)  # 0.2 s · 27.5 Hz ≈ 5.5 cycles < 10
    result = compute_pac(low, high, FS, enabled=True,
                         high_band=(150.0, 250.0))
    assert result["gate_pass"] is False
    assert any("reliability_gate_failed" in reason for reason in result["reasons"])


def test_pac_is_deterministic_given_the_seed():
    low, high = _coupled_pair()
    first = compute_pac(low, high, FS, enabled=True,
                        high_band=(150.0, 250.0), surrogate_seed=7)
    second = compute_pac(low, high, FS, enabled=True,
                         high_band=(150.0, 250.0), surrogate_seed=7)
    assert first["surrogate_z"] == second["surrogate_z"]
    assert first["mvl"] == second["mvl"]
    other = compute_pac(low, high, FS, enabled=True,
                        high_band=(150.0, 250.0), surrogate_seed=8)
    assert other["surrogate_z"] != first["surrogate_z"]


def test_mvl_bounds_and_shape_validation():
    low, high = _coupled_pair(seed=3)
    mvl = mean_vector_length(low, high, FS, (20.0, 35.0), (150.0, 250.0))
    assert 0.0 <= mvl <= 1.0
    with pytest.raises(Exception, match="same shape"):
        mean_vector_length(_coupled()[: -10], _coupled(), FS, (20.0, 35.0), (60.0, 100.0))

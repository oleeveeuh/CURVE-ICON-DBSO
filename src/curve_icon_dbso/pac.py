"""OPTIONAL phase-amplitude coupling (PAC) for the follow-on protocol —
strictly gated, exploratory, and OFF unless explicitly enabled.

The literature review records the caution (Köhler 2022: a repetitive-
tapping PAC study found no simple PAC difference despite slower
behavior): PAC is physiologically motivated but must not be assumed to
be a superior feature, and it is vulnerable to waveform-shape and
artifact confounds. The frozen contract therefore admits PAC only
behind strict reliability and surrogate-QC gates:

* the caller must pass ``enabled=True`` — PAC never runs silently;
* reliability gate: at least ``MIN_CYCLES`` cycles of the phase-
  providing (low) band must fit in the signal;
* surrogate gate: the observed mean-vector-length must reach
  ``SURROGATE_Z_MIN`` standard deviations above a null built from
  ``N_SURROGATES`` circular shifts of the amplitude series (seeded,
  deterministic; shifts are at least 1 s apart from zero so tiny shifts
  cannot survive in the null);
* every return carries the gate verdicts and machine-readable reasons;
  a gated-out result has ``gate_pass=False`` and must not enter any
  feature block.

MVL (mean vector length) convention: normalize the high-band amplitude
by its mean, average unit phasors ``exp(i·phase_low)`` weighted by it;
MVL ∈ [0, 1] (dimensionless). Units otherwise follow
``curve_icon_dbso.signal_qc`` (raw sensor units in, dimensionless PAC
out).

Pure functions on arrays; synthetic fixtures only in verification; no
target, fold, or identity access anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .errors import DataValidationError
from .signal_qc import bandpass

# Frozen gate constants (exploratory; loosening them is a contract change).
MIN_CYCLES: int = 10
N_SURROGATES: int = 200
SURROGATE_Z_MIN: float = 2.0
MIN_SURROGATE_SHIFT_S: float = 1.0


def mean_vector_length(low_signal: np.ndarray, high_signal: np.ndarray,
                       fs: float, low_band: tuple[float, float],
                       high_band: tuple[float, float]) -> float:
    """Tort-style MVL: phase of the band-passed low signal, amplitude of
    the band-passed high signal (mean-normalized), dimensionless [0, 1]."""
    if low_signal.shape != high_signal.shape:
        raise DataValidationError(
            f"low/high signals must have the same shape, got "
            f"{low_signal.shape} vs {high_signal.shape}"
        )
    low = bandpass(low_signal, fs, low_band[0], low_band[1])
    high = bandpass(high_signal, fs, high_band[0], high_band[1])
    phase = np.angle(hilbert(low))
    amplitude = np.abs(hilbert(high))
    mean_amplitude = float(np.mean(amplitude))
    if not np.isfinite(mean_amplitude) or mean_amplitude <= 0:
        raise DataValidationError("degenerate high-band amplitude; MVL undefined")
    normalized = amplitude / mean_amplitude
    phasors = np.exp(1j * phase)
    return float(np.abs(np.mean(normalized * phasors)))


def compute_pac(
    low_signal: np.ndarray,
    high_signal: np.ndarray,
    fs: float,
    low_band: tuple[float, float] = (20.0, 35.0),
    high_band: tuple[float, float] = (60.0, 100.0),
    enabled: bool = False,
    n_surrogates: int = N_SURROGATES,
    surrogate_seed: int = 42,
    min_cycles: int = MIN_CYCLES,
    surrogate_z_min: float = SURROGATE_Z_MIN,
) -> dict:
    """Gated PAC (low-band phase → high-band amplitude) with explicit
    verdicts.

    ``enabled=False`` (the default) short-circuits to a disabled record —
    PAC is never computed silently. With ``enabled=True``:

    * reliability gate: ``duration · low_band_center ≥ min_cycles``;
    * surrogate gate: z of the observed MVL against ``n_surrogates``
      seeded circular shifts of the amplitude series (each shift ≥
      ``MIN_SURROGATE_SHIFT_S`` away from the identity shift, preserving
      the amplitude autocorrelation/spectrum);
    * ``gate_pass`` is true only when both gates pass and every value is
      finite. A gated-out result carries its reasons and must not enter
      any feature block.

    Deterministic given the seed; synthetic-fixture verification only.
    """
    reasons: list[str] = []
    if not enabled:
        return {"gate_pass": False, "mvl": None, "surrogate_z": None,
                "n_cycles": None, "reasons": ["pac_disabled_by_default"],
                "enabled": False}
    low_signal = np.asarray(low_signal, dtype=float).ravel()
    high_signal = np.asarray(high_signal, dtype=float).ravel()
    if low_signal.size < 16:
        raise DataValidationError("signal too short for gated PAC")
    duration_s = low_signal.size / fs
    center_hz = float(np.mean(low_band))
    n_cycles = duration_s * center_hz
    if n_cycles < min_cycles:
        reasons.append(f"reliability_gate_failed ({n_cycles:.1f} < {min_cycles} cycles)")
    observed = mean_vector_length(low_signal, high_signal, fs, low_band, high_band)
    # Band-limited amplitude and phase are computed ONCE; surrogates are
    # circular shifts of the amplitude series (spectrum- and
    # autocorrelation-preserving, target-independent).
    amplitude = np.abs(hilbert(bandpass(high_signal, fs, high_band[0], high_band[1])))
    phase = np.angle(hilbert(bandpass(low_signal, fs, low_band[0], low_band[1])))
    mean_amplitude = float(np.mean(amplitude))
    rng = np.random.default_rng(surrogate_seed)
    n = low_signal.size
    min_shift = max(int(MIN_SURROGATE_SHIFT_S * fs), 1)
    surrogate_mvl: list[float] = []
    if mean_amplitude <= 0 or n <= 2 * min_shift:
        reasons.append("surrogate_qc_failed (signal shorter than the minimum shift)")
    else:
        shifts = rng.integers(min_shift, n - min_shift, size=n_surrogates)
        for shift in shifts:
            shifted = np.roll(amplitude, int(shift))
            surrogate_mvl.append(float(np.abs(np.mean(
                (shifted / mean_amplitude) * np.exp(1j * phase)
            ))))
        if len(surrogate_mvl) < n_surrogates:
            reasons.append("surrogate_qc_failed (degenerate surrogates)")
    null = np.asarray(surrogate_mvl, dtype=float)
    if null.size >= 2 and float(np.std(null)) > 0:
        z = float((observed - float(np.mean(null))) / float(np.std(null)))
    else:
        z = float("nan")
        reasons.append("surrogate_qc_failed (null distribution degenerate)")
    if not np.isfinite(z) or z < surrogate_z_min:
        reasons.append(f"surrogate_gate_failed (z={z:.2f} < {surrogate_z_min})")
    gate_pass = not reasons and np.isfinite(observed)
    return {
        "gate_pass": bool(gate_pass),
        "mvl": float(observed),
        "surrogate_z": z,
        "n_cycles": float(n_cycles),
        "n_surrogates": int(null.size),
        "reasons": reasons,
        "enabled": True,
    }

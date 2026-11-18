"""Deterministic synthetic tests for the OPTIONAL DMD representation
experiment (frozen contract §8): exact reconstruction of a linear
oscillatory system, conjugate-pair structure, deterministic output, the
frozen rank rule and feature dimensionality, frequency recovery, the
negative-control dataset where band power must decode but the
rank-truncated noise-dominated DMD subspace cannot, and the frozen
comparison discipline. Synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.dmd_representation import (
    DMD_RANK_CAP,
    dmd_edge_features,
    dmd_feature_frame,
    dmd_node_features,
    dmd_rank,
    fit_dmd,
    run_representation_comparison,
)
from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.move_modeling import WindowDataset
from curve_icon_dbso.neural_features import spectral_features, welch_psd

FS = 1000.0


# ---------------------------------------------------------------------------
# Reconstruction and eigenstructure of a known linear system
# ---------------------------------------------------------------------------


def _linear_system_window(n_channels: int = 3, n_samples: int = 400,
                          rho: float = 0.95, theta: float = 0.31,
                          seed: int = 0) -> np.ndarray:
    """Trajectory of x_{t+1} = A x_t with A = diag(ρ·R(θ), 0.5) — a
    damped rotation plus an independent decaying channel."""
    rng = np.random.default_rng(seed)
    A = np.zeros((3, 3))
    A[0, 0] = rho * np.cos(theta)
    A[0, 1] = -rho * np.sin(theta)
    A[1, 0] = rho * np.sin(theta)
    A[1, 1] = rho * np.cos(theta)
    A[2, 2] = 0.5
    x = np.zeros((3, n_samples))
    x[:, 0] = rng.normal(0, 1.0, 3)
    for t in range(1, n_samples):
        x[:, t] = A @ x[:, t - 1]
    return x


def test_dmd_reconstructs_a_known_linear_system():
    window = _linear_system_window()
    fit = fit_dmd(window)
    assert fit["rank"] == 3  # dmd_rank(3)
    # Exact linear system: reconstruction at machine precision.
    assert fit["reconstruction_relative_error"] < 1e-6
    expected = {round(0.95, 6), 0.5}
    magnitudes = sorted({round(abs(lam), 3) for lam in fit["eigenvalues"]})
    # |λ| for the rotation pair is ρ (0.95); the decaying channel is 0.5.
    assert magnitudes == pytest.approx(sorted(expected), abs=1e-2)


def test_complex_eigenvalues_come_in_conjugate_pairs():
    window = _linear_system_window(theta=0.4)
    eigenvalues = fit_dmd(window)["eigenvalues"]
    complex_modes = [lam for lam in eigenvalues
                     if abs(lam.imag) > 1e-9]
    assert complex_modes, "the rotation must produce complex eigenvalues"
    remaining = list(complex_modes)
    for lam in complex_modes:
        conjugate = np.conj(lam)
        match = min(remaining, key=lambda other: abs(other - conjugate))
        assert abs(match - conjugate) < 1e-8
        remaining.remove(match)
    assert not remaining


# ---------------------------------------------------------------------------
# Determinism, rank rule, dimensionality
# ---------------------------------------------------------------------------


def test_dmd_output_is_deterministic():
    rng = np.random.default_rng(1)
    window = rng.normal(0, 1.0, (6, 500)) + 0.5 * np.sin(
        2 * np.pi * 20 * np.arange(500) / FS
    )
    first = dmd_node_features(window)
    second = dmd_node_features(window)
    assert first[0] == second[0]
    assert first[1]["compute_seconds"] == second[1]["compute_seconds"] or True
    # The features (not the timing diagnostic) must be bitwise identical.
    for key in first[0]:
        assert first[0][key] == second[0][key]


def test_rank_rule_is_frozen_min_8():
    assert dmd_rank(1) == 1
    assert dmd_rank(4) == 4
    assert dmd_rank(8) == 8
    assert dmd_rank(32) == DMD_RANK_CAP == 8
    with pytest.raises(DataValidationError, match="at least 1"):
        dmd_rank(0)


def test_node_feature_dimensionality_without_region_map():
    rng = np.random.default_rng(2)
    window = rng.normal(0, 1.0, (4, 500))
    features, diagnostics = dmd_node_features(window)
    rank = diagnostics["rank"]
    assert rank == 4
    assert len(features) == 4 * rank  # 3 mode dynamics + 1 node magnitude
    for k in range(rank):
        for suffix in ("log_abs_eigenvalue", "frequency_cycles_per_sample",
                       "log_amplitude", "node_magnitude"):
            assert f"mode{k}_{suffix}" in features
    # The one-step diagnostic is always finite and interpretable; the
    # multi-step trajectory error is present (≥ 0) only when the fitted
    # spectrum is stable, else None with an explicit status (v1.0.1 —
    # an unstable evolution's trajectory error measures divergence, not
    # misfit, and is never passed off as one).
    assert diagnostics["one_step_relative_error"] >= 0
    assert np.isfinite(diagnostics["one_step_relative_error"])
    if diagnostics["reconstruction_status"] == "computed":
        assert diagnostics["reconstruction_relative_error"] >= 0
        assert diagnostics["n_unstable_modes"] == 0
    else:
        assert diagnostics["reconstruction_status"] == (
            "skipped_unstable_eigenvalues"
        )
        assert diagnostics["reconstruction_relative_error"] is None
        assert diagnostics["n_unstable_modes"] > 0
    assert diagnostics["compute_seconds"] > 0


# ---------------------------------------------------------------------------
# v1.0.1 corrective: conjugate projection, one basis, interpretable
# reconstruction diagnostics
# ---------------------------------------------------------------------------


def test_stabilized_amplitudes_use_the_conjugate_transpose():
    """With rank = n_channels the orthonormalized basis Q is square
    unitary, so Q·(Qᴴx₀) must reproduce x₀ EXACTLY. For the complex
    rotation modes this fails if the plain transpose Q.T is used instead
    of the adjoint (the pre-v1.0.1 defect)."""
    window = _linear_system_window()  # 3 channels, rank 3 → square Q
    fit = fit_dmd(window)
    q = fit["modes_stabilized"]
    b = fit["amplitudes_stabilized"]
    assert np.iscomplexobj(q), "the rotation must yield complex modes"
    reconstruction = q @ b
    assert np.linalg.norm(reconstruction - window[:, 0]) < 1e-8
    # And the orthonormality itself is the complex one: QᴴQ = I.
    assert np.linalg.norm(q.conj().T @ q - np.eye(fit["rank"])) < 1e-10


def test_node_magnitudes_live_in_the_stabilized_basis():
    """Node magnitudes must be |Q_{i,k}|·|b_k| — the SAME basis as the
    amplitudes — not raw-Φ magnitudes times Q-basis amplitudes (the
    pre-v1.0.1 basis mismatch)."""
    rng = np.random.default_rng(5)
    window = rng.normal(0, 1.0, (6, 500))
    features, diagnostics = dmd_node_features(window)
    fit = fit_dmd(window)
    q = fit["modes_stabilized"]
    b = fit["amplitudes_stabilized"]
    rank = diagnostics["rank"]
    expected = np.abs(q) * np.abs(b)[None, :]
    for k in range(rank):
        assert features[f"mode{k}_node_magnitude"] == pytest.approx(
            float(np.median(expected[:, k])), rel=1e-12
        )


def test_unstable_spectrum_gets_interpretable_diagnostics():
    """An explosive fitted spectrum (|λ| > 1.05) must NOT produce a
    replayed-pseudo-trajectory error (the pre-v1.0.1 ≈10³ numbers): the
    multi-step trajectory error is None with an explicit status, and the
    one-step diagnostic stays a real one-step prediction — machine-exact
    for a short exact linear window, finite (conditioning-limited) for a
    long one whose 10³¹ dynamic range strains the pseudoinverse."""
    rho, theta = 1.2, 0.31  # unstable rotation
    window = _linear_system_window(rho=rho, theta=theta)
    fit = fit_dmd(window)
    magnitudes = [abs(lam) for lam in fit["eigenvalues"]]
    assert max(magnitudes) > 1.05
    assert fit["n_unstable_modes"] >= 2  # the conjugate pair
    assert fit["reconstruction_status"] == "skipped_unstable_eigenvalues"
    assert fit["reconstruction_relative_error"] is None
    assert np.isfinite(fit["one_step_relative_error"])
    # Short window: the one-step prediction of the exact linear system
    # is machine-exact regardless of eigenvalue stability.
    short = _linear_system_window(rho=rho, theta=theta, n_samples=60)
    assert fit_dmd(short)["one_step_relative_error"] < 1e-8


def test_feature_frame_summarizes_reconstruction_diagnostics():
    rng = np.random.default_rng(6)
    windows = [rng.normal(0, 1.0, (5, 400)) for _ in range(6)]
    _frame, diagnostics = dmd_feature_frame(windows)
    assert diagnostics["n_windows"] == 6
    assert diagnostics["mean_one_step_relative_error"] >= 0
    assert 0.0 <= diagnostics["multistep_computed_fraction"] <= 1.0
    assert diagnostics["mean_n_unstable_modes"] >= 0
    assert diagnostics["per_window_seconds"] > 0


def test_region_map_changes_dimensionality_and_never_compares_positions():
    rng = np.random.default_rng(3)
    window = rng.normal(0, 1.0, (4, 500))
    region_map = {0: "M1", 1: "M1", 2: "S1", 3: "PM"}
    features, diagnostics = dmd_node_features(window, region_map=region_map)
    rank = diagnostics["rank"]
    # Frozen region order: all protocol regions get columns (absent
    # regions are 0.0, never fabricated from other channels).
    assert len(features) == 3 * rank + rank * len(protocol.REGION_LABELS)
    for k in range(rank):
        assert features[f"mode{k}_region_M1_magnitude"] > 0
        assert features[f"mode{k}_region_other_unknown_magnitude"] == 0.0
    with pytest.raises(DataValidationError, match="missing OK channel"):
        dmd_node_features(window, region_map={0: "M1"})


def test_failed_channels_are_excluded_never_imputed():
    rng = np.random.default_rng(4)
    window = rng.normal(0, 1.0, (5, 500))
    window[2, :] = np.nan
    features, diagnostics = dmd_node_features(window)
    assert diagnostics["n_ok_channels"] == 4
    assert diagnostics["rank"] == 4  # min(8, 4 OK channels)


# ---------------------------------------------------------------------------
# Oscillatory-system frequency recovery
# ---------------------------------------------------------------------------


def test_dmd_recovers_known_oscillation_frequencies():
    t = np.arange(500) / FS
    window = np.vstack([
        np.sin(2 * np.pi * 10.0 * t),
        np.sin(2 * np.pi * 10.0 * t + 0.7),
        np.sin(2 * np.pi * 25.0 * t),
        np.sin(2 * np.pi * 25.0 * t + 1.3),
    ])
    fit = fit_dmd(window)
    recovered = sorted({
        round(abs(lam.imag) / (2 * np.pi) * FS, 1)
        for lam in fit["eigenvalues"] if abs(lam.imag) > 0.05
    })
    assert any(abs(f - 10.0) < 0.5 for f in recovered), recovered
    assert any(abs(f - 25.0) < 0.5 for f in recovered), recovered


# ---------------------------------------------------------------------------
# Negative control: power decodes, rank-truncated DMD does not
# ---------------------------------------------------------------------------


def _negative_control(n_subjects: int = 6, windows_per: int = 20,
                      seed: int = 7):
    """Negative control where DMD must NOT outperform power.

    16 channels carry EIGHT fixed, high-variance interference patterns
    (random spatial maps × fixed smooth temporal signals) plus a WEAK
    10 Hz tone whose amplitude per window is the target. The interference
    occupies the frozen rank-8 DMD subspace by construction (its modes
    dominate the variance), so the DMD node features cannot represent the
    tone — while the alpha-band relative power resolves it directly. This
    is the honest failure mode of a rank-truncated DMD on
    interference-dominated data, and the reason the contract compares it
    against power on held-out subjects instead of assuming superiority.
    """
    rng = np.random.default_rng(seed)
    n_channels, n_samples = 16, 500
    t = np.arange(n_samples) / FS
    # Fixed interference basis: 8 ORTHONORMAL spatial maps × 8 mutually
    # ORTHOGONAL temporal signals — the static block's singular values
    # are then all equal and strictly exceed the tone's, so the frozen
    # rank-8 truncation excludes the tone in every realization (no
    # borderline slippage).
    maps_raw = rng.normal(0, 1.0, (n_channels, 8))
    maps, _ = np.linalg.qr(maps_raw)          # orthonormal columns
    freqs_int = rng.uniform(1.0, 3.0, 8)
    phases = rng.uniform(0, 2 * np.pi, 8)
    signals_raw = np.vstack([
        np.sin(2 * np.pi * f * t + ph)
        for f, ph in zip(freqs_int, phases, strict=True)
    ])
    _, _, vh = np.linalg.svd(signals_raw, full_matrices=False)
    signals_orth = vh  # (8 × n_samples), already mutually orthonormal rows
    # Scale so every interference singular value (~141 after the
    # √(16·499/2) coherent-tone scale) sits FAR above the tone's
    # (amp ≤ 0.48 → σ ≤ ~30): the frozen rank-8 truncation then
    # excludes the tone in every realization.
    static = maps @ (signals_orth * np.sqrt(2.0)) * 100.0
    windows, targets, groups_list = [], [], []
    for subject in range(n_subjects):
        for _ in range(windows_per):
            amplitude = float(rng.uniform(0.16, 0.48))
            window = (
                static
                + amplitude * np.sin(2 * np.pi * 10.0 * t)[None, :]
                + 0.1 * rng.standard_normal((n_channels, n_samples))
            )
            windows.append(window)
            targets.append(amplitude)
            groups_list.append(f"s{subject:02d}")
    # Band-power features (median across channels, frozen aggregation).
    power_rows = []
    for window in windows:
        row: dict = {}
        per_channel_alpha = []
        for channel in window:
            freqs, psd = welch_psd(channel, FS)
            features = spectral_features(
                freqs, psd, {"bands_active": dict(protocol.STANDARD_BANDS_HZ),
                             "gamma_high": tuple(protocol.HIGH_GAMMA_DEFAULT)},
                100.0,
            )
            for key, value in features.items():
                if key.endswith("log_relative_power") or key.startswith(
                    "aperiodic_exponent"
                ):
                    row[key] = value if value is not None else 0.0
            if features["alpha_log_relative_power"] is not None:
                per_channel_alpha.append(features["alpha_log_relative_power"])
        row["alpha_log_relative_power"] = float(np.median(per_channel_alpha))
        power_rows.append(row)
    power_frame = pd.DataFrame(power_rows)
    y = np.asarray(targets)
    groups = np.asarray(groups_list)
    metadata = pd.DataFrame({
        "subject_position": groups,
        "recording_position": groups,
        "block": 0,
        "target_bin": np.arange(len(targets)) % windows_per,
        "target_time_s": np.arange(len(targets)) * 0.1,
        "sample_in_recording": np.arange(len(targets)) * 100,
    })
    return windows, power_frame, y, groups, metadata


def test_negative_control_power_decodes_dmd_does_not():
    windows, power_frame, y, groups, metadata = _negative_control()
    # Band power resolves the tone by construction.
    alpha = power_frame["alpha_log_relative_power"].to_numpy()
    corr_power = float(np.corrcoef(alpha, y)[0, 1])
    assert abs(corr_power) > 0.3, corr_power
    del alpha
    # The rank-truncated DMD must NOT outperform band power. (A small
    # residual correlation is intrinsic: the unmodeled coherent tone
    # perturbs the fitted operator through its spatial overlap with the
    # interference subspace — a real DMD property, not a defect. The
    # negative-control criterion is "no out-performance".)
    dmd_frame, _ = dmd_feature_frame(windows)
    best = 0.0
    for column in dmd_frame.columns:
        values = dmd_frame[column].to_numpy()
        if np.std(values) == 0:
            continue
        best = max(best, abs(float(np.corrcoef(values, y)[0, 1])))
    assert best < corr_power, (best, corr_power)


def test_frozen_comparison_identical_folds_and_discipline():
    windows, power_frame, y, groups, metadata = _negative_control(4, 12)
    dataset = WindowDataset(
        features=power_frame, target=y, groups=groups, metadata=metadata,
    )
    dmd_frame, diagnostics = dmd_feature_frame(windows)
    runs = run_representation_comparison(dataset, dmd_frame,
                                        power_frame=power_frame)
    assert set(runs) == {"multiband_power", "dmd_nodes",
                         "combined_power_plus_dmd_nodes"}
    # Identical held-out folds: same subjects, same window counts.
    for _name, result in runs.items():
        assert result.summary["n_subjects"] == 4
        assert result.summary["n_windows"] == len(y)
    # The negative control: power must not LOSE to DMD on held-out
    # subjects (and on this control it should win outright).
    power_r = runs["multiband_power"].summary["median_pearson_r"]
    dmd_r = runs["dmd_nodes"].summary["median_pearson_r"]
    assert power_r is not None and dmd_r is not None
    assert power_r > dmd_r
    # Cost reporting is part of the comparison contract.
    assert diagnostics["per_window_seconds"] > 0
    # Edge features are a gated secondary ablation.
    edge_pairs = [(0, 1), (1, 2)]
    edge_frame = pd.DataFrame(
        [dmd_edge_features(window, edge_pairs)[0] for window in windows]
    )
    gated = run_representation_comparison(
        dataset, dmd_frame, with_edges=True, edge_frame=edge_frame,
        power_frame=power_frame,
    )
    assert "dmd_edges_secondary" in gated
    with pytest.raises(DataValidationError, match="edge_frame"):
        run_representation_comparison(dataset, dmd_frame, with_edges=True)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_no_in_sample_superiority_claims():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src"
              / "curve_icon_dbso" / "dmd_representation.py").read_text()
    normalized = " ".join(source.split())
    assert "No novelty or superiority is claimed" in normalized
    assert "held-out" in normalized


def test_rank_is_never_tuned():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src"
              / "curve_icon_dbso" / "dmd_representation.py").read_text()
    assert "single frozen rank rule" in source
    assert "GridSearchCV" not in source

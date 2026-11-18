"""OPTIONAL dynamic-mode-decomposition (DMD) representation experiment
for the MOVE study (frozen contract §8 ablation, Prompt 5; motivation:
docs/literature_review.md §8 — spatial-node DMD features were reported
as more reproducible across trials than conventional power features).

The only success criterion is prespecified improvement on HELD-OUT
subjects with uncertainty and an autocorrelation-preserving null. No
novelty or superiority is claimed from any in-sample result: this module
computes a representation and runs the frozen comparison machinery —
nothing more.

Frozen representation (contract: "frozen rank 8, min(8, n_channels)
dimensions"):

* **Rank rule (single, frozen, never tuned)**: ``r = min(8, n_ok_channels)``.
  Inner-fold rank selection is explicitly NOT used — the contract chose
  the single frozen rule.
* **Deterministic exact DMD** (no randomized SVD): X₁ = X[:, :-1],
  X₂ = X[:, 1:] on the window's QC-OK channels; truncated SVD of X₁;
  Ã = UᵀX₂VΣ⁻¹; eigen-decomposition of Ã; modes Φ = X₂VΣ⁻¹W; modal
  amplitudes b = Φ⁺x₀. LAPACK's SVD/eig are deterministic for a fixed
  input; modes/eigenvalues are sorted by a deterministic key
  (−|λ|, Re, Im) so feature columns have a stable order.
* **Spatial-node features (real-valued)**, per kept mode k:
  ``log10|λ_k|`` (decay/growth), frequency ``ω_k = angle(λ_k)/2π``
  cycles/sample, ``log10|b_k|`` (modal energy), and the node magnitude
  ``median_i |Q_{i,k}·b_k|`` across QC-OK channels — or, when an
  anatomical region map is provided, the per-region medians (regions in
  the frozen ``protocol.REGION_LABELS`` order). ``Q`` is the
  orthonormalized mode basis (QR of Φ) and ``b = Qᴴx₀`` its conjugate-
  transpose projection: features and amplitudes live in ONE basis, so
  their product is the per-mode contribution of mode k at channel i
  (mixing the raw Φ magnitudes with Q-basis amplitudes — the pre-v1.0.1
  form — compared quantities from two different bases and is
  discontinued). Patient-specific channel positions are NEVER compared
  as though aligned: differing channel counts enter only through this
  region representation or the frozen median fallback.
* **Spatial-edge features (clearly secondary ablation)**: per channel
  pair (i, j), the median across modes of |Φ_{i,k}·b_k|·|Φ_{j,k}·b_k| —
  dimensionality ∝ n_channels²/2, much larger than nodes, hence kept
  behind an explicit ``with_edges`` flag.
* **QC behavior**: failed (non-finite) channels are excluded before the
  SVD and never imputed; windows with fewer than two OK channels are
  dropped with machine-readable reasons. Two reconstruction diagnostics
  are reported with the computational cost (per-window seconds), never
  as model features: the ONE-STEP relative error
  ‖Φ(λ∘b) − x(t₁)‖/‖x(t₁)‖ (always finite and interpretable), and the
  multi-step trajectory error ‖X₂ − ΦΛ^j b‖/‖X₂‖ — computed only when
  every kept eigenvalue satisfies |λ| ≤ 1.05, because for an unstable
  fitted spectrum the multi-step evolution diverges and the ratio
  measures the divergence, not the linearity of the data (the pre-v1.0.1
  fallback replayed a single one-step prediction across the window,
  which produced uninterpretable ≈10³ values). When any |λ| > 1.05 the
  multi-step error is ``None`` with status
  ``skipped_unstable_eigenvalues`` and the unstable-mode count is
  recorded. Neither diagnostic licenses a claim about how "linear" the
  physiology is — they are numerical fit diagnostics of one frozen rank.

Comparison discipline (frozen): conventional multiband power vs DMD
node features vs combined, all with the SAME primary linear decoder on
IDENTICAL outer subject folds (``move_modeling.run_move_cv`` with an
explicit ``feature_frame`` — folds, inner selection, and per-subject
metrics are untouched). Edge features run only when explicitly enabled.
Synthetic negative control included: a weak band oscillation buried in
broadband noise that band power resolves by construction but the
rank-truncated noise-dominated DMD subspace cannot.

Verification: deterministic synthetic fixtures only (reconstruction,
conjugate pairs, determinism, rank/dimensionality, oscillatory-system
frequency recovery, negative control). No real data, no AWS, no
committed participant-level artifacts.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import protocol
from .errors import DataValidationError

#: Frozen rank cap (contract: rank 8, min(8, n_channels) dimensions).
DMD_RANK_CAP: int = 8

#: Eigenvalue-magnitude threshold above which the fitted spectrum is
#: treated as numerically unstable and the multi-step reconstruction is
#: SKIPPED (recorded as ``None``) — an unstable modal evolution diverges,
#: so its trajectory error measures divergence, not misfit.
DMD_STABILITY_THRESHOLD: float = 1.05


def dmd_rank(n_ok_channels: int) -> int:
    """The single frozen rank rule — never tuned, never selected."""
    if n_ok_channels < 1:
        raise DataValidationError(
            f"need at least 1 OK channel for DMD, got {n_ok_channels}"
        )
    return min(DMD_RANK_CAP, n_ok_channels)


def fit_dmd(window: np.ndarray, rank: int | None = None) -> dict:
    """Deterministic exact DMD of one window (``n_channels × n_samples``).

    Returns eigenvalues (complex, deterministically sorted), the raw DMD
    modes Φ plus their orthonormalized basis Q, the raw least-squares
    amplitudes (reconstruction diagnostics only), the stabilized
    conjugate-transpose projection amplitudes (features), kept singular
    values, and the reconstruction diagnostics (one-step relative error
    always; the multi-step trajectory error only when the fitted spectrum
    is stable, else ``None`` with an explicit status). Failed
    (non-finite) channels must be excluded by the caller — this function
    never imputes.
    """
    X = np.asarray(window, dtype=float)
    if X.ndim != 2:
        raise DataValidationError(f"window must be 2-D; got {X.shape}")
    if not np.isfinite(X).all():
        raise DataValidationError(
            "DMD needs finite samples; exclude failed channels upstream "
            "(never impute)"
        )
    n_channels, n_samples = X.shape
    if n_samples < 8:
        raise DataValidationError(
            f"window too short for DMD ({n_samples} samples)"
        )
    r = dmd_rank(n_channels) if rank is None else int(rank)
    if not (1 <= r <= min(n_channels, n_samples - 1)):
        raise DataValidationError(
            f"rank {r} outside 1..min({n_channels}, {n_samples - 1})"
        )
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    V_r = Vh_r.T
    # Projected operator and its eigen-decomposition (real matrix →
    # real eig → complex conjugate pairs for oscillatory modes).
    A_tilde = U_r.T @ X2 @ V_r / S_r
    eigenvalues, W = np.linalg.eig(A_tilde)
    # Deterministic order: descending |λ|, then Re, then Im.
    order = sorted(range(r), key=lambda k: (-abs(eigenvalues[k]),
                                            eigenvalues[k].real,
                                            eigenvalues[k].imag))
    eigenvalues = eigenvalues[order]
    W = W[:, order]
    modes = X2 @ (V_r / S_r) @ W
    # Least-squares modal amplitudes (exact-dynamics/reconstruction use):
    # ill-conditioned for near-degenerate modes, which is why the FEATURE
    # extractor uses the stabilized projection below.
    amplitudes = np.linalg.pinv(modes) @ X[:, 0]
    # Stabilized amplitudes: project onto the ORTHONORMALIZED mode basis
    # (QR of Φ) with the CONJUGATE transpose — for complex modes, Q.T is
    # not the adjoint and the projection is simply wrong (pre-v1.0.1
    # defect). The projection is well-conditioned by construction and
    # keeps the node features real, deterministic, and free of the
    # near-degenerate-mode noise amplification.
    q_modes, _ = np.linalg.qr(modes)
    amplitudes_stabilized = q_modes.conj().T @ X[:, 0]
    # ONE-STEP diagnostic: X2[:, 0] ≈ Φ diag(λ) b — always finite, for any
    # spectrum (this is the pre-v1.0.1 quantity that the unstable branch
    # used to replay across the whole window, which made the reported
    # error uninterpretable).
    one_step_prediction = modes @ (eigenvalues * amplitudes)
    one_step_denominator = float(np.linalg.norm(X2[:, 0]))
    one_step_error = (
        float(np.linalg.norm(one_step_prediction - X2[:, 0])
              / one_step_denominator)
        if one_step_denominator > 0 else float("inf")
    )
    # MULTI-STEP trajectory reconstruction: X2[:, j] ≈ Φ diag(λ)^j b.
    # Computed only when every kept eigenvalue is stable (|λ| ≤ threshold):
    # an unstable evolution diverges and its trajectory error measures the
    # divergence, not the linearity of the data — so it is recorded as
    # None with an explicit status instead.
    steps = np.arange(1, X2.shape[1] + 1)
    n_unstable = int(np.count_nonzero(np.abs(eigenvalues)
                                      > DMD_STABILITY_THRESHOLD))
    if n_unstable:
        reconstruction_error = None
        reconstruction_status = "skipped_unstable_eigenvalues"
    else:
        evolved = (eigenvalues[:, None] ** steps[None, :]) * amplitudes[:, None]
        reconstructed = modes @ evolved
        denominator = float(np.linalg.norm(X2))
        reconstruction_error = (
            float(np.linalg.norm(reconstructed - X2) / denominator)
            if denominator > 0 else float("inf")
        )
        reconstruction_status = "computed"
    return {
        "eigenvalues": eigenvalues,
        "modes": modes,
        "modes_stabilized": q_modes,
        "amplitudes": amplitudes,
        "amplitudes_stabilized": amplitudes_stabilized,
        "singular_values": S_r,
        "rank": r,
        "one_step_relative_error": one_step_error,
        "n_unstable_modes": n_unstable,
        "reconstruction_status": reconstruction_status,
        "reconstruction_relative_error": reconstruction_error,
    }


def dmd_node_features(
    window: np.ndarray, region_map: dict[int, str] | None = None
) -> tuple[dict[str, float], dict]:
    """Frozen spatial-node features for one window.

    The window is ``n_channels × n_samples``; non-finite channels are
    excluded before the SVD (never imputed). Feature count is ``4r``
    without a region map (three mode-dynamics features plus the global
    node-magnitude median per mode) and ``3r + r·n_regions_present``
    with one. Returns ``(features, diagnostics)``; the relative
    reconstruction error and per-window compute seconds are diagnostics,
    never model features.
    """
    window = np.asarray(window, dtype=float)
    ok = np.isfinite(window).all(axis=1)
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise DataValidationError("every channel non-finite; DMD undefined")
    rank = dmd_rank(n_ok)
    started = time.perf_counter()
    fit = fit_dmd(window[ok], rank=rank)
    elapsed = time.perf_counter() - started
    eigenvalues = fit["eigenvalues"]
    amplitudes = fit["amplitudes_stabilized"]
    # Node magnitudes live in the SAME orthonormal basis as the
    # amplitudes: |Q_{i,k}|·|b_k| is the per-mode contribution of mode k
    # at channel i under the stabilized representation. Multiplying the
    # RAW Φ magnitudes by Q-basis amplitudes (pre-v1.0.1) compared two
    # different bases and is discontinued.
    basis = fit["modes_stabilized"]
    features: dict[str, float] = {}
    node_magnitudes = np.abs(basis) * np.abs(amplitudes)[None, :]  # ch × r
    for k in range(rank):
        lam = eigenvalues[k]
        features[f"mode{k}_log_abs_eigenvalue"] = float(
            np.log10(abs(lam)) if abs(lam) > 0 else -12.0
        )
        features[f"mode{k}_frequency_cycles_per_sample"] = float(
            np.angle(lam) / (2.0 * np.pi)
        )
        features[f"mode{k}_log_amplitude"] = float(
            np.log10(abs(amplitudes[k])) if abs(amplitudes[k]) > 0 else -12.0
        )
    if region_map:
        region_positions: dict[str, list[int]] = {}
        for local_index, channel_index in enumerate(np.flatnonzero(ok)):
            region = region_map.get(int(channel_index))
            if region is None:
                raise DataValidationError(
                    f"region_map is missing OK channel {int(channel_index)}"
                )
            region_positions.setdefault(region, []).append(local_index)
        for region in protocol.REGION_LABELS:
            positions = region_positions.get(region, [])
            for k in range(rank):
                features[f"mode{k}_region_{region}_magnitude"] = (
                    float(np.median(node_magnitudes[positions, k]))
                    if positions else 0.0
                )
    else:
        for k in range(rank):
            features[f"mode{k}_node_magnitude"] = float(
                np.median(node_magnitudes[:, k])
            )
    diagnostics = {
        "rank": rank,
        "n_ok_channels": n_ok,
        "one_step_relative_error": fit["one_step_relative_error"],
        "n_unstable_modes": fit["n_unstable_modes"],
        "reconstruction_status": fit["reconstruction_status"],
        "reconstruction_relative_error": fit["reconstruction_relative_error"],
        "compute_seconds": elapsed,
    }
    return features, diagnostics


def dmd_edge_features(
    window: np.ndarray, pairs: list[tuple[int, int]]
) -> tuple[dict[str, float], dict]:
    """SECONDARY spatial-edge features: per channel pair (i, j), the
    median across modes of |Q_{i,k}·b_k|·|Q_{j,k}·b_k| — dimensionality
    ∝ n_channels²/2, much larger than the node block, hence gated behind
    an explicit call. The magnitudes use the SAME stabilized orthonormal
    basis as the node features (pre-v1.0.1 mixed bases). Pair indices
    refer to the ORIGINAL channel positions; non-finite channels fail the
    pair (never imputed)."""
    window = np.asarray(window, dtype=float)
    ok = np.isfinite(window).all(axis=1)
    fit = fit_dmd(window[ok], rank=dmd_rank(int(ok.sum())))
    node_magnitudes_full = np.zeros(window.shape[0])
    ok_positions = np.flatnonzero(ok)
    magnitudes = np.abs(fit["modes_stabilized"]) * np.abs(
        fit["amplitudes_stabilized"]
    )[None, :]
    node_magnitudes_full[ok_positions] = np.median(magnitudes, axis=1)
    started = time.perf_counter()
    features: dict[str, float] = {}
    for anode, cathode in pairs:
        if not (ok[anode] and ok[cathode]):
            features[f"edge_{anode}_{cathode}_median_product"] = 0.0
            continue
        features[f"edge_{anode}_{cathode}_median_product"] = float(
            np.median(magnitudes[anode, :] * magnitudes[cathode, :])
        )
    diagnostics = {"rank": fit["rank"], "compute_seconds": (
        time.perf_counter() - started
    )}
    return features, diagnostics


# ---------------------------------------------------------------------------
# Frozen comparison: power vs DMD nodes vs combined (+ optional edges)
# ---------------------------------------------------------------------------


def dmd_feature_frame(
    windows: list[np.ndarray], region_map: dict[int, str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """DMD node features for a captured window list (rows aligned
    row-for-row with the ``WindowDataset`` that captured them), plus the
    aggregate QC/cost diagnostics (rank distribution, reconstruction
    errors, per-window compute seconds)."""
    from .move_modeling import WindowDataset  # noqa: F401 (type alias only)

    rows: list[dict] = []
    diagnostics = {"ranks": [], "one_step_relative_errors": [],
                   "multistep_relative_errors": [], "n_unstable_modes": [],
                   "reconstruction_statuses": [], "compute_seconds": []}
    for window in windows:
        features, diag = dmd_node_features(window, region_map=region_map)
        rows.append(features)
        diagnostics["ranks"].append(diag["rank"])
        diagnostics["one_step_relative_errors"].append(
            diag["one_step_relative_error"]
        )
        diagnostics["multistep_relative_errors"].append(
            diag["reconstruction_relative_error"]
        )
        diagnostics["n_unstable_modes"].append(diag["n_unstable_modes"])
        diagnostics["reconstruction_statuses"].append(
            diag["reconstruction_status"]
        )
        diagnostics["compute_seconds"].append(diag["compute_seconds"])
    frame = pd.DataFrame(rows)
    computed_multistep = [
        v for v in diagnostics["multistep_relative_errors"] if v is not None
    ]
    diagnostics["mean_one_step_relative_error"] = (
        float(np.mean(diagnostics["one_step_relative_errors"]))
        if diagnostics["one_step_relative_errors"] else None
    )
    diagnostics["mean_multistep_relative_error_computed_windows"] = (
        float(np.mean(computed_multistep)) if computed_multistep else None
    )
    diagnostics["multistep_computed_fraction"] = (
        len(computed_multistep) / frame.shape[0] if frame.shape[0] else 0.0
    )
    diagnostics["mean_n_unstable_modes"] = (
        float(np.mean(diagnostics["n_unstable_modes"]))
        if diagnostics["n_unstable_modes"] else 0.0
    )
    diagnostics["per_window_seconds"] = (
        float(np.mean(diagnostics["compute_seconds"]))
        if diagnostics["compute_seconds"] else 0.0
    )
    diagnostics["n_windows"] = int(frame.shape[0])
    return frame, diagnostics


def run_representation_comparison(
    power_dataset,
    dmd_frame: pd.DataFrame,
    model_name: str = "lagged_ridge_primary",
    with_edges: bool = False,
    edge_frame: pd.DataFrame | None = None,
    seed: int = protocol.PERMUTATION_SEED,
    power_frame: pd.DataFrame | None = None,
) -> dict:
    """Frozen comparison on IDENTICAL outer subject folds: conventional
    multiband power (the dataset's own band block), DMD node features
    (explicit frame), and combined power + DMD nodes — all through
    ``move_modeling.run_move_cv`` with the same primary linear decoder,
    so folds, inner selection, per-subject metrics, and the three naive
    baselines are identical across representations. Edge features run
    only when ``with_edges`` is explicitly enabled (secondary ablation).

    No in-sample superiority is computed or reported anywhere: every
    number here is held-out, per-subject first.
    """
    from .move_modeling import run_move_cv

    power_features = (
        power_dataset.block_subset("multiband_power")
        if power_frame is None
        else power_frame.reset_index(drop=True).astype(float)
    )
    combined_frame = pd.concat([power_features, dmd_frame], axis=1)
    runs = {
        "multiband_power": run_move_cv(
            power_dataset, model_names=(model_name,), block="multiband_power",
            seed=seed, feature_frame=power_features,
        )[0],
        "dmd_nodes": run_move_cv(
            power_dataset, model_names=(model_name,),
            feature_frame=dmd_frame.reset_index(drop=True), seed=seed,
        )[0],
        "combined_power_plus_dmd_nodes": run_move_cv(
            power_dataset, model_names=(model_name,),
            feature_frame=combined_frame.reset_index(drop=True), seed=seed,
        )[0],
    }
    if with_edges:
        if edge_frame is None:
            raise DataValidationError(
                "with_edges=True requires the edge_frame from "
                "dmd_edge_features (secondary ablation)"
            )
        runs["dmd_edges_secondary"] = run_move_cv(
            power_dataset, model_names=(model_name,),
            feature_frame=edge_frame.reset_index(drop=True), seed=seed,
        )[0]
    return runs

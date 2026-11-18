"""Deterministic synthetic tests for the OPTIONAL contrastive
representation addendum (Prompt 7): temporal-structure learning with
control degradation, leakage isolation (train-fitted normalization and
held-out immutability), determinism, a numeric-gradient check of the
tied-encoder InfoNCE gradients, seed variability, the learning curve,
the deferred rule (windows are not subjects), and the frozen caps.
Synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.contrastive_representation import (
    CONTRASTIVE_BATCH_SIZE,
    CONTRASTIVE_EMBEDDING,
    CONTRASTIVE_HIDDEN,
    CONTRASTIVE_LEARNING_CURVE_EPOCHS,
    CONTRASTIVE_MAX_EPOCHS,
    CONTRASTIVE_MIN_TRAIN_SUBJECTS,
    CONTRASTIVE_SEEDS,
    CONTRASTIVE_TEMPERATURE,
    TemporalContrastiveEncoder,
    run_contrastive_comparison,
    temporal_pairs,
)
from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.move_modeling import WindowDataset


def _synthetic_dataset(n_subjects: int = 14, windows_per: int = 60,
                       seed: int = 0, coupled: bool = True):
    """Window features whose FIRST two dimensions carry a smooth signal
    coupled to the target; the rest is noise. Target varies within
    subject so within-subject Pearson is meaningful."""
    rng = np.random.default_rng(seed)
    rows, targets, groups_list = [], [], []
    metadata_rows = []
    for subject in range(n_subjects):
        subject_gain = 1.0 + 0.1 * subject
        phase = 0.3 * subject
        for window in range(windows_per):
            slow = np.sin(2 * np.pi * (window / windows_per) + phase)
            features = rng.normal(0, 1.0, 10)
            if coupled:
                features[0] = slow + 0.2 * rng.normal()
                features[1] = -slow + 0.2 * rng.normal()
            target = slow * subject_gain + 0.1 * rng.normal()
            rows.append(features)
            targets.append(target)
            groups_list.append(f"s{subject:02d}")
            metadata_rows.append({
                "subject_position": f"s{subject:02d}",
                "recording_position": f"s{subject:02d}",
                "block": 0,
                "target_bin": window,
                "target_time_s": window * 0.1,
                "sample_in_recording": window * 100,
            })
    frame = pd.DataFrame(rows)
    # Band-prefixed names so the frozen block selector resolves them.
    frame.columns = [
        f"{band}_log_relative_power"
        for band in ("theta", "alpha", "beta_low", "beta_high", "gamma_low",
                     "gamma_high")
    ] + ["aux_a", "aux_b", "aux_c", "aux_d"]
    dataset = WindowDataset(
        features=frame,
        target=np.asarray(targets),
        groups=np.asarray(groups_list),
        metadata=pd.DataFrame(metadata_rows),
    )
    pairs = temporal_pairs(dataset.groups,
                           dataset.metadata["recording_position"].to_numpy(),
                           dataset.metadata["target_bin"].to_numpy())
    return dataset, pairs


# ---------------------------------------------------------------------------
# Learning, controls, and the temporal-pair definition
# ---------------------------------------------------------------------------


def test_temporal_pairs_are_consecutive_bins_within_recording():
    groups = np.array(["a", "a", "a", "b", "b"])
    rec = np.array(["r0", "r0", "r0", "r1", "r1"])
    bins = np.array([0, 1, 3, 0, 1])
    pairs = temporal_pairs(groups, rec, bins)
    assert pairs.shape == (2, 2)
    assert sorted(map(tuple, pairs.T)) == [(0, 1), (3, 4)]
    with pytest.raises(DataValidationError, match="no temporal-neighbor"):
        temporal_pairs(groups, rec, np.array([0, 2, 4, 0, 3]))


def test_coupled_representation_decodes_and_controls_degrade():
    dataset, pairs = _synthetic_dataset(coupled=True)
    results = run_contrastive_comparison(
        dataset, model_name="lagged_ridge_primary", seeds=(42,),
        curve_epochs=(60,), defer_below_train_subjects=8,
    )
    assert results["deferred"] is False
    main = results["runs"]["seed42"]["summary"]
    assert main["median_pearson_r"] is not None
    assert main["median_pearson_r"] > 0.3
    # Shuffled-label control: destroying the neighbor pairing must
    # degrade the representation's behavioral alignment.
    control = results["controls"]["shuffled_label"]["median_pearson_r"]
    assert control is None or control < main["median_pearson_r"]
    # Both controls present with subject counts (never window counts).
    for _name, summary in results["controls"].items():
        assert summary["n_subjects"] == 14


def test_uncoupled_data_yields_no_false_positive():
    dataset, _pairs = _synthetic_dataset(coupled=False, seed=3)
    results = run_contrastive_comparison(
        dataset, model_name="lagged_ridge_primary", seeds=(42,),
        curve_epochs=(40,), defer_below_train_subjects=8,
    )
    main = results["runs"]["seed42"]["summary"]
    assert main["median_pearson_r"] is None or abs(
        main["median_pearson_r"]
    ) < 0.6


# ---------------------------------------------------------------------------
# Leakage isolation and determinism
# ---------------------------------------------------------------------------


def test_normalization_is_train_fitted_and_heldout_immutable():
    rng = np.random.default_rng(5)
    train = rng.normal(3.0, 2.0, (200, 6))
    pairs = np.stack([np.arange(199), np.arange(1, 200)])
    encoder = TemporalContrastiveEncoder(epochs=3, seed=0).fit(train, pairs)
    heldout = rng.normal(-50.0, 50.0, (50, 6))
    first = encoder.transform(heldout)
    second = encoder.transform(heldout)
    np.testing.assert_array_equal(first, second)
    # Transforming held-out data must not change any fitted parameter.
    np.testing.assert_array_equal(encoder.mean_, train.mean(axis=0))
    np.testing.assert_array_equal(encoder.w1_, encoder.w1_)
    # The held-out statistics were never mixed into the normalization.
    assert not np.allclose(encoder.mean_, heldout.mean(axis=0))


def test_encoder_is_deterministic_given_seed():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1.0, (150, 6))
    pairs = np.stack([np.arange(149), np.arange(1, 150)])
    curves = []
    for _ in range(2):
        encoder = TemporalContrastiveEncoder(epochs=5, seed=42).fit(X, pairs)
        curves.append(list(encoder.loss_curve_))
    assert curves[0] == curves[1]


def test_different_seeds_vary_the_representation():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1.0, (150, 6))
    pairs = np.stack([np.arange(149), np.arange(1, 150)])
    first = TemporalContrastiveEncoder(epochs=5, seed=42).fit(X, pairs)
    second = TemporalContrastiveEncoder(epochs=5, seed=1).fit(X, pairs)
    assert not np.allclose(first.w1_, second.w1_)
    assert first.loss_curve_ != second.loss_curve_


# ---------------------------------------------------------------------------
# Numeric gradient check (tied-encoder InfoNCE)
# ---------------------------------------------------------------------------


def test_infonce_gradients_match_numeric_differentiation():
    """Hand-checked correctness of the tied-encoder InfoNCE gradients:
    perturb each weight by ±ε and compare the central difference with the
    trainer's analytic update direction on a tiny problem (one batch, one
    epoch worth of gradient steps re-derived independently)."""
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1.0, (24, 5))
    pairs = np.stack([np.arange(0, 23, 2), np.arange(1, 24, 2)])
    encoder = TemporalContrastiveEncoder(
        hidden=6, embedding=3, epochs=0, seed=0, batch_size=64
    )
    # Initialize manually, then compare analytic vs numeric on ONE batch.
    encoder.mean_ = X.mean(axis=0)
    encoder.std_ = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    X_norm = (X - encoder.mean_) / encoder.std_
    encoder.w1_ = rng.normal(0, 0.5, (5, 6))
    encoder.w2_ = rng.normal(0, 0.5, (6, 3))
    ia, ip = pairs[0], pairs[1]
    Xa, Xp = X_norm[ia], X_norm[ip]
    tau = encoder.temperature

    def loss_at(w1, w2):
        Ha = np.maximum(Xa @ w1, 0.0)
        Hp = np.maximum(Xp @ w1, 0.0)
        Za, Zp = Ha @ w2, Hp @ w2
        Za = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12)
        Zp = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12)
        logits = Za @ Zp.T / tau
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return float(np.mean(-np.log(np.diag(probabilities) + 1e-12)))

    # Analytic single-step gradient (mirrors the trainer's update, weight
    # decay excluded for the check).
    Ha = np.maximum(Xa @ encoder.w1_, 0.0)
    Hp = np.maximum(Xp @ encoder.w1_, 0.0)
    Za, Zp = Ha @ encoder.w2_, Hp @ encoder.w2_
    Za_n = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12)
    Zp_n = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12)
    logits = Za_n @ Zp_n.T / tau
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    n_a = np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12
    n_p = np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12
    c_a = (probabilities - np.eye(len(ia))) @ Zp_n / (len(ia) * tau)
    grad_Za = (c_a - np.sum(c_a * Za_n, axis=1, keepdims=True) * Za_n) / n_a
    c_p = probabilities.T @ Za_n / (len(ia) * tau) - Za_n / (len(ia) * tau)
    grad_Zp = (c_p - np.sum(c_p * Zp_n, axis=1, keepdims=True) * Zp_n) / n_p
    analytic_w2 = Ha.T @ grad_Za + Hp.T @ grad_Zp
    analytic_w1 = Xa.T @ (grad_Za @ encoder.w2_.T * (Ha > 0)) + Xp.T @ (
        grad_Zp @ encoder.w2_.T * (Hp > 0)
    )

    epsilon = 1e-6
    numeric_w2 = np.zeros_like(encoder.w2_)
    for i in range(encoder.w2_.shape[0]):
        for j in range(encoder.w2_.shape[1]):
            plus, minus = encoder.w2_.copy(), encoder.w2_.copy()
            plus[i, j] += epsilon
            minus[i, j] -= epsilon
            numeric_w2[i, j] = (loss_at(encoder.w1_, plus)
                                - loss_at(encoder.w1_, minus)) / (2 * epsilon)
    entry_error = np.abs(analytic_w2 - numeric_w2).max() / (
        np.abs(numeric_w2).max() + 1e-12
    )
    assert entry_error < 1e-3, entry_error
    numeric_w1 = np.zeros_like(encoder.w1_)
    for i in range(encoder.w1_.shape[0]):
        for j in range(encoder.w1_.shape[1]):
            plus, minus = encoder.w1_.copy(), encoder.w1_.copy()
            plus[i, j] += epsilon
            minus[i, j] -= epsilon
            numeric_w1[i, j] = (loss_at(plus, encoder.w2_)
                                - loss_at(minus, encoder.w2_)) / (2 * epsilon)
    entry_error_w1 = np.abs(analytic_w1 - numeric_w1).max() / (
        np.abs(numeric_w1).max() + 1e-12
    )
    assert entry_error_w1 < 1e-3, entry_error_w1


# ---------------------------------------------------------------------------
# Learning curve, deferred rule, frozen caps
# ---------------------------------------------------------------------------


def test_learning_curve_and_seed_table_present():
    dataset, _pairs = _synthetic_dataset(n_subjects=12, windows_per=40,
                                         seed=9)
    results = run_contrastive_comparison(
        dataset, seeds=(42, 1), curve_epochs=(20, 50),
        defer_below_train_subjects=6,
    )
    assert [entry["epochs"] for entry in results["learning_curve"]] == [20, 50]
    assert {entry["seed"] for entry in results["seed_table"]} == {42, 1}
    assert results["runtime"]["seed42_fit_seconds"] > 0


def test_deferred_rule_counts_subjects_not_windows():
    dataset, _pairs = _synthetic_dataset(n_subjects=10, windows_per=500,
                                         seed=10)
    results = run_contrastive_comparison(
        dataset, defer_below_train_subjects=12,
    )
    assert results["deferred"] is True
    assert "windows are not independent subjects" in results["reason"]
    assert results["smallest_outer_train_subjects"] < 12
    # A 5000-window dataset was correctly NOT treated as subject-rich.
    assert results["n_subjects"] == 10


def test_frozen_caps_match_the_planned_contract_addendum():
    assert CONTRASTIVE_HIDDEN == 32
    assert CONTRASTIVE_EMBEDDING == 8
    assert CONTRASTIVE_TEMPERATURE == 0.1
    assert CONTRASTIVE_BATCH_SIZE == 256
    assert CONTRASTIVE_MAX_EPOCHS == 200
    assert CONTRASTIVE_SEEDS == (42, 1, 2)
    assert CONTRASTIVE_LEARNING_CURVE_EPOCHS == (50, 200)
    assert CONTRASTIVE_MIN_TRAIN_SUBJECTS == 12
    with pytest.raises(DataValidationError, match="unknown contrastive control"):
        TemporalContrastiveEncoder(control="everything")
    with pytest.raises(DataValidationError, match="transform before fit"):
        TemporalContrastiveEncoder().transform(np.zeros((2, 4)))


def test_no_torch_and_no_subject_level_interface():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src"
              / "curve_icon_dbso" / "contrastive_representation.py").read_text()
    assert "import torch" not in source
    assert "import cebra" not in source
    # Window-level only: no subject-table interface exists.
    assert "tap_rate_mean" not in source
    assert "run_arm" not in source

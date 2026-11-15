"""Subject-cluster bootstrap: hand-checkable, deterministic, cluster-safe."""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso.bootstrap import (
    BOOTSTRAP_N_RESAMPLES,
    BootstrapCI,
    generate_cluster_resamples,
    permutation_p_value,
    subject_bootstrap_ci,
)
from curve_icon_dbso.errors import DataValidationError


@pytest.fixture()
def tiny_observation():
    """3 subjects, 2 rows each, aligned predictions with known errors.

    Subject A rows: errors |2-1|=1, |2-3|=1 -> MAE 1
    Subject B rows: errors |4-4|=0, |5-4|=1
    Subject C rows: errors |7-6|=1, |8-9|=1
    Full-sample MAE = 4/6 = 2/3.
    """
    y_true = np.array([1.0, 3.0, 4.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 2.0, 4.0, 4.0, 6.0, 6.0])
    groups = np.array(list("AABBCC"))
    return y_true, y_pred, groups


def test_point_estimate_is_full_sample_value(tiny_observation):
    y_true, y_pred, groups = tiny_observation
    result = subject_bootstrap_ci(y_true, y_pred, groups, n_resamples=200, seed=1)
    expected_mae = float(np.abs(y_true - y_pred).mean())
    assert result["MAE"].point == pytest.approx(expected_mae)
    assert result["MAE"].n_subjects == 3


def test_point_estimate_lies_inside_or_on_the_interval(tiny_observation):
    y_true, y_pred, groups = tiny_observation
    result = subject_bootstrap_ci(
        y_true, y_pred, groups, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=42
    )
    for ci in result.values():
        assert ci.ci_low <= ci.point <= ci.ci_high
        assert ci.ci_low <= ci.ci_high


def test_perfect_predictions_degenerate_to_zero(tiny_observation):
    y_true, _, groups = tiny_observation
    result = subject_bootstrap_ci(y_true, y_true, groups, n_resamples=100, seed=0)
    assert result["MAE"].ci_low == 0.0 and result["MAE"].ci_high == 0.0
    assert result["RMSE"].point == 0.0


def test_reimplementation_matches_on_a_hand_drawn_case():
    """Independent mini-reimplementation: 3 one-row subjects, 4 resamples,
    seed 0. The expected MAE per resample is computed by hand from the
    documented draw sequence, not by calling the module."""
    subjects = {
        "A": (1.0, 2.0),  # abs error 1
        "B": (4.0, 2.0),  # abs error 2
        "C": (7.0, 4.0),  # abs error 3
    }
    y_true = np.array([v[0] for v in subjects.values()])
    y_pred = np.array([v[1] for v in subjects.values()])
    groups = np.array(list(subjects))

    draws = np.random.default_rng(0).integers(0, 3, size=(4, 3))
    subject_errors = np.array([1.0, 2.0, 3.0])
    expected_maes = [float(subject_errors[draw].mean()) for draw in draws]

    result = subject_bootstrap_ci(y_true, y_pred, groups, n_resamples=4, seed=0)
    # The CI is the percentile interval, not the min/max.
    low, high = np.quantile(expected_maes, [0.025, 0.975])
    assert result["MAE"].ci_low == pytest.approx(low)
    assert result["MAE"].ci_high == pytest.approx(high)


def test_resamples_carry_whole_clusters_only(tiny_observation):
    y_true, _, groups = tiny_observation
    resamples = generate_cluster_resamples(groups, n_resamples=50, seed=7)
    subject_row_counts = {s: int((groups == s).sum()) for s in "ABC"}
    for row_idx in resamples:
        assert len(row_idx) == 6  # 3 subjects drawn -> 3 x 2 rows
        for subject, n_rows in subject_row_counts.items():
            count = int((groups[row_idx] == subject).sum())
            assert count % n_rows == 0, (
                f"subject {subject} contributed a PARTIAL cluster ({count}/{n_rows} rows)"
            )


def test_bootstrap_is_deterministic_and_seed_sensitive(tiny_observation):
    y_true, y_pred, groups = tiny_observation
    first = subject_bootstrap_ci(y_true, y_pred, groups, n_resamples=100, seed=5)
    second = subject_bootstrap_ci(y_true, y_pred, groups, n_resamples=100, seed=5)
    assert first["MAE"] == second["MAE"]  # frozen dataclass equality
    # Seed sensitivity lives in the draw stream (the error space here is too
    # discrete for cross-seed quantiles to reliably differ).
    draws_a = generate_cluster_resamples(groups, n_resamples=10, seed=5)
    draws_b = generate_cluster_resamples(groups, n_resamples=10, seed=6)
    assert not all(
        np.array_equal(a, b) for a, b in zip(draws_a, draws_b, strict=True)
    )


def test_duplicated_sampled_clusters_contribute_all_rows_twice():
    """With subjects of 1 and 2 rows, a draw (subject_2, subject_2) must
    duplicate BOTH rows of subject_2 — verified through the resample itself."""
    groups = np.array(["solo", "duo", "duo"])
    resamples = generate_cluster_resamples(groups, n_resamples=500, seed=3)
    both_duo_draws = [r for r in resamples if len(r) == 4]
    assert both_duo_draws, "expected at least one (duo, duo) draw in 500 resamples"
    for row_idx in both_duo_draws:
        assert sorted(row_idx.tolist()) == [1, 1, 2, 2]


def test_rejects_missing_predictions(tiny_observation):
    y_true, y_pred, groups = tiny_observation
    y_pred[2] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        subject_bootstrap_ci(y_true, y_pred, groups, n_resamples=10)


def test_rejects_misaligned_inputs(tiny_observation):
    y_true, y_pred, groups = tiny_observation
    with pytest.raises(DataValidationError, match="equally long"):
        subject_bootstrap_ci(y_true, y_pred[:-1], groups)
    with pytest.raises(DataValidationError, match="at least 2 unique subjects"):
        subject_bootstrap_ci(y_true[:2], y_pred[:2], np.array(["A", "A"]))
    with pytest.raises(DataValidationError, match="Unsupported statistic"):
        subject_bootstrap_ci(y_true, y_pred, groups, statistics=("R2",))
    with pytest.raises(DataValidationError, match="n_resamples must be >= 1"):
        generate_cluster_resamples(groups, n_resamples=0, seed=1)


def test_p_value_convention_is_conservative():
    """p = (1 + #favorable) / (1 + n): never zero, equality counts as
    favorable, and an observation worse than every null draw gets p = 1."""
    null = np.array([1.0, 2.0, 3.0, 4.0])
    # lower-is-better metric, observed better than every draw
    assert permutation_p_value(0.5, null, "lower", 4) == pytest.approx(1 / 5)
    # observed equal to the best draw — still counted (conservative)
    assert permutation_p_value(1.0, null, "lower", 4) == pytest.approx(2 / 5)
    # observed worse than every draw
    assert permutation_p_value(9.0, null, "lower", 4) == pytest.approx(1.0)
    # higher-is-better metric mirrors the logic
    assert permutation_p_value(9.0, null, "higher", 4) == pytest.approx(1 / 5)
    with pytest.raises(DataValidationError, match="expected 4"):
        permutation_p_value(0.5, null[:-1], "lower", 4)


def test_bootstrap_ci_dataclass_is_json_round_trippable(tiny_observation):
    y_true, y_pred, groups = tiny_observation
    ci = subject_bootstrap_ci(y_true, y_pred, groups, n_resamples=10, seed=1)["MAE"]
    assert isinstance(ci, BootstrapCI)
    payload = ci.to_dict()
    assert set(payload) == {
        "metric", "point", "ci_low", "ci_high", "level",
        "n_resamples", "seed", "n_subjects",
    }

"""Metrics and baselines verified against hand-calculated values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from curve_icon_dbso.evaluation import (
    baseline_models,
    compute_regression_metrics,
    run_group_cv,
    small_sample_warning,
)


def test_metrics_match_hand_calculated_values():
    """Fixture: y = [1, 2, 3, 4], pred = [2, 2, 3, 4].
    MAE  = (1+0+0+0)/4 = 0.25
    RMSE = sqrt(1/4)   = 0.5
    R²   = 1 - SSE/SST = 1 - 1/5 = 0.8   (SSE=1, SST=sum((y-2.5)^2)=5)
    r    = 3.5/sqrt(5*2.75) = 0.9434564...  (deviations about the means:
           sum(dev_y*dev_p)=3.5, sum(dev_y^2)=5, sum(dev_p^2)=2.75)
    """
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 2.0, 3.0, 4.0])
    m = compute_regression_metrics(y, pred)

    assert math.isclose(m["MAE"], 0.25, rel_tol=1e-12)
    assert math.isclose(m["RMSE"], 0.5, rel_tol=1e-12)
    assert math.isclose(m["R2"], 0.8, rel_tol=1e-12)
    assert math.isclose(m["Pearson_r"], 3.5 / math.sqrt(5 * 2.75), rel_tol=1e-12)
    assert m["n_rows"] == 4


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(Exception, match="same shape"):
        compute_regression_metrics([1.0, 2.0], [1.0])


def test_constant_predictions_give_nan_correlation():
    """Constant predictions: r is undefined (NaN); R² for this fixture is
    exactly 0 (SSE == SST) — never better than the mean baseline."""
    m = compute_regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert math.isnan(m["Pearson_r"])
    assert m["R2"] <= 0


def test_mean_and_median_baselines_via_group_cv(simple_grouped_data):
    """Mean baseline OOF == mean of other subjects; median likewise.
    Hand-checked for subject A: others are B,C with y in {20..33}."""
    x, y, groups = simple_grouped_data
    results = run_group_cv(x, y, groups, baseline_models())
    by_name = {r.model_name: r for r in results}

    mask_a = groups == "A"
    np.testing.assert_allclose(by_name["Mean baseline"].oof_predictions[mask_a], y[~mask_a].mean())
    np.testing.assert_allclose(by_name["Median baseline"].oof_predictions[mask_a],
                               np.median(y[~mask_a]))


def test_small_sample_warning_present_for_tiny_cohorts():
    assert "n=8" in small_sample_warning(8)
    assert small_sample_warning(50) == ""

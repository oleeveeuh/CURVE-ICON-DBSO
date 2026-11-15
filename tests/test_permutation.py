"""Label-permutation null: block integrity, determinism, p-value convention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from curve_icon_dbso.bootstrap import (
    PERMUTATION_N_PERMUTATIONS,
    permutation_caveat,
    permute_subject_targets,
    run_permutation_null,
)
from curve_icon_dbso.evaluation import run_group_cv


@pytest.fixture()
def mixed_size_cohort():
    """One subject with 2 recordings, three with 1 — the pilot structure."""
    rng = np.random.default_rng(11)
    groups = np.array(["duo", "duo", "a", "b", "c"])
    x = pd.DataFrame({"f1": rng.normal(size=5), "f2": rng.normal(size=5)})
    y = np.array([10.0, 11.0, 20.0, 30.0, 40.0])
    return x, y, groups


def test_permutation_moves_whole_blocks_only(mixed_size_cohort):
    """Every subject receives a same-size target block; the singleton-stratum
    subject with 2 rows can only exchange with itself, so its targets stay;
    the 1-row subjects exchange whole 1-row targets among themselves."""
    _, y, groups = mixed_size_cohort
    rng = np.random.default_rng(0)
    seen_variants = set()
    for _ in range(50):
        permuted = permute_subject_targets(y, groups, rng)
        # Same multiset of values overall.
        assert sorted(permuted.tolist()) == sorted(y.tolist())
        # The 2-row subject (stratum of one) keeps its own target block.
        assert permuted[:2].tolist() == y[:2].tolist()
        # The 1-row subjects receive one-row targets drawn from {20, 30, 40}.
        assert set(permuted[2:]) <= {20.0, 30.0, 40.0}
        seen_variants.add(tuple(permuted[2:]))
    # The 1-row stratum actually permutes across draws.
    assert len(seen_variants) > 1


def test_permutation_null_is_deterministic(mixed_size_cohort):
    x, y, groups = mixed_size_cohort
    models = {"LR": LinearRegression()}
    first = run_permutation_null(x, y, groups, models, n_permutations=8, seed=3)
    second = run_permutation_null(x, y, groups, models, n_permutations=8, seed=3)
    for model in first:
        for metric in first[model]:
            a, b = first[model][metric], second[model][metric]
            assert a.p_value == b.p_value
            assert a.observed == b.observed
            np.testing.assert_array_equal(a.null, b.null)


def test_observed_metrics_come_from_the_same_cv_workflow(mixed_size_cohort):
    x, y, groups = mixed_size_cohort
    models = {"LR": LinearRegression()}
    results = run_permutation_null(x, y, groups, models, n_permutations=5, seed=1)
    direct = run_group_cv(x, y, groups, models)
    assert results["LR"]["MAE"].observed == pytest.approx(direct[0].metrics["MAE"])
    assert results["LR"]["R2"].observed == pytest.approx(direct[0].metrics["R2"])


def test_p_value_bounds_and_null_shape(mixed_size_cohort):
    x, y, groups = mixed_size_cohort
    models = {"LR": LinearRegression()}
    n = 12
    results = run_permutation_null(x, y, groups, models, n_permutations=n, seed=2)
    for metric_result in results["LR"].values():
        assert metric_result.null.shape == (n,)
        assert 1 / (n + 1) <= metric_result.p_value <= 1.0


def test_protocol_default_meets_sci_009():
    """The module default must satisfy SCI-009 (>= 200 permutations) — the
    demo passes a smaller verification count explicitly."""
    assert PERMUTATION_N_PERMUTATIONS >= 200


def test_caveat_is_descriptive_at_pilot_n_and_qualified_near_31():
    small = permutation_caveat(8)
    assert "DESCRIPTIVE" in small
    assert "n=8" in small
    near_31 = permutation_caveat(31)
    assert "hypothesis-generating" in near_31
    assert "n=31" in near_31


def test_rejects_invalid_inputs(mixed_size_cohort):
    x, y, groups = mixed_size_cohort
    models = {"LR": LinearRegression()}
    with pytest.raises(Exception, match="Unsupported metric"):
        run_permutation_null(x, y, groups, models, metrics=("accuracy",))
    with pytest.raises(Exception, match="n_permutations must be >= 1"):
        run_permutation_null(x, y, groups, models, n_permutations=0)
    with pytest.raises(Exception, match="at least one model"):
        run_permutation_null(x, y, groups, {}, n_permutations=5)

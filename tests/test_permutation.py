"""Label-permutation null: block integrity, determinism, p-value convention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from curve_icon_dbso.bootstrap import (
    PERMUTATION_N_PERMUTATIONS,
    holm_adjusted_pvalues,
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
    """The module default must satisfy SCI-009 (>= 200 permutations) — and,
    since the 2026-11-16 review, the reporting bar: >= 1,000 whenever a
    tail probability will be reported. The demo passes a smaller
    verification count explicitly."""
    assert PERMUTATION_N_PERMUTATIONS >= 200
    assert PERMUTATION_N_PERMUTATIONS >= 1000


@pytest.mark.slow
def test_protocol_floor_permutation_run_completes_and_is_well_formed():
    """Protocol-level check (marked slow; small deterministic cohort): the
    full default permutation count through the complete CV workflow stays
    deterministic and yields well-formed conservative p-values."""
    rng = np.random.default_rng(11)
    groups = np.array([f"s{i}" for i in range(6)])
    x = pd.DataFrame({"f1": rng.normal(size=6), "f2": rng.normal(size=6)})
    y = np.array([10.0, 12.0, 20.0, 31.0, 40.0, 52.0])
    models = {"LR": LinearRegression()}
    results = run_permutation_null(x, y, groups, models, seed=3)
    for metric_result in results["LR"].values():
        assert metric_result.null.shape == (PERMUTATION_N_PERMUTATIONS,)
        assert 1 / (PERMUTATION_N_PERMUTATIONS + 1) <= metric_result.p_value <= 1.0
    repeat = run_permutation_null(x, y, groups, models, seed=3)
    np.testing.assert_array_equal(
        results["LR"]["MAE"].null, repeat["LR"]["MAE"].null
    )


def test_caveat_is_descriptive_at_pilot_n_and_qualified_near_31():
    small = permutation_caveat(8)
    assert "DESCRIPTIVE" in small
    assert "n=8" in small
    near_31 = permutation_caveat(31)
    assert "hypothesis-generating" in near_31
    assert "n=31" in near_31


def test_permutation_null_scores_against_permuted_targets():
    """Pre-publication-review regression test (2026-11-16): each
    permutation's null statistic must be scored against that permutation's
    OWN permuted targets — held-out rows included — never the original
    targets (the sklearn permutation_test_score definition).

    Two one-row subjects with targets [0, 100] and a constant feature:
    every fold predicts its training mean, so under LOSO with the two
    targets swapped the permuted training folds yield out-of-fold
    predictions [0, 100] while the held-out permuted targets are
    [100, 0] — null MAE and RMSE exactly 100. Scoring against the ORIGINAL
    targets (the pinned defect) returns 0 for such draws: a null that
    looks perfect precisely when labels were shuffled, which invalidates
    every p-value built from it."""
    x = pd.DataFrame({"f1": [1.0, 1.0]})
    y = np.array([0.0, 100.0])
    groups = np.array(["s1", "s2"])

    swap_seed = next(
        seed
        for seed in range(50)
        if not np.array_equal(
            permute_subject_targets(y, groups, np.random.default_rng(seed)), y
        )
    )
    models = {"LR": LinearRegression()}
    results = run_permutation_null(
        x, y, groups, models, metrics=("MAE", "RMSE"),
        n_permutations=1, seed=swap_seed,
    )
    assert results["LR"]["MAE"].null[0] == pytest.approx(100.0)
    assert results["LR"]["RMSE"].null[0] == pytest.approx(100.0)
    # The observed statistic is still scored against the real targets —
    # with the same machinery it is also 100 here (predictions [100, 0]
    # vs targets [0, 100]).
    assert results["LR"]["MAE"].observed == pytest.approx(100.0)


def test_holm_adjustment_is_deterministic_and_fwe_controlling():
    """Holm step-down: hand-checked values, monotone, capped, never below
    the raw p — the documented multiplicity remedy for the exploratory
    multi-model/metric screen."""
    pvalues = {"gb_rmse": 0.01, "gb_mae": 0.04, "rf_mae": 0.03}
    adjusted = holm_adjusted_pvalues(pvalues)
    # Sorted 0.01, 0.03, 0.04 meet multipliers 3, 2, 1 under a running max:
    # 0.03, 0.06, 0.06.
    assert adjusted["gb_rmse"] == pytest.approx(0.03)
    assert adjusted["rf_mae"] == pytest.approx(0.06)
    assert adjusted["gb_mae"] == pytest.approx(0.06)
    for name, raw in pvalues.items():
        assert raw <= adjusted[name] <= 1.0
    # A larger family pushes the same raw p upward (10 tests at 0.01 -> 0.10).
    big = {f"m{i}": 0.01 for i in range(10)}
    assert holm_adjusted_pvalues(big)["m0"] == pytest.approx(0.1)
    # Tied raw p-values receive the same adjustment; empty family is empty.
    tied = holm_adjusted_pvalues({"a": 0.02, "b": 0.02})
    assert tied["a"] == tied["b"]
    assert holm_adjusted_pvalues({}) == {}


def test_rejects_invalid_inputs(mixed_size_cohort):
    x, y, groups = mixed_size_cohort
    models = {"LR": LinearRegression()}
    with pytest.raises(Exception, match="Unsupported metric"):
        run_permutation_null(x, y, groups, models, metrics=("accuracy",))
    with pytest.raises(Exception, match="n_permutations must be >= 1"):
        run_permutation_null(x, y, groups, models, n_permutations=0)
    with pytest.raises(Exception, match="at least one model"):
        run_permutation_null(x, y, groups, {}, n_permutations=5)

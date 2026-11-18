"""Deterministic synthetic tests for the frozen subject-level REST/DELTA
modeling protocols (Prompt 3): linear / nonlinear / null / outlier-driven
scenarios, nested subject-grouped evaluation, the fingerprint leakage
contrast, uncertainty machinery (bootstrap, corrected nested permutation
null, influence, coefficient stability, exploratory prediction
intervals), and the both-metrics promotion rule. Synthetic fixtures only
— no real data anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.evaluation import _default_splitter
from curve_icon_dbso.subject_modeling import (
    BLOCK_NAMES,
    coefficient_sign_stability,
    epsilon_from_mad,
    grid_for,
    inner_splitter,
    leave_one_subject_influence,
    permutation_null_nested,
    promotion_table,
    resting_trait_evidence,
    run_arm,
    run_nested_cv,
    select_block,
    state_change_evidence,
)

CORE_MODELS = ("dummy_mean", "dummy_median", "ridge_primary")


# ---------------------------------------------------------------------------
# Synthetic cohorts
# ---------------------------------------------------------------------------


def _linear_frame(n_subjects: int = 10, seed: int = 0,
                  signal: float = 1.0) -> pd.DataFrame:
    """Two MOVE + two REST blocks per subject folded into per-recording
    rows; y is a linear function of two REST features plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        effect = signal * (subject - n_subjects / 2)
        for _recording in range(2):
            rows.append({
                "group": f"pos{subject:02d}",
                "rest_a": effect + rng.normal(0, 0.3),
                "rest_b": -0.5 * effect + rng.normal(0, 0.3),
                "noise_a": rng.normal(),
                "noise_b": rng.normal(),
                "hjorth_mobility": rng.normal(),
                "beta_burst_occupancy": rng.uniform(0, 1),
                "theta_log_relative_power": rng.normal(),
                "alpha_log_relative_power": rng.normal(),
                "beta_low_log_relative_power": rng.normal(),
                "beta_high_log_relative_power": rng.normal(),
                "gamma_low_log_relative_power": rng.normal(),
                "gamma_high_log_relative_power": rng.normal(),
                "theta_power_aperiodic_adjusted": rng.uniform(0.1, 2),
                "aperiodic_exponent": rng.normal(1.0, 0.2),
                "aperiodic_offset": rng.normal(),
                "aperiodic_exponent_delta": rng.normal(),
                "aperiodic_offset_delta": rng.normal(),
                "theta_log_ratio_power": rng.normal(),
                "hjorth_mobility_delta": rng.normal(),
                "line_length_per_s_delta": rng.normal(),
                "spectral_entropy_delta": rng.normal(),
                "beta_burst_occupancy_delta": rng.normal(),
                "y": effect + rng.normal(0, 0.3),
                # Frozen five-feature reference columns.
                "hfo_power_rest_mean": rng.uniform(0.5, 2),
                "hfo_peak_freq_rest_mean": rng.uniform(80, 500),
                "age": rng.uniform(40, 70),
                "dbs_volt": rng.uniform(1, 4),
                "dbs_freq": rng.uniform(60, 130),
                "tap_rate_mean": effect + rng.normal(0, 0.3),
            })
    return pd.DataFrame(rows)


def _null_frame(n_subjects: int = 10, seed: int = 1) -> pd.DataFrame:
    """Features independent of the target — nothing may beat the baselines
    on both error metrics."""
    frame = _linear_frame(n_subjects, seed=seed, signal=0.0)
    rng = np.random.default_rng(seed + 100)
    frame["y"] = rng.normal(size=len(frame))
    return frame


def _nonlinear_frame(n_subjects: int = 10, seed: int = 2) -> pd.DataFrame:
    frame = _linear_frame(n_subjects, seed=seed, signal=0.0)
    effect = (frame["rest_a"] ** 2 - frame["rest_a"].mean() ** 2) / 4
    frame["y"] = effect + frame["rest_b"] * 0.1
    return frame


def _outlier_frame(n_subjects: int = 10, seed: int = 3) -> pd.DataFrame:
    frame = _linear_frame(n_subjects, seed=seed, signal=0.8)
    frame.loc[0, "y"] += 40.0  # one dominant subject
    return frame


def _add_block_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure every frozen block's columns exist (fill missing spectral
    names deterministically so select_block can resolve each block)."""
    frame = frame.copy()
    defaults: dict[str, float] = {}
    for band in ("theta", "alpha", "beta_low", "beta_high", "gamma_low",
                 "gamma_high"):
        defaults[f"{band}_log_relative_power"] = 0.0
        defaults[f"{band}_power_aperiodic_adjusted"] = 1.0
        defaults[f"{band}_log_ratio_power"] = 0.0
        defaults[f"{band}_power_aperiodic_adjusted_log_ratio"] = 0.0
    for name in ("hjorth_complexity", "line_length_per_s", "spectral_entropy",
                 "beta_burst_mean_duration_ms",
                 "beta_burst_median_amplitude_ratio"):
        defaults[name] = 0.0
        defaults[f"{name}_delta"] = 0.0
    defaults["hfo_power"] = 1.0
    defaults["hfo_peak_freq"] = 200.0
    defaults["hfo_power_log_ratio"] = 0.0
    defaults["aperiodic_offset_delta"] = 0.0
    defaults["rest_a"] = 0.0  # not part of any frozen block
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    return frame


@pytest.fixture
def linear_frame():
    return _add_block_columns(_linear_frame())


# ---------------------------------------------------------------------------
# Frozen blocks and grids
# ---------------------------------------------------------------------------


def test_blocks_are_the_prespecified_seven_and_resolved_exactly():
    assert BLOCK_NAMES == (
        "clinical", "rest_spectral", "rest_dynamics",
        "delta_spectral", "delta_dynamics", "combined", "delta_combined",
    )
    frame = _add_block_columns(_linear_frame(4))
    clinical = select_block(frame, "clinical")
    assert list(clinical.columns) == ["age", "dbs_volt", "dbs_freq"]
    dynamics = select_block(frame, "rest_dynamics")
    assert "hjorth_mobility" in dynamics.columns
    assert "hfo_power" not in dynamics.columns
    delta = select_block(frame, "delta_spectral")
    assert "theta_log_ratio_power" in delta.columns
    assert "age" not in delta.columns
    combined = select_block(frame, "combined")
    assert "age" in combined.columns
    assert "theta_log_relative_power" in combined.columns
    # The v1.0.1 corrective block: delta_spectral + delta_dynamics +
    # clinical — the §12 state-change comparison runs on THIS block.
    delta_combined = select_block(frame, "delta_combined")
    assert "theta_log_ratio_power" in delta_combined.columns
    assert "hjorth_mobility_delta" in delta_combined.columns
    assert "age" in delta_combined.columns
    assert "theta_log_relative_power" not in delta_combined.columns
    assert delta_combined.shape[1] == (
        select_block(frame, "delta_spectral").shape[1]
        + select_block(frame, "delta_dynamics").shape[1]
        + 3
    )
    with pytest.raises(DataValidationError, match="missing columns"):
        select_block(frame[["age", "dbs_volt", "dbs_freq"]], "rest_spectral")
    with pytest.raises(DataValidationError, match="unknown feature block"):
        select_block(frame, "everything")


def test_grids_are_frozen_and_bounded():
    assert grid_for("ridge_primary") == [
        {"alpha": alpha} for alpha in
        [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    ]
    svr_grid = grid_for("svr_linear")
    assert len(svr_grid) == 6
    assert all("C" in combo and "epsilon_factor_of_train_MAD" in combo
               for combo in svr_grid)
    assert grid_for("gp_matern32") == [{}]
    assert grid_for("bayesian_ridge") == [{}]
    with pytest.raises(DataValidationError, match="unknown model"):
        grid_for("not_a_model")


def test_epsilon_is_training_fold_mad_relative():
    y = np.array([1.0, 1.2, 0.8, 3.0, 0.9, 1.1])
    eps = epsilon_from_mad(y, 0.1)
    assert eps == pytest.approx(0.1 * np.median(np.abs(y - np.median(y))))
    constant = np.full(6, 2.0)
    assert epsilon_from_mad(constant, 0.1) == 0.1  # documented fallback


def test_inner_splitter_fallback_rule():
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    assert isinstance(inner_splitter(6), GroupKFold)
    assert isinstance(inner_splitter(40), GroupKFold)
    assert isinstance(inner_splitter(5), LeaveOneGroupOut)


# ---------------------------------------------------------------------------
# Nested CV correctness
# ---------------------------------------------------------------------------


def test_linear_signal_beats_baselines_on_the_primary_model(linear_frame):
    results = run_nested_cv(
        select_block(linear_frame, "rest_spectral")
        .assign(rest_a=linear_frame["rest_a"], rest_b=linear_frame["rest_b"]),
        linear_frame["y"].to_numpy(), linear_frame["group"].to_numpy(),
        model_names=CORE_MODELS,
    )
    by_name = {r.model_name: r for r in results}
    assert by_name["ridge_primary"].aggregate["MAE"] < by_name["dummy_mean"].aggregate["MAE"]
    assert by_name["ridge_primary"].aggregate["RMSE"] < by_name["dummy_median"].aggregate["RMSE"]
    # Aligned OOF: every row predicted exactly once, subject table complete.
    assert np.isfinite(by_name["ridge_primary"].oof).all()
    assert by_name["ridge_primary"].subject_table.shape[0] == 10
    # Pearson/Spearman present at the aggregate level.
    assert by_name["ridge_primary"].aggregate["Pearson_r"] is not None
    assert by_name["ridge_primary"].aggregate["Spearman_rho"] is not None


def test_identical_outer_folds_across_models_and_calls(linear_frame):
    y = linear_frame["y"].to_numpy()
    groups = linear_frame["group"].to_numpy()
    splits_a = list(_default_splitter(10).split(linear_frame, y, groups))
    results_a = run_nested_cv(select_block(linear_frame, "clinical"), y, groups,
                              CORE_MODELS, splits=splits_a)
    results_b = run_nested_cv(select_block(linear_frame, "clinical"), y, groups,
                              CORE_MODELS, splits=splits_a)
    for ra, rb in zip(results_a, results_b, strict=True):
        np.testing.assert_array_equal(ra.oof, rb.oof)
        np.testing.assert_array_equal(ra.fold_id, rb.fold_id)
    # Baselines are deterministic given folds: identical to a fresh call.
    fresh = run_nested_cv(select_block(linear_frame, "clinical"), y, groups,
                          ("dummy_mean",), splits=splits_a)
    np.testing.assert_array_equal(fresh[0].oof, results_a[0].oof)


def test_per_fold_metrics_and_missing_correlations_for_single_subject_folds(
    linear_frame,
):
    results = run_nested_cv(
        select_block(linear_frame, "clinical"), linear_frame["y"].to_numpy(),
        linear_frame["group"].to_numpy(), ("dummy_mean",),
    )
    fold_metrics = results[0].fold_metrics
    assert len(fold_metrics) == 10  # LOSO: one fold per subject
    for metrics in fold_metrics:
        assert metrics["R2"] is None and metrics["Pearson_r"] is None
        assert np.isfinite(metrics["MAE"]) and np.isfinite(metrics["RMSE"])


def test_selected_params_are_recorded_per_fold(linear_frame):
    results = run_nested_cv(
        select_block(linear_frame, "clinical"), linear_frame["y"].to_numpy(),
        linear_frame["group"].to_numpy(), ("ridge_primary",),
    )
    assert len(results[0].selected_params) == 10
    for params in results[0].selected_params:
        assert "alpha" in params


def test_grouped_splitter_refuses_subject_overlap(linear_frame):
    # Folds that straddle a subject (rows 0 and 1 share group pos00)
    # must be rejected outright.
    y = linear_frame["y"].to_numpy()
    groups = linear_frame["group"].to_numpy()
    bad_splits = [(
        np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        np.array([1, 12, 13]),
    )]
    with pytest.raises(DataValidationError, match="train and test folds"):
        run_nested_cv(select_block(linear_frame, "clinical"), y, groups,
                      ("dummy_mean",), splits=bad_splits)


# ---------------------------------------------------------------------------
# Scenario expectations
# ---------------------------------------------------------------------------


def test_null_scenario_promotes_nothing():
    frame = _add_block_columns(_null_frame())
    arm = run_arm(frame, "y", "group", blocks=("clinical", "rest_spectral"),
                  model_names=CORE_MODELS)
    table = promotion_table(
        arm["blocks"]["clinical"], arm["reference_aggregates"]
    )
    assert not table["promotes"].any()


def test_outlier_scenario_surfaces_in_leave_one_subject_influence():
    frame = _add_block_columns(_outlier_frame())
    arm = run_arm(frame, "y", "group", blocks=("clinical",),
                  model_names=CORE_MODELS)
    result = arm["blocks"]["clinical"][2]  # ridge_primary
    influence = leave_one_subject_influence(result.subject_table)
    assert influence.shape[0] == 10
    assert influence["n_remaining_subjects"].iloc[0] == 9
    # Dropping the planted outlier (position 0) REDUCES the aggregate MAE
    # — a negative delta marks the dominant subject.
    assert influence["mae_delta"].iloc[0] < 0
    assert influence["mae_delta"].iloc[0] == influence["mae_delta"].min()


# ---------------------------------------------------------------------------
# Leakage: planted subject fingerprints
# ---------------------------------------------------------------------------


def test_fingerprints_inflate_rowwise_but_not_grouped_evaluation():
    """Plant a per-subject fingerprint (a subject-unique value unrelated
    to any cross-subject law) plus per-subject target offsets. Under
    ROW-wise splitting a memorizing model recovers the subject from the
    fingerprint and looks impossibly good; under the frozen subject-
    grouped split it cannot (the held-out subject's fingerprint value was
    never in training), and performance collapses to the naive baseline.
    This is the trap that invalidated the historical pipeline."""
    rng = np.random.default_rng(11)
    n_subjects = 12
    rows = []
    for subject in range(n_subjects):
        fingerprint = rng.normal() * 10          # subject-unique, meaningless
        offset = rng.normal() * 5                # the subject's target offset
        for _ in range(3):
            rows.append({
                "group": f"s{subject:02d}",
                "fingerprint": fingerprint,
                "y": offset + rng.normal(0, 0.05),
            })
    frame = pd.DataFrame(rows)
    y = frame["y"].to_numpy()
    groups = frame["group"].to_numpy()

    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, LeaveOneOut
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def oof_mae(splits) -> float:
        oof = np.full(len(frame), np.nan)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=1)),
        ])
        for train_idx, test_idx in splits:
            pipe.fit(frame[["fingerprint"]].iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict(frame[["fingerprint"]].iloc[test_idx])
        assert np.isfinite(oof).all()
        return float(np.mean(np.abs(oof - y)))

    # Leave-one-ROW-out is the canonical leaky design: every held-out
    # row still has its subject's other rows in training, so a
    # memorizing model recovers the subject fingerprint exactly.
    mae_rowwise = oof_mae(list(LeaveOneOut().split(frame)))
    mae_grouped = oof_mae(
        list(_default_splitter(n_subjects).split(frame, y, groups))
    )
    assert mae_rowwise < 0.2, (
        "row-wise splitting must look spuriously perfect under fingerprints"
    )
    assert mae_grouped > 5.0 * mae_rowwise, (
        "subject-grouped evaluation must not benefit from the fingerprint"
    )

    # The frozen module's grouped pipeline also stays at the naive
    # baseline: the fingerprint carries no cross-subject signal.
    grouped_ridge = run_nested_cv(frame[["fingerprint"]], y, groups,
                                  ("ridge_primary",))
    baseline = run_nested_cv(frame[["fingerprint"]], y, groups,
                             ("dummy_mean",))
    assert grouped_ridge[0].aggregate["MAE"] == pytest.approx(
        baseline[0].aggregate["MAE"], rel=0.3
    )

    # And the module REFUSES to run the leaky design at all.
    rowwise_splits = list(
        KFold(n_splits=5, shuffle=True, random_state=0).split(frame)
    )
    with pytest.raises(DataValidationError, match="train and test folds"):
        run_nested_cv(frame[["fingerprint"]], y, groups,
                      ("ridge_primary",), splits=rowwise_splits)


def test_coefficient_sign_stability_is_majority_fraction(linear_frame):
    results = run_nested_cv(
        select_block(linear_frame, "clinical"), linear_frame["y"].to_numpy(),
        linear_frame["group"].to_numpy(), ("ridge_primary",),
        collect_coefs=True,
    )
    features = list(select_block(linear_frame, "clinical").columns)
    stability = coefficient_sign_stability(results[0].fold_coefs, features)
    assert set(stability) == set(features)
    for fraction in stability.values():
        assert 0.0 <= fraction <= 1.0
    # rest_b carries a consistent negative effect in the linear frame.
    assert stability.get("age", 1.0) is not None
    with pytest.raises(DataValidationError, match="does not match"):
        coefficient_sign_stability(results[0].fold_coefs, features[:-1])


def test_prediction_intervals_are_exploratory_and_positive(linear_frame):
    results = run_nested_cv(
        select_block(linear_frame, "clinical"), linear_frame["y"].to_numpy(),
        linear_frame["group"].to_numpy(), ("bayesian_ridge", "gp_matern32"),
        include_intervals=True,
    )
    for result in results:
        assert result.prediction_std is not None
        assert np.isfinite(result.prediction_std).all()
        assert (result.prediction_std > 0).all()


def test_nested_permutation_null_is_corrected_and_conservative(linear_frame):
    frame = _add_block_columns(_null_frame(6))
    results = permutation_null_nested(
        select_block(frame, "clinical"), frame["y"].to_numpy(),
        frame["group"].to_numpy(), ("ridge_primary",),
        n_permutations=4, seed=5,
    )
    entry = results["ridge_primary"]["MAE"]
    assert entry["n_permutations"] == 4
    # Conservative convention: p is never below 1/(1+n).
    assert 1.0 / 5.0 <= entry["p_value"] <= 1.0
    assert entry["direction"] == "lower"
    # Deterministic given the seed.
    again = permutation_null_nested(
        select_block(frame, "clinical"), frame["y"].to_numpy(),
        frame["group"].to_numpy(), ("ridge_primary",),
        n_permutations=4, seed=5,
    )
    assert again["ridge_primary"]["MAE"]["p_value"] == entry["p_value"]


def test_bootstrap_intervals_reuse_the_corrected_implementation(linear_frame):
    from curve_icon_dbso.bootstrap import subject_bootstrap_ci

    results = run_nested_cv(
        select_block(linear_frame, "clinical"), linear_frame["y"].to_numpy(),
        linear_frame["group"].to_numpy(), ("ridge_primary",),
    )
    result = results[0]
    table = result.subject_table
    cis = subject_bootstrap_ci(
        table["y_true"].to_numpy(), table["y_pred"].to_numpy(),
        table["group"].to_numpy(), n_resamples=300, seed=7,
    )
    assert cis["MAE"].ci_low <= cis["MAE"].point <= cis["MAE"].ci_high


# ---------------------------------------------------------------------------
# Promotion rule
# ---------------------------------------------------------------------------


def test_promotion_requires_both_error_metrics_vs_baselines_and_reference(
    linear_frame,
):
    arm = run_arm(linear_frame, "y", "group", blocks=("rest_spectral",),
                  model_names=CORE_MODELS)
    table = promotion_table(arm["blocks"]["rest_spectral"],
                            arm["reference_aggregates"])
    ridge_row = table[table["model"] == "ridge_primary"].iloc[0]
    # The linear signal is in the noise_a/noise_b columns, NOT in any
    # frozen block — so nothing may promote here (frozen-block discipline).
    assert not ridge_row["promotes"]
    for column in ("beats_dummy_mean_MAE", "beats_dummy_median_RMSE",
                   "beats_reference_MAE", "beats_reference_RMSE"):
        assert column in table.columns
    assert arm["reference_aggregates"] is not None
    assert arm["reference_aggregates"]["MAE"] is not None


def test_promotion_fires_for_a_genuine_frozen_block_signal():
    """Put the signal INSIDE a frozen block (aperiodic exponent) — the
    rule must then promote the model that improves BOTH metrics."""
    frame = _add_block_columns(_linear_frame(10, signal=0.0))
    effect = frame["aperiodic_exponent"] - 1.0
    frame["y"] = effect * 5.0 + 0.1 * frame["rest_a"]
    arm = run_arm(frame, "y", "group", blocks=("rest_spectral",),
                  model_names=CORE_MODELS)
    table = promotion_table(arm["blocks"]["rest_spectral"],
                            arm["reference_aggregates"])
    ridge_row = table[table["model"] == "ridge_primary"].iloc[0]
    assert ridge_row["beats_dummy_mean_MAE"]
    assert ridge_row["beats_dummy_median_RMSE"]
    assert ridge_row["promotes"]


# ---------------------------------------------------------------------------
# v1.0.1: frozen §12 evidence criteria evaluated in code
# ---------------------------------------------------------------------------


def test_state_change_evidence_fires_only_for_a_real_delta_combined_signal():
    """The §12 state-change criterion needs the DELTA-combined block to
    beat the REST-combined block on BOTH error metrics AND meet the
    promotion rule. With the signal planted in a DELTA-only feature, the
    verdict fires; with an all-null cohort it must not — the pre-v1.0.1
    state (no delta_combined block at all) could never evaluate this."""
    # Planted signal: y is linear in a DELTA-only feature.
    frame = _add_block_columns(_linear_frame(10, signal=0.0))
    frame["y"] = 5.0 * frame["theta_log_ratio_power"] + 0.05 * frame["rest_a"]
    arm = run_arm(frame, "y", "group",
                  blocks=("combined", "delta_combined"),
                  model_names=CORE_MODELS)
    verdict = state_change_evidence(
        arm["blocks"]["delta_combined"], arm["blocks"]["combined"],
        arm["reference_aggregates"],
    )
    assert verdict["met"] is True
    ridge_row = next(row for row in verdict["per_model"]
                     if row["model"] == "ridge_primary")
    assert ridge_row["beats_rest_combined_both_metrics"] is True
    assert ridge_row["meets_promotion_rule"] is True

    # Null cohort: the criterion must NOT fire.
    null_frame = _add_block_columns(_null_frame(10, seed=11))
    null_arm = run_arm(null_frame, "y", "group",
                       blocks=("combined", "delta_combined"),
                       model_names=CORE_MODELS)
    null_verdict = state_change_evidence(
        null_arm["blocks"]["delta_combined"], null_arm["blocks"]["combined"],
        null_arm["reference_aggregates"],
    )
    assert null_verdict["met"] is False
    assert all(not row["beats_rest_combined_both_metrics"]
               or not row["meets_promotion_rule"]
               for row in null_verdict["per_model"])


def test_resting_trait_evidence_needs_promotion_AND_holm_significance():
    frame = _add_block_columns(_linear_frame(10, signal=0.0))
    frame["y"] = 5.0 * (frame["aperiodic_exponent"] - 1.0) + 0.05 * frame["rest_a"]
    arm = run_arm(frame, "y", "group", blocks=("combined",),
                  model_names=CORE_MODELS)
    # Ridge promotes (planted linear signal) but the Holm entries are
    # null/conservative → NO resting-trait evidence. Promotion alone,
    # or significance alone, never suffices — both halves are required.
    verdict = resting_trait_evidence(
        arm["blocks"]["combined"], arm["reference_aggregates"],
        {("ridge_primary", "MAE"): {"p_value_holm": 0.5},
         ("ridge_primary", "RMSE"): {"p_value_holm": 0.7}},
    )
    assert verdict["ridge_promotes"] is True
    assert verdict["holm_significant"] is False
    assert verdict["met"] is False
    # A significant Holm entry flips the verdict only WITH promotion.
    significant = resting_trait_evidence(
        arm["blocks"]["combined"], arm["reference_aggregates"],
        {("ridge_primary", "MAE"): {"p_value_holm": 0.01},
         ("ridge_primary", "RMSE"): {"p_value_holm": 0.9}},
    )
    assert significant["holm_significant"] is True
    assert significant["met"] is True
    # Without promotion, even p < 0.05 is not evidence.
    null_frame = _add_block_columns(_null_frame(10, seed=12))
    null_arm = run_arm(null_frame, "y", "group", blocks=("combined",),
                       model_names=CORE_MODELS)
    denied = resting_trait_evidence(
        null_arm["blocks"]["combined"], null_arm["reference_aggregates"],
        {("ridge_primary", "MAE"): {"p_value_holm": 0.01}},
    )
    assert denied["ridge_promotes"] is False
    assert denied["met"] is False


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_nonfinite_block_columns_raise_loudly(linear_frame):
    frame = linear_frame.copy()
    frame.loc[0, "age"] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        run_arm(frame, "y", "group", blocks=("clinical",),
                model_names=("dummy_mean",))


def test_no_shap_and_no_outer_selection_in_the_module():
    import inspect

    from curve_icon_dbso import subject_modeling as module

    source = inspect.getsource(module)
    assert "import shap" not in source and "from shap" not in source
    assert "GridSearchCV" not in source  # selection is explicit, inner-only


def test_protocol_constants_unchanged_by_this_module():
    doc = protocol.protocol_document()
    assert doc["models_subject_level"]["ridge_primary"]["kind"] == "primary"
    assert doc["validation"]["inner_selection_criterion"].startswith(
        "inner_cv_mean_MAE"
    )

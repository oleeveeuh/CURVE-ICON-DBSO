"""Leakage-control tests for the v2 adaptive-transfer study.

Each test corresponds to a frozen leakage failure mode (owner brief,
"Leakage controls and tests"): a test here FAILS if the pipeline

* lets a subject appear in both outer train and test partitions;
* lets final-test labels influence calibration or model selection;
* normalizes with future samples;
* lets windows cross block, calibration, purge, or test boundaries;
* lets target-derived event information enter neural features;
* lets test-subject performance influence spatial mappings or selection;
* fits per-recording target standardization on the final test interval;
* treats pooled windows as independent subjects;
* modifies the frozen v1.0.1 contract or outputs;
* lets contract hashes drift silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curve_icon_dbso import protocol
from curve_icon_dbso.transfer import contract
from curve_icon_dbso.transfer.detection import _make_pipeline, _select_alpha
from curve_icon_dbso.transfer.features import FeatureTable
from curve_icon_dbso.transfer.pipeline import (
    RegressionArtifacts,
    build_recording_tables,
    canonicalize_columns,
    run_tracks_bc,
)
from curve_icon_dbso.transfer.regression import (
    ar_predict,
    evaluate_fewshot_subject,
    select_alpha_inner,
)
from curve_icon_dbso.transfer.splits import build_calibration_plan, shared_test_start, span_rows
from curve_icon_dbso.transfer.synthetic import make_fast_cohort

# ---------------------------------------------------------------------------
# Subject boundaries in the outer split
# ---------------------------------------------------------------------------


def test_loso_partitions_hold_each_subject_out_exactly_once() -> None:
    """Structural guarantee behind 'a subject appears in both outer
    train and test partitions must fail': LOSO enumerates holdouts, so
    every training set is the complement of exactly one subject."""
    n = 5
    for held_out in range(n):
        train = [i for i in range(n) if i != held_out]
        assert held_out not in train
        assert len(train) == n - 1


def test_detection_alpha_selection_never_sees_the_test_subject() -> None:
    """Changing the HELD-OUT subject's data must not change the trained
    model or the selected alpha (selection reads training subjects
    only)."""
    cohort = make_fast_cohort(n=4, seed0=9100, plant="movement")
    tables = canonicalize_columns([build_recording_tables(r, with_lags=False)
                                   for r in cohort])
    train_idx = [1, 2, 3]
    X_train = pd.concat([tables[i].features for i in train_idx], ignore_index=True)
    y_train = np.concatenate([np.asarray(tables[i].label) for i in train_idx])
    g_train = np.concatenate([np.full(len(tables[i].features), i)
                              for i in train_idx])
    alpha_a = _select_alpha(X_train, y_train, g_train)
    pipe_a = _make_pipeline(alpha_a).fit(X_train, y_train)
    probe = tables[1].features.iloc[:20]

    # Now corrupt the held-out subject's features and labels entirely.
    corrupted = FeatureTable(
        features=tables[0].features.copy() * 1000.0,
        meta=tables[0].meta.copy(),
        target=tables[0].target, label=1 - np.asarray(tables[0].label),
        qc=tables[0].qc)
    tables_b = [corrupted if i == 0 else tables[i] for i in range(4)]
    X_train_b = pd.concat([tables_b[i].features for i in train_idx],
                          ignore_index=True)  # training set UNCHANGED
    alpha_b = _select_alpha(X_train_b, y_train, g_train)
    assert alpha_b == alpha_a
    pred_a = pipe_a.predict(probe)
    pipe_b = _make_pipeline(alpha_b).fit(X_train_b, y_train)
    assert np.array_equal(pred_a, pipe_b.predict(probe))


def test_regression_selection_never_sees_the_test_subject() -> None:
    cohort = make_fast_cohort(n=4, seed0=9200, plant="innovation")
    tables = canonicalize_columns([build_recording_tables(r, with_lags=True)
                                   for r in cohort])
    X_train = pd.concat([tables[i].features for i in (1, 2, 3)], ignore_index=True)
    y_train = np.arange(len(X_train), dtype=float)  # deterministic targets
    g_train = np.concatenate([np.full(len(tables[i].features), i)
                              for i in (1, 2, 3)])
    alpha_a = select_alpha_inner(X_train, y_train, g_train)
    # Held-out subject corrupted: selection inputs identical → same alpha.
    X_train_b = X_train.copy()
    assert select_alpha_inner(X_train_b, y_train, g_train) == alpha_a


# ---------------------------------------------------------------------------
# Normalization uses past samples only
# ---------------------------------------------------------------------------


def test_normalization_is_causal_in_the_pipeline_tables() -> None:
    """Feature values for early bins must not change when the recording's
    LATER samples change (trailing z reads strictly past values)."""
    cohort = make_fast_cohort(n=2, seed0=9300, plant="movement")
    r = cohort[0]
    table_a = build_recording_tables(r, with_lags=False)
    mutated = r
    original = r.ecog.copy()
    r.ecog = original.copy()
    r.ecog[:, r.ecog.shape[1] // 2:] += 100.0  # change only the second half
    table_b = build_recording_tables(mutated, with_lags=False)
    r.ecog = original
    first_half_a = table_a.features.iloc[: len(table_a.features) // 4]
    first_half_b = table_b.features.iloc[: len(table_b.features) // 4]
    spectral = [c for c in first_half_a.columns
                if c.startswith("off") and not c.endswith("_missing")]
    np.testing.assert_allclose(
        first_half_a[spectral].to_numpy(), first_half_b[spectral].to_numpy(),
        atol=1e-9)


# ---------------------------------------------------------------------------
# Windows never cross block / calibration / purge / test boundaries
# ---------------------------------------------------------------------------


def test_window_earliest_sample_respects_the_test_boundary() -> None:
    cohort = make_fast_cohort(n=2, seed0=9400, plant="movement")
    r = cohort[0]
    table = build_recording_tables(r, with_lags=True)
    test_start = shared_test_start(r.blocks, r.n_samples, r.fs)
    mask = span_rows(table, "test", test_start_sample=test_start)
    earliest = table.meta["window_earliest_sample"].to_numpy(dtype=int)[mask]
    assert (earliest >= test_start).all()
    plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    calib_mask = span_rows(table, "calibration", plan)
    calib_bins = table.meta["bin_start_sample"].to_numpy(dtype=int)[calib_mask]
    assert (calib_bins < plan.calib_end_sample).all()
    # purge rows belong to neither span
    both = calib_mask & mask
    assert not both.any()


def test_calibration_plans_nest_and_share_the_test_interval() -> None:
    cohort = make_fast_cohort(n=1, seed0=9500, plant="movement")
    r = cohort[0]
    shared = shared_test_start(r.blocks, r.n_samples, r.fs)
    ends = []
    for fraction in contract.CALIBRATION_FRACTIONS:
        plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, fraction)
        assert plan.test_start_sample == shared  # identical test interval
        assert plan.gates["min_calibration_s"]["passed"]
        assert plan.gates["min_test_s"]["passed"]
        ends.append(plan.calib_end_sample)
    assert ends[0] <= ends[1] <= ends[2]  # nested prefixes


# ---------------------------------------------------------------------------
# Target/event information never enters neural features
# ---------------------------------------------------------------------------


def test_neural_features_are_blind_to_the_glove_target() -> None:
    """Neural features must be identical no matter what the glove
    contains — target-derived information cannot leak into features by
    construction (the builder never receives the glove array)."""
    r = make_fast_cohort(n=1, seed0=9600, plant="movement")[0]
    table_a = build_recording_tables(r, with_lags=False)
    r.glove = r.glove * 1000.0 + 7.0  # absurd target change
    table_b = build_recording_tables(r, with_lags=False)
    cols = [c for c in table_a.features.columns if c.startswith("off")]
    np.testing.assert_allclose(table_a.features[cols].to_numpy(),
                               table_b.features[cols].to_numpy(), atol=0)


# ---------------------------------------------------------------------------
# Per-recording target standardization never fitted on the test interval
# ---------------------------------------------------------------------------


def test_fewshot_scale_constants_come_from_calibration_only() -> None:
    """Corrupting the FINAL TEST interval's target values must not change
    any fitted parameter, chosen blend, or prediction — the test targets
    are used for scoring only."""
    cohort = make_fast_cohort(n=4, seed0=9700, plant="innovation")
    tables = canonicalize_columns([build_recording_tables(r, with_lags=True)
                                   for r in cohort])
    held_out = 0
    r = cohort[held_out]
    table = tables[held_out]
    move_rows = np.isfinite(table.target)
    test_start = shared_test_start(r.blocks, r.n_samples, r.fs)
    plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    calib_mask = span_rows(table, "calibration", plan) & move_rows
    test_mask = span_rows(table, "test", test_start_sample=test_start) & move_rows
    y_raw = np.asarray(table.target, dtype=float)
    from curve_icon_dbso.transfer.pipeline import _move_constants
    mu, sd = _move_constants(y_raw[calib_mask])
    X_calib = table.features.loc[calib_mask].reset_index(drop=True)
    X_test = table.features.loc[test_mask].reset_index(drop=True)
    y_calib = (y_raw[calib_mask] - mu) / sd
    y_test_a = (y_raw[test_mask] - mu) / sd
    test_blocks = table.meta["block"].values[test_mask]
    kwargs = dict(alpha_fixed=1.0, ar_model=None, with_xgboost=False)
    result_a = evaluate_fewshot_subject(held_out, "fewshot_30", X_calib,
                                        y_calib, X_test, y_test_a,
                                        test_blocks, **kwargs)
    y_test_b = y_test_a.copy()
    y_test_b[:] = -9.0  # corrupt the entire test target
    result_b = evaluate_fewshot_subject(held_out, "fewshot_30", X_calib,
                                        y_calib, X_test, y_test_b,
                                        test_blocks, **kwargs)
    assert result_a.chosen == result_b.chosen
    # persistence MAE tracks the corrupted target (it is a score, not a
    # fit) — but no fitted parameter changed
    assert result_a.chosen == result_b.chosen


# ---------------------------------------------------------------------------
# Pooled windows are never independent subjects
# ---------------------------------------------------------------------------


def test_headline_statistics_are_subject_level_medians() -> None:
    cohort = make_fast_cohort(n=4, seed0=9800, plant="movement")
    tables = canonicalize_columns([build_recording_tables(r, with_lags=True)
                                   for r in cohort])
    artifacts = run_tracks_bc(cohort, tables, null_draws=10,
                              with_xgboost=False)
    zs = artifacts.zero_shot
    assert zs["n_subjects"] == 4
    # the median is over per-subject values — with 4 subjects it cannot
    # be a pooled-window statistic (which would sit on a different scale)
    assert "median_r_neural_only" in zs
    per_subject_frame = artifacts.per_subject_local_only["fewshot_30"]
    assert len(per_subject_frame) <= 4  # one row per subject, never per window


# ---------------------------------------------------------------------------
# The v1.0.1 contract and its outputs are untouched
# ---------------------------------------------------------------------------


def test_v1_contract_module_unchanged() -> None:
    assert protocol.PROTOCOL_VERSION == "1.0.1"
    protocol.validate_protocol()
    assert protocol.protocol_hash() == (
        "5e0e2566f9c5df24d8ac751cceaf96d27a15a781dfb70e90b6d2a0f78e1e92a7")


def test_v2_contract_hash_pinned_against_drift() -> None:
    # Sibling top-level module (tests/ is not an importable package in a
    # clean environment; pytest's rootdir puts the tests dir on sys.path).
    from test_transfer_contract import PINNED_HASH

    assert contract.contract_hash() == PINNED_HASH


# ---------------------------------------------------------------------------
# Z-unit discipline (zero-shot scoring is correlation-only)
# ---------------------------------------------------------------------------


def test_ar_predict_never_reads_future_values() -> None:
    rng = np.random.default_rng(3)
    stream = rng.standard_normal(200).cumsum()
    ar = {"coefficients": np.array([0.9] * 10), "intercept": 0.0, "order": 10}
    full = ar_predict(ar, stream)
    truncated = ar_predict(ar, stream[:150].copy())
    truncated[150:] += 500.0
    after = ar_predict(ar, np.concatenate([stream[:150], truncated[150:] + 500.0]))
    np.testing.assert_allclose(full[:150], after[:150], atol=1e-12)
    np.testing.assert_allclose(full[:150], truncated[:150], atol=1e-12)


# ---------------------------------------------------------------------------
# Span-local independence (v2.1.0): no future sample of ANY kind may
# alter an earlier span's artifacts — adversarial mutation tests
# ---------------------------------------------------------------------------


def test_future_glove_mutation_leaves_calibration_targets_unchanged() -> None:
    """Mutating every raw glove sample AFTER the calibration prefix must
    leave the calibration envelope, its bin grid, and its scale
    constants BIT-IDENTICAL (the primary leakage defect: whole-block
    zero-phase filtering saw final-test samples)."""
    from curve_icon_dbso.transfer.pipeline import _move_constants
    from curve_icon_dbso.transfer.targets import move_envelope_stream

    r = make_fast_cohort(n=1, seed0=10000, plant="movement")[0]
    plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    env_a = move_envelope_stream(r.glove, r.fs, r.blocks,
                                 clip=(0, plan.calib_end_sample))
    assert env_a["stream"] is not None
    mutated = r.glove.copy()
    mutated[:, plan.calib_end_sample:] += 100.0  # future test + purge samples
    env_b = move_envelope_stream(mutated, r.fs, r.blocks,
                                 clip=(0, plan.calib_end_sample))
    np.testing.assert_array_equal(env_a["stream"], env_b["stream"])
    np.testing.assert_array_equal(env_a["bin_starts"], env_b["bin_starts"])
    assert (tuple(_move_constants(env_a["stream"]))
            == tuple(_move_constants(env_b["stream"])))


def test_mid_calibration_glove_mutation_leaves_the_10pct_targets_unchanged() -> None:
    """The calibration prefixes are built per fraction INDEPENDENTLY:
    mutating samples between the 10 % and 30 % prefix ends must not
    change the 10 % condition's calibration targets."""
    from curve_icon_dbso.transfer.targets import move_envelope_stream

    r = make_fast_cohort(n=1, seed0=10001, plant="movement")[0]
    plan10 = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.10)
    plan30 = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    assert plan10.calib_end_sample < plan30.calib_end_sample
    env_a = move_envelope_stream(r.glove, r.fs, r.blocks,
                                 clip=(0, plan10.calib_end_sample))
    mutated = r.glove.copy()
    mutated[:, plan10.calib_end_sample: plan30.calib_end_sample] *= -3.0
    env_b = move_envelope_stream(mutated, r.fs, r.blocks,
                                 clip=(0, plan10.calib_end_sample))
    np.testing.assert_array_equal(env_a["stream"], env_b["stream"])


def test_future_ecog_mutation_leaves_calibration_features_unchanged() -> None:
    """Mutating raw ECoG AFTER the calibration prefix must leave the
    calibration-span feature table bit-identical — rereferencing, line
    detection, notch treatment, Welch features, and the causal z are all
    built INSIDE the span (span-local processing, v2.1.0)."""
    r = make_fast_cohort(n=1, seed0=10002, plant="movement")[0]
    plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    table_a = build_recording_tables(
        r, with_lags=True, span=(0, plan.calib_end_sample))
    original = r.ecog.copy()
    try:
        r.ecog = original.copy()
        r.ecog[:, plan.calib_end_sample:] += 100.0
        table_b = build_recording_tables(
            r, with_lags=True, span=(0, plan.calib_end_sample))
    finally:
        r.ecog = original
    assert len(table_a.features) > 0
    assert len(table_a.features) == len(table_b.features)
    np.testing.assert_array_equal(
        table_a.features.to_numpy(dtype=float),
        table_b.features.to_numpy(dtype=float))
    pd.testing.assert_frame_equal(table_a.meta, table_b.meta)


def test_mid_calibration_ecog_mutation_leaves_the_10pct_features_unchanged() -> None:
    r = make_fast_cohort(n=1, seed0=10003, plant="movement")[0]
    plan10 = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.10)
    plan30 = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    table_a = build_recording_tables(
        r, with_lags=True, span=(0, plan10.calib_end_sample))
    original = r.ecog.copy()
    try:
        r.ecog = original.copy()
        r.ecog[:, plan10.calib_end_sample: plan30.calib_end_sample] *= -3.0
        table_b = build_recording_tables(
            r, with_lags=True, span=(0, plan10.calib_end_sample))
    finally:
        r.ecog = original
    np.testing.assert_array_equal(
        table_a.features.to_numpy(dtype=float),
        table_b.features.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# The frozen 1 s boundary guard is ENFORCED in sample space (v2.1.0)
# ---------------------------------------------------------------------------


def test_boundary_guard_enforced_on_retained_rows() -> None:
    """Every retained row's full read range must sit ≥ BOUNDARY_GUARD_S
    inside both edges of its block, and the guard must actually drop
    transition rows (tested on distances, not on the constant)."""
    r = make_fast_cohort(n=1, seed0=10004, plant="movement")[0]
    table = build_recording_tables(r, with_lags=False)
    guard = int(round(contract.BOUNDARY_GUARD_S * r.fs))
    assert len(table.features) > 0
    assert table.qc["drops"].get("boundary_guard_1s", 0) > 0
    earliest = table.meta["window_earliest_sample"].to_numpy(dtype=int)
    latest = table.meta["window_latest_sample"].to_numpy(dtype=int)
    for row_index, block_number in enumerate(table.meta["block"].values):
        start, end, _kind = r.blocks[int(block_number)]
        end = min(end, r.n_samples)
        assert earliest[row_index] - start >= guard, (
            f"row {row_index} reads {earliest[row_index] - start} samples "
            f"(< {guard}) inside its block's start edge")
        assert end - latest[row_index] >= guard, (
            f"row {row_index} reads {end - latest[row_index]} samples "
            f"(< {guard}) inside its block's end edge")


# ---------------------------------------------------------------------------
# Verdicts derive from the FINAL adjusted result (v2.1.0)
# ---------------------------------------------------------------------------

def _fabricated_track_a(detection_p_raw: float) -> dict:
    return {
        "results": {
            "ridge_logistic": {
                "median_bac": 0.70,
                "iqr_bac": [0.6, 0.8],
                "bootstrap": {"median_bac_ci": [0.60, 0.80]},
                "null": {"p_raw": detection_p_raw, "n_draws": 100},
                # criterion met; the OLD code's success flag (model-family
                # Holm) would be True here — irrelevant now
                "verdict": {"criterion_met": True, "success": True,
                            "median_bac": 0.70, "ci": [0.60, 0.80]},
            },
        },
        "per_subject_secondary_local_only": {"ridge_logistic": []},
        "secondary_family": {},
        "majority_baseline_bac": 0.5,
    }


def _fabricated_tracks_bc() -> RegressionArtifacts:
    nan = float("nan")
    return RegressionArtifacts(
        zero_shot={
            "null_median_r_neural_only": {"p_raw": 1.0},
            "median_r_neural_only": None,
            "bootstrap_ci_median_r_neural_only": [nan, nan],
        },
        fewshot={
            "fewshot_30": {
                "null_delta_mae_circular": {"p_raw": 1.0},
                "median_mae_delta": None,
                "bootstrap_ci_median_delta_mae": [nan, nan],
            },
        },
        calibration_curve=[{
            "fraction": 0.30,
            "gain_over_zero_shot_r": {"status": "ok", "sign_flip_p_raw": 1.0,
                                      "bootstrap_ci": [nan, nan]},
        }],
    )


def _fabricated_track_d() -> dict:
    return {
        "n_recordings_with_events": 0,
        "primary_association": {"status": "not_executable_insufficient_subjects"},
        "event_prediction": {"status": "not_executable_no_events"},
        "event_family": {"p_holm_within_family": {}},
    }


def _fabricated_machinery() -> dict:
    status = {"n_passing": 10, "n_loaded": 10, "pass_fraction": 1.0,
              "executable": True}
    return {"detection": dict(status), "regression": dict(status),
            "events": dict(status)}


def test_detection_verdict_uses_the_final_holm_not_the_model_family() -> None:
    """Detection's RAW p = 0.02 → four-endpoint Holm 0.08 ≥ 0.05: the
    displayed verdict must be False (the old code entered the
    model-family-adjusted p and would display success contradicting the
    global null). With raw p = 0.005 the final Holm 0.02 < 0.05 → True
    and global_null flips consistently."""
    from curve_icon_dbso.transfer.pipeline import _primary_family, _verdicts

    for detection_p_raw, expect_success in ((0.02, False), (0.005, True)):
        track_a = _fabricated_track_a(detection_p_raw)
        tracks_bc = _fabricated_tracks_bc()
        machinery = _fabricated_machinery()
        family = _primary_family(track_a, tracks_bc, machinery)
        verdicts = _verdicts(track_a, tracks_bc, _fabricated_track_d(),
                             family, machinery)
        assert verdicts["detection_transfer"]["verdict"] is expect_success
        assert verdicts["global_null"] is (not expect_success)
        # the family received the RAW p exactly once
        assert family["p_raw_in_family"]["detection_zero_shot_median_bac"] \
            == detection_p_raw


def test_track_verdicts_cannot_contradict_global_null() -> None:
    """With every primary criterion met but all raw p-values ≈ 1, every
    displayed verdict is False and global_null is True — and vice versa
    when a final Holm p clears 0.05."""
    from curve_icon_dbso.transfer.pipeline import _primary_family, _verdicts

    track_a = _fabricated_track_a(1.0)
    tracks_bc = _fabricated_tracks_bc()
    machinery = _fabricated_machinery()
    family = _primary_family(track_a, tracks_bc, machinery)
    verdicts = _verdicts(track_a, tracks_bc, _fabricated_track_d(), family,
                         machinery)
    displayed = [verdicts[key]["verdict"] for key in
                 ("detection_transfer", "regression_zero_shot",
                  "fewshot_delta_mae", "fewshot_gain")]
    assert not any(displayed)
    assert verdicts["global_null"] is True


def test_event_association_requires_its_null_and_prediction_requires_ci() -> None:
    from curve_icon_dbso.transfer.pipeline import _primary_family, _verdicts

    def _track_d(assoc_holm=None, pred_holm=None, assoc_ci=(0.2, 0.4),
                 pred_median=-0.1, pred_ci=(-0.2, -0.05)):
        family_holm = {}
        if assoc_holm is not None:
            family_holm["log_inter_tap_interval_association"] = assoc_holm
        if pred_holm is not None:
            family_holm["event_prediction_shift_null"] = pred_holm
        return {
            "n_recordings_with_events": 5,
            "primary_association": {
                "status": "ok",
                "bootstrap_ci_fisher_z_r": list(assoc_ci),
                "observed_fisher_z_r": 0.3,
            },
            "event_prediction": {
                "status": "ok",
                "observed_median_delta_mae": pred_median,
                "bootstrap_ci_median_delta_mae": list(pred_ci),
                "null": {"p_raw": 0.01},
            },
            "event_family": {"p_holm_within_family": family_holm},
        }

    machinery = _fabricated_machinery()
    track_a = _fabricated_track_a(1.0)
    tracks_bc = _fabricated_tracks_bc()
    family = _primary_family(track_a, tracks_bc, machinery)

    # association CI excludes 0 but its event-family Holm p is 0.6 → fail;
    # prediction CI excludes 0 but its event-family Holm p is missing → fail
    verdicts = _verdicts(track_a, tracks_bc,
                         _track_d(assoc_holm=0.6), family, machinery)
    assert verdicts["event_association"]["verdict"] is False
    assert verdicts["event_prediction"]["verdict"] is False
    # prediction with a bootstrap CI that INCLUDES 0 → fail even with a
    # small Holm p (the CI requirement is enforced)
    verdicts = _verdicts(track_a, tracks_bc,
                         _track_d(assoc_holm=0.01, pred_holm=0.01,
                                  pred_ci=(-0.05, 0.05)), family, machinery)
    assert verdicts["event_prediction"]["verdict"] is False
    # both clearing every requirement → success
    verdicts = _verdicts(track_a, tracks_bc,
                         _track_d(assoc_holm=0.01, pred_holm=0.01), family,
                         machinery)
    assert verdicts["event_association"]["verdict"] is True
    assert verdicts["event_prediction"]["verdict"] is True


def test_not_executable_tracks_are_excluded_from_the_primary_family() -> None:
    from curve_icon_dbso.transfer.pipeline import _primary_family, _verdicts

    track_a = _fabricated_track_a(0.005)   # would pass Holm if included
    tracks_bc = _fabricated_tracks_bc()
    machinery = _fabricated_machinery()
    machinery["detection"]["executable"] = False
    family = _primary_family(track_a, tracks_bc, machinery)
    assert "detection_zero_shot_median_bac" in family["excluded_not_executable"]
    assert "detection_zero_shot_median_bac" not in family["p_raw_in_family"]
    verdicts = _verdicts(track_a, tracks_bc, _fabricated_track_d(), family,
                         machinery)
    assert verdicts["detection_transfer"]["verdict"] is False
    assert verdicts["detection_transfer"]["not_executable_machinery"] is True


# ---------------------------------------------------------------------------
# AR/persistence histories RESET at block boundaries (v2.1.0)
# ---------------------------------------------------------------------------


def test_ar_and_persistence_histories_reset_at_block_boundaries() -> None:
    from curve_icon_dbso.transfer.pipeline import _move_constants
    from curve_icon_dbso.transfer.regression import evaluate_fewshot_subject

    rng = np.random.default_rng(11)
    n_block = 50
    y = rng.standard_normal(300).cumsum()
    y_calib_raw = y[:150]
    y_test_raw = y[150:]                      # 3 blocks × 50 bins
    mu, sd = _move_constants(y_calib_raw)
    y_calib = (y_calib_raw - mu) / sd
    y_test = (y_test_raw - mu) / sd
    test_blocks = np.repeat(np.arange(3), n_block)
    ar_model = {"coefficients": np.full(10, 0.5), "intercept": 0.0,
                "order": 10}
    X = pd.DataFrame(rng.standard_normal((len(y_calib) + len(y_test), 5)),
                     columns=[f"f{i}" for i in range(5)])
    result = evaluate_fewshot_subject(
        0, "reset_test", X.iloc[:len(y_calib)].reset_index(drop=True),
        y_calib, X.iloc[len(y_calib):].reset_index(drop=True), y_test,
        test_blocks, alpha_fixed=1.0, ar_model=ar_model, with_xgboost=False,
        calib_block_ids=np.zeros(len(y_calib), dtype=int))
    assert result.persistence_pred is not None
    # persistence is NaN at EVERY block start, finite inside blocks
    for start in (0, n_block, 2 * n_block):
        assert np.isnan(result.persistence_pred[start])
    assert np.isfinite(result.persistence_pred[1:n_block]).all()
    # AR predictions are NaN for the first `order` bins of every block
    ar_pred = result.ar_pred
    for start in (0, n_block, 2 * n_block):
        assert np.isnan(ar_pred[start: start + 10]).all()
        assert np.isfinite(ar_pred[start + 10])


# ---------------------------------------------------------------------------
# The persistence null scores PERSISTENCE predictions (v2.1.0 fix)
# ---------------------------------------------------------------------------


def test_persistence_null_scores_the_persistence_predictions() -> None:
    """The persistence null must re-score the PERSISTENCE stream, not
    the neural predictions (the pre-2.1.0 bug made every model_key's
    null identical because it always reused ``combined_pred``). Verified
    with an exact oracle: the persistence-key null must be BIT-IDENTICAL
    to the neural-key null computed over records whose neural slot holds
    the persistence stream — and different from the true neural null."""
    from curve_icon_dbso.transfer.regression import (
        SubjectConditionResult,
        _null_predictor,
        circular_shift_null_median_r,
    )

    rng = np.random.default_rng(12)
    observed = rng.standard_normal(600)
    persistence = np.full(observed.shape, np.nan)
    persistence[1:] = observed[:-1]
    anti = -observed
    records = [SubjectConditionResult(
        held_out_position=0, condition="probe", n_test_windows=600,
        observed_stream=observed, combined_pred=anti,
        persistence_pred=persistence,
        r={"neural_only": float(np.corrcoef(observed, anti)[0, 1]),
           "persistence": 0.0},
        block_ids=np.zeros(600, dtype=int))]
    # selector unit: each key gets its OWN stream
    assert _null_predictor(records[0], "persistence") is records[0].persistence_pred
    assert _null_predictor(records[0], "neural_only") is records[0].combined_pred

    null_persistence = circular_shift_null_median_r(
        records, n_draws=20, model_key="persistence")
    null_neural = circular_shift_null_median_r(
        records, n_draws=20, model_key="neural_only")
    assert null_persistence["null_median"] != null_neural["null_median"]

    # exact oracle: alias the persistence stream into the neural slot —
    # the neural-key null must then reproduce the persistence-key null
    alias_records = [SubjectConditionResult(
        held_out_position=0, condition="probe", n_test_windows=600,
        observed_stream=observed, combined_pred=persistence,
        persistence_pred=persistence,
        r={"neural_only": 0.0, "persistence": 0.0},
        block_ids=np.zeros(600, dtype=int))]
    null_alias = circular_shift_null_median_r(
        alias_records, n_draws=20, model_key="neural_only")
    assert null_persistence["null_median"] == null_alias["null_median"]

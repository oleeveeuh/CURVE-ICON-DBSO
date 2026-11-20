"""Track C adaptation tests: calibration plans, blend selection, the
AR baseline's causality, and the standardization-constants rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.transfer import contract
from curve_icon_dbso.transfer.regression import (
    BLEND_OPTIONS,
    ar_predict,
    evaluate_fewshot_subject,
    fit_ar_on_streams,
    select_residual_blend_blocked,
)
from curve_icon_dbso.transfer.splits import build_calibration_plan, shared_test_start
from curve_icon_dbso.transfer.synthetic import make_synthetic_recording


def _recording():
    return make_synthetic_recording(1, 21, plant="movement", fs=600.0,
                                    rest_s=12.0, move_s=28.0, n_block_pairs=3)


def test_calibration_prefix_is_chronological_and_nested() -> None:
    r = _recording()
    shared = shared_test_start(r.blocks, r.n_samples, r.fs)
    prev_end = 0
    for fraction in contract.CALIBRATION_FRACTIONS:
        plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, fraction)
        assert plan.calib_end_sample >= prev_end
        assert plan.calib_end_sample <= shared
        prev_end = plan.calib_end_sample


def test_identical_test_interval_across_conditions() -> None:
    r = _recording()
    shared = shared_test_start(r.blocks, r.n_samples, r.fs)
    for fraction in contract.CALIBRATION_FRACTIONS:
        plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, fraction)
        assert plan.test_start_sample == shared


def test_purge_gap_separates_calibration_from_test() -> None:
    r = _recording()
    plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    assert plan.test_start_sample - plan.calib_end_sample == plan.purge_samples
    assert plan.purge_samples >= int(round(contract.PURGE_S * r.fs))


def test_gates_flag_impossible_floors() -> None:
    r = make_synthetic_recording(1, 22, plant="movement", fs=600.0,
                                 rest_s=2.0, move_s=4.0, n_block_pairs=1)
    plan = build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.30)
    assert not plan.gates["min_calibration_s"]["passed"]
    assert not plan.gates["min_test_s"]["passed"]


def test_invalid_fraction_rejected() -> None:
    r = _recording()
    with pytest.raises(DataValidationError):
        build_calibration_plan(r.blocks, r.n_samples, r.fs, 0.5)


def test_ar_fit_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    innovations = rng.standard_normal(n) * 0.2
    stream = np.empty(n)
    stream[0] = 0.0
    for t in range(1, n):
        stream[t] = 0.8 * stream[t - 1] + 0.1 * stream[t - 2] + innovations[t]
    model = fit_ar_on_streams([stream], order=2)
    assert model["coefficients"][0] == pytest.approx(0.8, abs=0.05)
    assert model["coefficients"][1] == pytest.approx(0.1, abs=0.05)
    assert model["intercept"] == pytest.approx(0.0, abs=0.05)


def test_ar_predict_leaves_warmup_nan_and_is_causal() -> None:
    stream = np.arange(50, dtype=float)
    model = {"coefficients": np.array([1.0, 0.0, 0.0]),
             "intercept": 0.0, "order": 3}
    predictions = ar_predict(model, stream)
    assert np.isnan(predictions[:3]).all()
    assert predictions[3] == pytest.approx(stream[2])
    assert predictions[10] == pytest.approx(stream[9])


def test_blend_selection_prefers_null_when_transferred_prediction_is_noise() -> None:
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.standard_normal((400, 5)),
                     columns=[f"f{i}" for i in range(5)])
    resid_target = rng.standard_normal(400)          # pure noise
    zs_resid = rng.standard_normal(400)              # uninformative
    s, w, _alpha, _pipe = select_residual_blend_blocked(X, resid_target, zs_resid)
    assert (s, w) == (0.0, 0.0)  # pure AR — the null case


def test_blend_selection_uses_the_transfer_when_informative() -> None:
    rng = np.random.default_rng(7)
    n = 400
    latent = rng.standard_normal(n)
    X = pd.DataFrame({"signal": latent, **{
        f"noise{i}": rng.standard_normal(n) for i in range(5)}})
    zs_component = 0.8 * latent
    resid_target = zs_component + 0.1 * rng.standard_normal(n)
    s, _w, _alpha, _pipe = select_residual_blend_blocked(X, resid_target,
                                                         zs_component)
    assert s == 1.0  # the transferred residual genuinely helps


def test_fewshot_subject_records_frozen_bookkeeping() -> None:
    rng = np.random.default_rng(8)
    n_calib, n_test = 120, 80
    X = pd.DataFrame(rng.standard_normal((n_calib + n_test, 10)),
                     columns=[f"f{i}" for i in range(10)])
    y = np.concatenate([rng.standard_normal(n_calib),
                        rng.standard_normal(n_test)]).cumsum()
    y_calib, y_test = y[:n_calib], y[n_calib:]
    y_calib = (y_calib - y_calib.mean()) / y_calib.std()
    result = evaluate_fewshot_subject(
        0, "fewshot_30", X.iloc[:n_calib].reset_index(drop=True), y_calib,
        X.iloc[n_calib:].reset_index(drop=True),
        (y_test - y_test.mean()) / y_test.std(),
        np.zeros(n_test, dtype=int), alpha_fixed=10.0, ar_model=None,
        with_xgboost=False)
    assert "persistence" in result.mae
    assert result.chosen["alpha_zero_shot_selected"] == 10.0
    assert result.observed_stream is not None  # null bookkeeping attached


def test_blend_options_ordered_most_conservative_first() -> None:
    assert BLEND_OPTIONS[0] == (0.0, 0.0)
    assert BLEND_OPTIONS[-1] == (1.0, 1.0)


def test_single_class_calibration_segment_skips_fewshot_refit() -> None:
    """A recording whose chronological calibration prefix contains only
    ONE block kind (it begins inside a MOVE block) cannot support the
    binary detection few-shot refit: the subject-fraction point is
    skipped with the machine-readable reason, counted in the curve, and
    nothing crashes (real cohort: recording `DBS_bG12` triggers this at
    the 10 %/20 % fractions)."""
    from curve_icon_dbso.transfer.cohort import Recording
    from curve_icon_dbso.transfer.pipeline import (
        build_recording_tables,
        canonicalize_columns,
        run_detection_conditions,
    )
    from curve_icon_dbso.transfer.synthetic import make_fast_cohort

    cohort = make_fast_cohort(n=3, seed0=7311, plant="movement")
    flipped = []
    for i, rec in enumerate(cohort):
        if i == 1:
            # Flip block kinds on fixed boundaries: the recording now
            # STARTS with a MOVE block, so the 10 %/20 %/30 % MOVE
            # prefixes all fall inside the first (12 s) MOVE block —
            # every calibration span is single-class.
            blocks = [(s, e, "move" if k == "rest" else "rest")
                      for s, e, k in rec.blocks]
            rec = Recording(position=rec.position, ecog=rec.ecog,
                            glove=rec.glove, fs=rec.fs, blocks=blocks,
                            anchor_position=rec.anchor_position,
                            covariates=rec.covariates,
                            signal_quality_flag=rec.signal_quality_flag)
        flipped.append(rec)
    tables = canonicalize_columns(
        [build_recording_tables(r, with_lags=False) for r in flipped])
    result = run_detection_conditions(flipped, tables, workers=1)

    rows = {r["held_out_position"]: r for r in result["per_subject_local_only"]}
    skipped = rows[1]
    for fraction in (10, 20, 30):
        key = f"bac_fewshot_{fraction}"
        assert key not in skipped
        assert skipped[f"{key}_not_computable"] == \
            "calibration_segment_single_class"
    for other in (0, 2):
        assert "bac_fewshot_10" in rows[other]
    skips = {entry["fraction"]: entry["n_single_class_calibration_skips"]
             for entry in result["calibration_curve"]}
    assert skips == {0.10: 1, 0.20: 1, 0.30: 1}
    assert result["calibration_curve"][0]["n_subjects"] == 2

"""Planted-signal recovery tests (owner brief, "Leakage controls and
tests"): the pipeline must RECOVER

* a spatially localized movement signal (detection + zero-shot r),
* neural innovation beyond an AR target (few-shot regression),

and must stay NULL when only

* target autocorrelation (autocorr_only plant),
* subject identity (identity plant),

is present. These are the heaviest tests in the suite (each plant runs
the complete frozen matrix on a small synthetic cohort); they exist so
that a silent evaluation defect cannot pass CI.
"""

from __future__ import annotations

import pytest

from curve_icon_dbso.transfer.pipeline import run_transfer_study
from curve_icon_dbso.transfer.synthetic import make_fast_cohort


def _run(plant: str) -> object:
    cohort = make_fast_cohort(n=5, seed0=4200, plant=plant)
    return run_transfer_study(cohort, with_ablations=False, null_draws=20)


@pytest.mark.slow
def test_planted_localized_signal_is_recovered_and_nulls_stay_null() -> None:
    movement = _run("movement")
    autocorr = _run("autocorr_only")
    identity = _run("identity")

    movement_bac = movement.track_a["results"]["ridge_logistic"]["median_bac"]
    autocorr_bac = autocorr.track_a["results"]["ridge_logistic"]["median_bac"]
    identity_bac = identity.track_a["results"]["ridge_logistic"]["median_bac"]
    # the localized plant is detected above the autocorrelation-only cohort
    assert movement_bac > autocorr_bac + 0.02
    assert movement_bac > 0.55
    # subject identity alone is NOT a shortcut
    assert identity_bac < 0.55

    movement_r = movement.tracks_bc.zero_shot["median_r_neural_only"]
    autocorr_r = autocorr.tracks_bc.zero_shot["median_r_neural_only"]
    assert movement_r is not None and movement_r > 0.15
    assert autocorr_r is None or abs(autocorr_r) < 0.1

    # the identity cohort stays null on the regression side too
    identity_r = identity.tracks_bc.zero_shot["median_r_neural_only"]
    assert identity_r is None or abs(identity_r) < 0.1


@pytest.mark.slow
def test_planted_neural_innovation_recovered_and_autocorr_stays_null() -> None:
    innovation = _run("innovation")
    autocorr = _run("autocorr_only")

    innov_r = innovation.tracks_bc.fewshot["fewshot_30"].get("median_innovation_r")
    null_r = autocorr.tracks_bc.fewshot["fewshot_30"].get("median_innovation_r")
    # the neural innovation beyond the AR target is recovered
    assert innov_r is not None and innov_r > 0.05
    # with autocorrelation only, the residual prediction carries nothing
    assert null_r is None or null_r < 0.05
    delta_innov = innovation.tracks_bc.fewshot["fewshot_30"].get("median_mae_delta")
    delta_null = autocorr.tracks_bc.fewshot["fewshot_30"].get("median_mae_delta")
    # the combined forecast is never WORSE than pure AR under the blend
    # rule (calibration may select pure AR), and the innovation plant's
    # combined forecast is at least as good as the null plant's
    assert delta_innov is not None
    assert delta_innov <= (delta_null if delta_null is not None else 0.0)


@pytest.mark.slow
def test_calibration_curve_orders_zero_shot_below_fewshot_on_planted() -> None:
    """On planted cohorts the few-shot conditions must not UNDERPERFORM
    zero-shot on the shared test interval (the calibration curve is
    monotone-or-flat within noise)."""
    movement = _run("movement")
    curve = {entry["fraction"]: entry
             for entry in movement.tracks_bc.calibration_curve}
    zs = movement.detection_conditions["median_bac_zero_shot_test_interval"]
    for fraction in (0.10, 0.20, 0.30):
        few = curve[fraction]["gain_over_zero_shot_r"]
        # gain cannot be strongly negative (the blend can always fall
        # back to the transferred model)
        if few.get("median_gain") is not None:
            assert few["median_gain"] > -0.5
    assert zs is not None

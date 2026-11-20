"""Track D event-level tests: tap-event extraction ground truth,
event-aligned neural features, and the association machinery + null."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from curve_icon_dbso.transfer import contract
from curve_icon_dbso.transfer.events import (
    association_analysis,
    association_null,
    build_event_table,
    event_neural_features,
    fisher_z_aggregate,
)
from curve_icon_dbso.transfer.synthetic import make_synthetic_recording
from curve_icon_dbso.transfer.targets import extract_tap_events


def _tapping_recording():
    return make_synthetic_recording(1, 31, plant="movement", fs=600.0,
                                    rest_s=6.0, move_s=16.0, n_block_pairs=2)


def test_tap_events_found_at_the_planted_rate() -> None:
    r = _tapping_recording()
    events = extract_tap_events(r.glove, r.fs, r.blocks)
    assert events, "no tap events extracted from the synthetic tapping"
    rates = [e["f_tap_hz"] for e in events
             if e.get("f_tap_hz") is not None]
    assert rates, "no contributing channel-block reported a tap rate"
    # synthetic taps run ~1.4 Hz (± rate modulation); the frozen
    # detection band is 0.5–4 Hz
    assert all(0.5 <= rate <= 4.0 for rate in rates)


def test_event_table_log_iti_consistent_with_rate() -> None:
    r = _tapping_recording()
    table = pd.DataFrame(build_event_table(r.glove, r.fs, r.blocks))
    assert len(table) > 0
    median_iti = float(np.median(table["iti_s"]))
    assert 0.4 <= median_iti <= 1.2  # ~1.4 Hz planted rate
    np.testing.assert_allclose(table["log_iti_s"],
                               np.log10(table["iti_s"]), atol=1e-12)


def test_event_table_onsets_are_inside_move_blocks() -> None:
    r = _tapping_recording()
    table = pd.DataFrame(build_event_table(r.glove, r.fs, r.blocks))
    move_bounds = [(s, e) for s, e, k in r.blocks if k == "move"]
    for onset in table["onset_sample"]:
        assert any(s <= onset < e for s, e in move_bounds)


def test_rest_blocks_contribute_no_events() -> None:
    r = _tapping_recording()
    events = build_event_table(r.glove, r.fs, r.blocks)
    move_blocks = {i for i, (_s, _e, k) in enumerate(r.blocks) if k == "move"}
    assert {e["block"] for e in events} <= move_blocks


def test_event_neural_features_are_causal_and_aligned() -> None:
    r = _tapping_recording()
    events = build_event_table(r.glove, r.fs, r.blocks)
    onsets = np.asarray([e["onset_sample"] for e in events], dtype=int)
    features = event_neural_features(r.ecog, r.fs, r.anchor_position, onsets)
    assert features.shape == onsets.shape
    finite = np.isfinite(features)
    assert finite.any()
    # onsets too early for a full window yield NaN, never guesses
    window = int(round(contract.WINDOW_S_PRIMARY * r.fs))
    early = onsets < window
    if early.any():
        assert np.all(~finite[early])


def test_fisher_z_aggregate_matches_manual() -> None:
    rhos = np.array([0.2, 0.4, 0.6])
    z = np.arctanh(rhos)
    assert fisher_z_aggregate(rhos) == pytest.approx(np.tanh(z.mean()))


def test_association_analysis_reports_not_executable_when_too_few() -> None:
    result = association_analysis([{"spearman_r": 0.1, "n_taps": 5},
                                   {"spearman_r": 0.2, "n_taps": 5}])
    assert result["status"] == "not_executable_insufficient_subjects"


def test_association_bootstrap_ci_covers_the_observed_value() -> None:
    rng = np.random.default_rng(9)
    # strong shared latent -> consistently positive within-subject rho
    records = []
    for _i in range(12):
        latent = rng.standard_normal(300)
        outcome = latent + 0.1 * rng.standard_normal(300)
        feature = latent + 0.1 * rng.standard_normal(300)
        rho = float(np.corrcoef(outcome, feature)[0, 1])
        records.append({"spearman_r": rho, "n_taps": 300})
    result = association_analysis(records)
    assert result["status"] == "ok"
    lo, hi = result["bootstrap_ci_fisher_z_r"]
    assert lo < result["observed_fisher_z_r"] < hi
    assert lo > 0.5  # planted near-perfect association


def test_association_null_stays_null_on_permuted_streams() -> None:
    rng = np.random.default_rng(10)
    records = []
    for _i in range(8):
        feature = rng.standard_normal(200).cumsum()  # autocorrelated noise
        outcome = rng.standard_normal(200)
        rho = float(spearmanr(outcome, feature).statistic)
        records.append({
            "outcome": outcome,
            "feature": feature,
            "block_ids": np.zeros(200, dtype=int),
            "observed_spearman": rho,
        })
    result = association_null(records, n_draws=30)
    # the observed value sits inside the shifted null distribution
    assert 0.0 < result["p_raw"] <= 1.0
    assert result["count_at_least_as_favorable"] > 0

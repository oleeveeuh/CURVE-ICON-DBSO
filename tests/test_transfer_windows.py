"""Window/feature causality tests (v2 contract §6): the frozen off-by-one
contract, strictly-past causal normalization, block/span boundary
guards, and boundary metadata used to forbid window crossing."""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso.transfer import contract
from curve_icon_dbso.transfer.features import build_grid_features, causal_trailing_z
from curve_icon_dbso.transfer.synthetic import make_synthetic_recording


def test_causal_trailing_z_never_uses_present_or_future() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0, -100.0, 5.0, 6.0])
    out = causal_trailing_z(values, trailing_bins=3, clip=5.0)
    # bin i must equal (x_i − mean(past 3))/std(past 3)
    for i in (3, 4, 7):
        past = values[max(0, i - 3):i]
        expected = (values[i] - past.mean()) / past.std()
        assert out[i] == pytest.approx(expected, abs=1e-12)
    # warm-up bins are NaN, never zero-filled guesses
    assert np.isnan(out[:3]).all()


def test_causal_trailing_z_clips() -> None:
    values = np.array([0.0, 0.0, 0.0, 0.0, 50.0])
    out = causal_trailing_z(values, trailing_bins=3, clip=1.0)
    assert out[4] <= 1.0


def test_causal_trailing_z_future_change_does_not_alter_past() -> None:
    rng = np.random.default_rng(7)
    base = rng.standard_normal(200).cumsum()
    mutated = base.copy()
    mutated[150:] += 500.0  # huge future change
    out_a = causal_trailing_z(base, 50, 5.0)
    out_b = causal_trailing_z(mutated, 50, 5.0)
    assert np.allclose(out_a[:150], out_b[:150], equal_nan=True)


def test_windows_never_contain_the_target_instant() -> None:
    """The lag-0 window ends at the target bin's start (center-based
    contract: end = round(center·fs), exclusive)."""
    recording = make_synthetic_recording(1, 11, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=1)
    table = build_grid_features(recording.ecog, recording.fs,
                                recording.anchor_position,
                                recording.blocks, with_lags=False)
    update = int(round(contract.UPDATE_S * recording.fs))
    for _, row in table.meta.iterrows():
        end0 = row["bin_start_sample"] + update
        assert end0 <= recording.blocks[0][1] or row["block"] != 0


def test_windows_cannot_start_before_their_block() -> None:
    recording = make_synthetic_recording(1, 12, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=1)
    table = build_grid_features(recording.ecog, recording.fs,
                                recording.anchor_position,
                                recording.blocks, with_lags=True)
    starts = table.meta["window_earliest_sample"].to_numpy(dtype=int)
    bins = table.meta["bin_start_sample"].to_numpy(dtype=int)
    assert (starts >= 0).all()
    assert (starts <= bins).all()  # windows end at/after the bin start


def test_block_boundaries_never_crossed() -> None:
    """Windows of block k never read samples from block k+1."""
    recording = make_synthetic_recording(1, 13, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=2)
    table = build_grid_features(recording.ecog, recording.fs,
                                recording.anchor_position,
                                recording.blocks, with_lags=True)
    update = int(round(contract.UPDATE_S * recording.fs))
    for _, row in table.meta.iterrows():
        block_end = recording.blocks[row["block"]][1]
        assert row["bin_start_sample"] + update <= block_end


def test_boundary_guard_enforced_in_sample_space() -> None:
    """The frozen 1.0 s boundary guard is ENFORCED by the builder
    itself (v2.1.0): retained rows' full read ranges sit ≥ 1 s inside
    both edges of their blocks, transition rows are actually dropped,
    and a window can never be within the guard of a REST/MOVE edge."""
    recording = make_synthetic_recording(1, 17, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=2)
    table = build_grid_features(recording.ecog, recording.fs,
                                recording.anchor_position,
                                recording.blocks, with_lags=False)
    guard = int(round(contract.BOUNDARY_GUARD_S * recording.fs))
    assert guard >= int(round(contract.WINDOW_S_PRIMARY * recording.fs))
    assert len(table.features) > 0
    assert table.qc["drops"].get("boundary_guard_1s", 0) > 0
    earliest = table.meta["window_earliest_sample"].to_numpy(dtype=int)
    latest = table.meta["window_latest_sample"].to_numpy(dtype=int)
    for row_index, block_number in enumerate(table.meta["block"].values):
        start, end, _kind = recording.blocks[int(block_number)]
        end = min(end, recording.ecog.shape[1])
        assert earliest[row_index] - start >= guard
        assert end - latest[row_index] >= guard


def test_grid_feature_table_shape_and_missingness() -> None:
    recording = make_synthetic_recording(1, 14, plant="missing_anchor",
                                         fs=600.0, rest_s=6.0, move_s=6.0,
                                         n_block_pairs=1, plant_offset_delta=1)
    table = build_grid_features(recording.ecog, recording.fs,
                                recording.anchor_position,
                                recording.blocks, with_lags=False)
    assert "grid_off_+0_missing" in table.features.columns
    # The dead anchor channel is a DYNAMIC (QC) absence: it contributes
    # no features (coverage < 1, valid offsets < 9) and is never
    # median-filled from other channels. Static missingness stays
    # geometric (the edge offset only).
    assert table.features["grid_off_+0_missing"].iloc[0] == 0.0
    assert table.features["coverage_frac"].iloc[0] < 1.0
    assert table.meta["n_offsets_valid"].max() <= 8


def test_nonfinite_dead_channel_survives_rereference_and_notch() -> None:
    """A real cohort dead contact is NON-FINITE (entirely NaN), not a
    finite flatline — recordings DBS_bG67/DBS_bG72 carry one. The grid
    and bipolar builders must both succeed: the dead channel/pair is
    excluded by the QC mask, never notched, never imputed, and its
    absence stays represented."""
    from curve_icon_dbso.transfer.features import (
        build_bipolar_features,
        prepare_strip,
    )

    recording = make_synthetic_recording(1, 18, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=1)
    recording.ecog[3, :] = np.nan          # an entirely dead contact
    prepared, ok, qc = prepare_strip(recording.ecog, recording.fs)
    assert not ok[3]                        # excluded from the mask
    assert ok.sum() == 7                    # every other channel OK
    assert not np.isfinite(prepared[3]).all()  # raw values kept, never imputed
    grid = build_grid_features(recording.ecog, recording.fs,
                               recording.anchor_position, recording.blocks,
                               with_lags=False)
    assert len(grid.features) > 0
    assert grid.features["coverage_frac"].iloc[0] < 1.0
    bipolar = build_bipolar_features(recording.ecog, recording.fs,
                                     recording.blocks, with_lags=False)
    assert len(bipolar.features) > 0        # dead pairs drop out, build lives


def test_prepare_strip_rereferences_and_records_line_noise() -> None:
    from curve_icon_dbso.signal_qc import common_median_reference as _cmr
    from curve_icon_dbso.transfer.features import prepare_strip
    from curve_icon_dbso.transfer.spatial import ok_channel_mask

    recording = make_synthetic_recording(1, 15, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=1)
    prepared, ok, qc = prepare_strip(recording.ecog, recording.fs)
    assert prepared.shape == recording.ecog.shape
    assert ok.sum() == 8
    # The rereference step itself zeroes the cross-channel median exactly
    # (the later line-noise notch is linear and may drift it slightly).
    referenced = _cmr(recording.ecog, ok_mask=ok_channel_mask(recording.ecog))
    assert np.abs(np.median(referenced[ok], axis=0)).max() < 1e-9
    # After the notch the residual median stays negligible vs channel scale.
    assert np.abs(np.median(prepared[ok], axis=0)).max() < 0.5
    assert "line_freq_hz" in qc


def test_feature_columns_carry_lag_suffix_only_for_regression() -> None:
    recording = make_synthetic_recording(1, 16, plant="movement", fs=600.0,
                                         rest_s=6.0, move_s=6.0,
                                         n_block_pairs=1)
    det = build_grid_features(recording.ecog, recording.fs,
                              recording.anchor_position, recording.blocks,
                              with_lags=False)
    reg = build_grid_features(recording.ecog, recording.fs,
                              recording.anchor_position, recording.blocks,
                              with_lags=True)
    assert any(c.endswith("_hfa") for c in det.features.columns)
    assert not any(c.endswith("_hfa") for c in reg.features.columns)
    assert any(c.endswith("_hfa_0.5s") for c in reg.features.columns)

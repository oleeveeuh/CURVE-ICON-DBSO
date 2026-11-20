"""Spatial mapping tests (v2 contract §5): grid-relative anchoring,
explicit missingness, dual-strip exclusion — no silent median pooling,
no invented anatomy."""

from __future__ import annotations

import numpy as np
import pytest

from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.transfer import contract
from curve_icon_dbso.transfer.spatial import (
    bipolar_pairs,
    build_grid_map,
    missingness_columns,
    ok_channel_mask,
)


def test_anchor_maps_to_channel_anchor_minus_one() -> None:
    grid = build_grid_map(8, 5)
    assert grid.offset_to_channel[contract.ANCHOR_OFFSET] == 4
    assert grid.offset_to_channel[-1] == 3
    assert grid.offset_to_channel[1] == 5


def test_offsets_beyond_strip_edges_are_explicitly_missing() -> None:
    grid = build_grid_map(8, 3)  # anchor near the left edge
    assert grid.offset_to_channel[0] == 2
    assert -1 not in grid.missing_offsets  # channel 1 exists
    assert -3 in grid.missing_offsets and -4 in grid.missing_offsets
    grid2 = build_grid_map(8, 7)
    assert 3 in grid2.missing_offsets and 4 in grid2.missing_offsets


def test_grid_is_cohort_uniform_nine_positions() -> None:
    for anchor in range(1, 9):
        grid = build_grid_map(8, anchor)
        assert set(grid.offset_to_channel) | set(grid.missing_offsets) == \
            set(contract.GRID_OFFSETS)
        assert contract.ANCHOR_OFFSET in grid.offset_to_channel


def test_second_strip_is_excluded_unanchored() -> None:
    grid = build_grid_map(16, 5)
    assert grid.n_strips == 2
    assert grid.unanchored_extra_channels == 8
    # no grid offset may map into the second strip (channels 8..15)
    assert all(0 <= ch <= 7 for ch in grid.offset_to_channel.values())
    summary = grid.coverage_summary()
    assert summary["unanchored_extra_channels"] == 8
    assert contract.SECOND_STRIP_POLICY == "excluded_unanchored"


def test_invalid_anchor_positions_rejected() -> None:
    with pytest.raises(DataValidationError):
        build_grid_map(8, 0)
    with pytest.raises(DataValidationError):
        build_grid_map(8, 9)
    with pytest.raises(DataValidationError):
        build_grid_map(7, 4)  # not a whole 8-contact strip


def test_missingness_columns_mark_absence_never_medians() -> None:
    grid = build_grid_map(8, 2)
    columns = missingness_columns(grid)
    marked = {k for k, v in columns.items() if v == 1.0}
    # offsets -2..-4 map below channel 0 for anchor 2
    assert "grid_off_-2_missing" in marked
    assert "grid_off_-4_missing" in marked
    # the anchor is never missing
    assert columns["grid_off_+0_missing"] == 0.0


def test_ok_channel_mask_flags_flat_and_nonfinite() -> None:
    rng = np.random.default_rng(0)
    ecog = rng.standard_normal((4, 100))
    ecog[1] = 0.0                       # flatline
    ecog[2, 10] = np.nan                # non-finite
    mask = ok_channel_mask(ecog)
    assert mask.tolist() == [True, False, False, True]


def test_bipolar_ablation_pairs_are_adjacent() -> None:
    grid = build_grid_map(8, 5)
    pairs = bipolar_pairs(grid)
    assert pairs == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

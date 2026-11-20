"""Grid-relative, motor-anchor-anchored spatial mapping (v2 contract §5).

No MNI coordinates, transforms, or per-contact anatomical labels exist
for this cohort (feasibility audit, 2026-11-19). The frozen v2 spatial
representation is therefore a FIXED GRID-RELATIVE TOPOLOGY: ordinal
positions −4…+4 relative to the subject's motor-anchor contact on the
anchored strip, with explicit missingness indicators. The second strip
of dual-strip recordings is excluded from the primary grid (its anchor
assignment is undocumented) — it is never silently median-pooled into
the grid. Nothing here guesses anatomy; everything derives from the
recorded anchor position and the uniform strip adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import DataValidationError
from . import contract


@dataclass
class GridMap:
    """The per-recording offset→channel map (position-labeled, never
    identity-labeled).

    ``offset_to_channel`` maps grid offsets present on the anchored
    strip to 0-based indices into the recording's strip-channel matrix
    (P-label order). Absent offsets are the explicit missing set.
    """

    anchor_position: int
    n_strip_channels: int
    offset_to_channel: dict[int, int]
    missing_offsets: tuple[int, ...] = field(default_factory=tuple)
    n_strips: int = 1
    unanchored_extra_channels: int = 0

    @property
    def offsets(self) -> tuple[int, ...]:
        return contract.GRID_OFFSETS

    def coverage_summary(self) -> dict:
        """Aggregate, identifier-free spatial-coverage record."""
        return {
            "anchor_position": int(self.anchor_position),
            "n_strip_channels": int(self.n_strip_channels),
            "n_grid_positions": len(contract.GRID_OFFSETS),
            "n_present": len(self.offset_to_channel),
            "n_missing": len(self.missing_offsets),
            "missing_offsets": [int(o) for o in self.missing_offsets],
            "n_strips": int(self.n_strips),
            "unanchored_extra_channels": int(self.unanchored_extra_channels),
        }


def build_grid_map(strip_channel_count: int, anchor_position: int) -> GridMap:
    """Map the frozen grid offsets onto a recording's strip channels.

    The recording's strip matrix is P-label order: channels 0–7 are
    strip 1 contacts 1–8; channels 8–15 (when present) are the
    undocumented second strip, which is excluded from the grid per the
    frozen ``SECOND_STRIP_POLICY``.

    Raises:
        DataValidationError: On an anchor position outside the frozen
            range or a recording with fewer than 8 strip channels.
    """
    lo, hi = contract.ANCHOR_POSITION_RANGE
    if not (lo <= int(anchor_position) <= hi):
        raise DataValidationError(
            f"anchor position {anchor_position} outside the frozen range [{lo}, {hi}]"
        )
    n_strips = strip_channel_count // 8
    if strip_channel_count < 8 or strip_channel_count % 8 != 0:
        raise DataValidationError(
            f"strip channel count {strip_channel_count} is not a whole number of "
            "8-contact strips"
        )
    anchor_idx = int(anchor_position) - 1  # strip-1 channel index
    offset_to_channel: dict[int, int] = {}
    missing: list[int] = []
    for offset in contract.GRID_OFFSETS:
        channel = anchor_idx + offset
        if 0 <= channel <= 7:
            offset_to_channel[offset] = channel
        else:
            missing.append(offset)
    return GridMap(
        anchor_position=int(anchor_position),
        n_strip_channels=int(strip_channel_count),
        offset_to_channel=offset_to_channel,
        missing_offsets=tuple(missing),
        n_strips=int(n_strips),
        unanchored_extra_channels=max(0, strip_channel_count - 8),
    )


def bipolar_pairs(grid_map: GridMap) -> list[tuple[int, int]]:
    """Adjacent-contact bipolar pairs on the anchored strip (the single
    prespecified spatial ablation): strip-1 contacts (1,2), (2,3), …
    (7,8) as ``(anode, cathode)`` channel-index pairs."""
    return [(i, i + 1) for i in range(7)]


def missingness_columns(grid_map: GridMap) -> dict[str, float]:
    """Static per-recording missingness indicator columns (frozen:
    absence is represented, never median-imputed)."""
    columns = {
        f"grid_off_{o:+d}_missing": float(o in grid_map.missing_offsets)
        for o in contract.GRID_OFFSETS
    }
    columns["grid_unanchored_extra_strip"] = float(
        grid_map.unanchored_extra_channels > 0
    )
    return columns


def ok_channel_mask(strip_ecog: np.ndarray) -> np.ndarray:
    """Whole-recording per-channel basic QC mask (finite, non-flat).

    Per-window QC (the stricter v1.0.1 §4 rules) runs later per window;
    this mask gates the common-median reference only.
    """
    strip_ecog = np.asarray(strip_ecog, dtype=float)
    masks = np.isfinite(strip_ecog).all(axis=1)
    for channel in range(strip_ecog.shape[0]):
        if masks[channel]:
            values = strip_ecog[channel]
            masks[channel] = (float(np.max(values) - np.min(values)) > 1e-10)
    if not masks.any():
        raise DataValidationError("no usable strip channel for the grid reference")
    return masks

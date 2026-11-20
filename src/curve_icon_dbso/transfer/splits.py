"""Calibration/test spans (v2 contract §9; owner floors 2026-11-19).

Calibration = chronological first {10, 20, 30} % of a recording's
eligible MOVE samples (nested prefixes). The final test interval is
IDENTICAL across conditions: everything after the 30 % prefix end plus
the frozen 2.0 s purge. Calibration spans may interleave REST blocks
(that data is chronologically earlier and legitimately labeled); the
regression track uses only MOVE bins of each span. Windows are forbidden
to cross a span, block, purge, or test boundary (enforced via the
feature builder's ``window_earliest_sample`` metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..errors import DataValidationError
from . import contract

if TYPE_CHECKING:
    from .features import FeatureTable


@dataclass
class CalibrationPlan:
    """One recording's span layout (sample-index space, absolute)."""

    n_samples: int
    fraction: float
    calib_end_sample: int            # calibration span = [0, calib_end)
    test_start_sample: int           # test span = [test_start, n_samples)
    calibration_move_samples: int
    test_move_samples: int
    purge_samples: int
    gates: dict

    def span_clip(self, span: str) -> tuple[int, int]:
        """Absolute sample bounds ``[lo, hi)`` of one span (calibration
        or test). Everything a span reads — ECoG preprocessing, features,
        glove targets — is restricted to these bounds (span-local
        processing, v2.1.0)."""
        if span == "calibration":
            return 0, self.calib_end_sample
        if span == "test":
            return self.test_start_sample, self.n_samples
        raise DataValidationError(f"unknown span '{span}'")

    def clip_blocks(self, blocks: list[tuple[int, int, str]],
                    span: str) -> list[tuple[int, int, str]]:
        """Clip the recording's blocks to one span (calibration or
        test). The clipped list is what the feature builder receives, so
        no window can start before the span or end after it."""
        lo, hi = self.span_clip(span)
        clipped = []
        for start, end, kind in blocks:
            s, e = max(start, lo), min(end, hi)
            if e > s:
                clipped.append((s, e, kind))
        return clipped


def _move_prefix_end(blocks: list[tuple[int, int, str]], n_prefix_samples: int) -> int:
    """Absolute sample index where the chronological MOVE prefix of
    ``n_prefix_samples`` samples ends (inside a MOVE block)."""
    remaining = n_prefix_samples
    for start, end, kind in blocks:
        if kind != "move":
            continue
        length = end - start
        if remaining <= length:
            return start + remaining
        remaining -= length
    raise DataValidationError("MOVE prefix extends past the last MOVE block")


def _move_samples_within(blocks: list[tuple[int, int, str]], lo: int, hi: int) -> int:
    total = 0
    for start, end, kind in blocks:
        if kind != "move":
            continue
        s, e = max(start, lo), min(end, hi)
        if e > s:
            total += e - s
    return total


def build_calibration_plan(
    blocks: list[tuple[int, int, str]], n_samples: int, fs: float,
    fraction: float = contract.FEWSHOT_PRIMARY_FRACTION,
) -> CalibrationPlan:
    """Span layout for one recording at one calibration fraction, with
    the frozen gates evaluated (``gates`` carries pass/fail + reasons).

    The test interval is computed from THIS fraction's prefix end plus
    the purge; callers wanting the identical-across-conditions test
    interval build every plan at the primary (largest) fraction for the
    test boundary and evaluate smaller fractions against it.
    """
    if fraction not in contract.CALIBRATION_FRACTIONS:
        raise DataValidationError(
            f"fraction {fraction} is not one of the frozen calibration fractions"
        )
    total_move = sum(e - s for s, e, kind in blocks if kind == "move")
    prefix_samples = int(np.ceil(fraction * total_move))
    calib_end = _move_prefix_end(blocks, prefix_samples)
    purge_samples = int(round(contract.PURGE_S * fs))
    # The test interval is IDENTICAL across calibration conditions (frozen
    # §9 rule): every plan uses the boundary derived from the PRIMARY
    # (largest) fraction, regardless of the fraction requested here.
    test_start = shared_test_start(blocks, n_samples, fs)
    calib_move = _move_samples_within(blocks, 0, calib_end)
    test_move = _move_samples_within(blocks, test_start, n_samples)
    gates = {
        "min_calibration_s": {
            "required": contract.MIN_CALIBRATION_S,
            "observed_s": calib_move / fs,
            "passed": bool(calib_move / fs >= contract.MIN_CALIBRATION_S),
        },
        "min_test_s": {
            "required": contract.MIN_TEST_S,
            "observed_s": test_move / fs,
            "passed": bool(test_move / fs >= contract.MIN_TEST_S),
        },
    }
    return CalibrationPlan(
        n_samples=int(n_samples),
        fraction=float(fraction),
        calib_end_sample=int(calib_end),
        test_start_sample=int(test_start),
        calibration_move_samples=int(calib_move),
        test_move_samples=int(test_move),
        purge_samples=int(purge_samples),
        gates=gates,
    )


def shared_test_start(
    blocks: list[tuple[int, int, str]], n_samples: int, fs: float,
) -> int:
    """The identical test-interval start for every calibration condition
    of one recording: after the largest (primary) fraction's prefix plus
    the purge."""
    total_move = sum(e - s for s, e, kind in blocks if kind == "move")
    prefix_samples = int(np.ceil(contract.FEWSHOT_PRIMARY_FRACTION * total_move))
    calib_end = _move_prefix_end(blocks, prefix_samples)
    return min(calib_end + int(round(contract.PURGE_S * fs)), n_samples)


def span_rows(table: FeatureTable, span: str, plan: CalibrationPlan | None = None,
              test_start_sample: int | None = None) -> np.ndarray:
    """Row-selection helper: given a FeatureTable built over the whole
    recording, return the boolean mask of rows belonging to ``span``.

    A row belongs to the calibration span when its bin starts before
    ``calib_end``; to the test span when its bin starts at/after
    ``test_start`` AND its earliest lagged window still starts at/after
    ``test_start`` (no window crosses the purge boundary — frozen
    leakage rule). Rows in the purge gap belong to neither span.
    """
    bin_start = np.asarray(table.meta["bin_start_sample"], dtype=int)
    if span == "calibration":
        if plan is None:
            raise DataValidationError("calibration span requires the plan")
        return bin_start < plan.calib_end_sample
    if span == "test":
        start = test_start_sample
        if start is None and plan is not None:
            start = plan.test_start_sample
        if start is None:
            raise DataValidationError("test span requires a start sample")
        earliest = np.asarray(table.meta["window_earliest_sample"], dtype=int)
        return (bin_start >= start) & (earliest >= start)
    raise DataValidationError(f"unknown span '{span}'")

"""Synthetic request payloads for the serving API (DAT-006).

Deterministic, schema-valid example rows derived from the synthetic
cohort pipeline — the same machinery that trains the served model — so
documentation, tests and (Phase 5) load traffic never need real data.

Everything here is safe to publish: the values are generated from fixed
seeds and carry no participant information. Helpers exist for the
malformed/boundary cases the API must reject too, but those fixtures
live in the test suite (``tests/test_api_contract.py``); this module
keeps only generators that produce *well-formed* payloads, plus
:func:`shifted_payload_rows` for the Phase 5 drift demonstration (its
output intentionally violates the serving limits and would be rejected
by ``/predict`` — it is for distribution-shift analysis, not requests).

This module must stay importable WITHOUT the ``api`` extra: stdlib +
numpy/pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import features as features_mod
from ..schema import FEATURE_LIMITS
from ..synthetic import generate_synthetic_cohort


def _cohort_feature_frame(n_subjects: int, seed: int) -> pd.DataFrame:
    """Feature matrix from the synthetic cohort, in canonical schema order."""
    from ..schema import FEATURES

    cohort = generate_synthetic_cohort(n_subjects=n_subjects, seed=seed)
    enriched = features_mod.add_clinical_composites(cohort)
    x, _, _ = features_mod.build_model_matrix(enriched)
    return x[FEATURES]


def synthetic_payload_rows(n_rows: int, *, seed: int = 42, n_subjects: int = 8) -> list[dict]:
    """``n_rows`` schema-valid feature rows (deterministic, order-stable).

    Rows are sampled from the synthetic cohort feature matrix with a
    fixed-seed permutation, so the output is reproducible and valid for
    ``POST /predict`` batches of any size up to the API limit.
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    frame = _cohort_feature_frame(n_subjects, seed)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(frame), size=n_rows)
    return [
        {name: float(frame.iloc[idx][name]) for name in frame.columns}
        for idx in indices
    ]


def boundary_payload_rows() -> list[dict]:
    """Rows exercising the schema limits (all-low / all-high / mixed).

    Every value is exactly at a documented limit and therefore valid —
    the boundary of the accepted range, not a violation (DAT-006).
    """
    lows = {name: limits[0] for name, limits in FEATURE_LIMITS.items()}
    highs = {name: limits[1] for name, limits in FEATURE_LIMITS.items()}
    mixed = dict(lows)
    for name in sorted(FEATURE_LIMITS)[::2]:  # deterministic alternation
        mixed[name] = FEATURE_LIMITS[name][1]
    return [lows, highs, mixed]


def shifted_payload_rows(
    n_rows: int, *, seed: int = 42, shift_fraction: float = 0.5
) -> list[dict]:
    """Distribution-shifted rows for the Phase 5 drift demonstration (OBS-006).

    Valid rows are shifted outward by ``shift_fraction`` of each feature's
    limit range — usually beyond the serving limits, so these are analysis
    fixtures, not request bodies.
    """
    if not 0.0 < shift_fraction <= 2.0:
        raise ValueError(f"shift_fraction must be in (0, 2], got {shift_fraction}")
    rows = synthetic_payload_rows(n_rows, seed=seed)
    shifted = []
    for row in rows:
        out = {}
        for name, value in row.items():
            low, high = FEATURE_LIMITS[name]
            out[name] = float(value + shift_fraction * (high - low))
        shifted.append(out)
    return shifted

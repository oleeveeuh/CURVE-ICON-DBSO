"""Population Stability Index (PSI) drift demonstration (OBS-006/OBS-007).

Educational signal for the PUBLIC SYNTHETIC track only. PSI compares the
binned distribution of incoming feature values against the versioned
reference aggregates baked into the synthetic model artifact at training
time.

This is NOT validated clinical drift monitoring:

* the reference is the deterministic synthetic cohort, not any real
  population;
* the 0.2 warning threshold is a commonly cited convention, not a
  calibrated decision rule for this (or any) clinical model;
* nothing here supports a claim about real-data shift or model decay.

Privacy: functions in this module accept and return AGGREGATES only —
bin edges, expected shares, and per-feature PSI floats. Raw feature
values never enter a report, and reports never contain per-row data.

Versioning: the aggregates live in the manifest under
``reference_aggregates`` and are fully deterministic given the training
seed (quantile bin edges of the training cohort), so an artifact rebuilt
from the same commit reproduces them byte-for-byte.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

#: OBS-007: PSI above this value is a WARNING for the demonstration —
#: a documented educational threshold, not a calibrated clinical rule.
DEFAULT_PSI_THRESHOLD: Final[float] = 0.2

#: Smoothing constant so empty bins (log of zero) stay finite. Chosen
#: once, documented here, and applied identically to both distributions.
_EPSILON: Final[float] = 1e-6

#: Number of quantile bins in the reference aggregates (fixed, versioned
#: implicitly by the artifact contents).
N_BINS: Final[int] = 10


def reference_aggregates(frame: pd.DataFrame, n_bins: int = N_BINS) -> dict[str, Any]:
    """Quantile bin edges + expected share per bin, per feature.

    Deterministic given ``frame`` (the training cohort's feature matrix):
    the edges are exact numpy quantiles and the shares are the observed
    bin fractions, rounded to 6 decimals for byte-stability.
    """
    aggregates: dict[str, Any] = {}
    for feature in frame.columns:
        values = frame[feature].to_numpy(dtype=float)
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)  # constant regions collapse; keep edges strict
        if len(edges) < 2:
            raise ValueError(f"feature {feature!r} has no value spread; cannot bin")
        idx = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        share = (counts / counts.sum()).round(6)
        aggregates[feature] = {
            # Full-precision edges: cohort values can sit exactly ON an edge,
            # and rounding the boundary would move such values into the
            # neighboring bin at scoring time (observed as PSI ~5 on a clean
            # sample). json round-trips these floats exactly, and the edges
            # remain deterministic given the training seed.
            "bin_edges": [float(e) for e in edges],
            "expected_share": [float(s) for s in share],
            "n_reference": int(len(values)),
        }
    return aggregates


def population_stability_index(
    expected: np.ndarray | list[float], actual: np.ndarray | list[float]
) -> float:
    """PSI between two binned distributions: ``sum (a_i - e_i) ln(a_i/e_i)``.

    Both inputs are share vectors (sum ≈ 1). Empty bins are smoothed with
    the module epsilon and the vectors renormalized, so a zero bin never
    produces ``log(0)``. Returns a non-negative float (0 for identical
    distributions).
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.ndim != 1 or e.size == 0:
        raise ValueError(f"expected/actual must be equal-shape 1-D vectors, got {e.shape} vs {a.shape}")
    e = np.clip(e, _EPSILON, None)
    a = np.clip(a, _EPSILON, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Share of ``values`` in each bin defined by strict inner edges."""
    edges_arr = np.asarray(edges, dtype=float)
    idx = np.clip(np.searchsorted(edges_arr[1:-1], values, side="right"), 0, len(edges_arr) - 2)
    counts = np.bincount(idx, minlength=len(edges_arr) - 1)
    return counts / max(counts.sum(), 1)


def compute_drift_report(
    reference_aggregates: dict[str, Any],
    frame: pd.DataFrame,
    *,
    threshold: float = DEFAULT_PSI_THRESHOLD,
) -> dict[str, Any]:
    """Per-feature PSI of ``frame`` against the reference aggregates.

    Returns an AGGREGATE-only report::

        {
          "threshold": 0.2,
          "features": {"<feature>": <psi float>, ...},   # aggregate floats
          "drifted_features": [...]                      # psi > threshold
        }

    Raises ``KeyError``-style ``ValueError`` when the artifact predates
    the Phase 5 aggregates (retrain to obtain them).
    """
    if not reference_aggregates:
        raise ValueError(
            "artifact has no reference_aggregates — it predates the Phase 5 "
            "drift demonstration; retrain with scripts/train_synthetic_model.py"
        )
    features: dict[str, float] = {}
    for feature, agg in reference_aggregates.items():
        if feature not in frame.columns:
            raise ValueError(f"frame is missing reference feature {feature!r}")
        values = frame[feature].to_numpy(dtype=float)
        actual = _bin_shares(values, agg["bin_edges"])
        features[feature] = round(
            population_stability_index(agg["expected_share"], actual), 6
        )
    drifted = sorted(f for f, psi in features.items() if psi > threshold)
    return {
        "threshold": threshold,
        "features": features,
        "drifted_features": drifted,
    }

"""The canonical, versioned serving feature schema (DAT-001).

ONE definition, used by every consumer of the serving contract:

* the deterministic synthetic **trainer** (``curve_icon_dbso.training``),
  which builds its model matrix from these feature names;
* the **API** request validation (``curve_icon_dbso.api.schemas`` builds
  its Pydantic models from these names and limits);
* the **model manifest**, which records ``schema_version`` and
  ``schema_hash`` so an artifact cannot be served against a schema it
  was not trained with (MOD-003);
* the **synthetic payload generator** (``curve_icon_dbso.api.payloads``).

The active schema is the prespecified 7-feature outcome contract used by
the current demonstrator (``curve-icon-demo`` /
``features.PRESPECIFIED_FEATURES``) — NOT the historical, invalidated
108-feature set. It is frozen: changing anything in this module changes
``schema_hash()`` and therefore invalidates every existing artifact
(they fail readiness by design until retrained).

Validation contract (DAT-003), enforced by the API:

* exactly the seven features below, all finite JSON numbers
  (integers accepted; strings, booleans, NaN/Infinity rejected);
* values must lie within :data:`FEATURE_LIMITS`;
* no unknown fields, no subject identifiers, no extra keys;
* batches of :data:`BATCH_MIN_ROWS`..:data:`BATCH_MAX_ROWS` rows;
* rows must be complete — the API does not impute missing features
  (missingness handling is a fold-local research-pipeline concern).
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from . import features as features_mod

#: Version of THIS schema document. Bump on any change to features,
#: limits, or batch bounds; old artifacts then fail readiness loudly.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: The prespecified serving features, in canonical (training-matrix)
#: order. Aliased from the demonstrator's contract so there is exactly
#: one list in the codebase (DAT-001/DAT-002).
FEATURES: Final[list[str]] = list(features_mod.PRESPECIFIED_FEATURES)

#: Target the served model predicts (recorded in the manifest; the API
#: itself never accepts a target).
TARGET: Final[str] = features_mod.TARGET_COLUMN

#: Subject grouping column. Research-pipeline concept only — the API
#: must never accept or require it (API-006).
GROUP: Final[str] = features_mod.GROUP_COLUMN

# Serving limits (low, high) per feature. These are documented
# synthetic operating limits for the demonstration API, chosen to be
# generous around the synthetic generator's ranges while still rejecting
# nonsense values (DAT-003 "documented physical or synthetic limits"):
#
# * hfo_power_mean: Welch band power is a PSD integral — nonnegative by
#   construction; the upper bound is a synthetic-scale operating limit
#   (the generator draws 0.5-2.0).
# * hfo_peak_freq_mean: the peak must lie inside the extraction band
#   (HFOConfig 80-500 Hz) — a hard extractor limit, not a choice.
# * the three asymmetry indices are |R - L| / (R + L) and are therefore
#   mathematically bounded to [0, 1].
# * age / years_since_diagnosis: adult-cohort plausibility bounds.
FEATURE_LIMITS: Final[dict[str, tuple[float, float]]] = {
    "hfo_power_mean": (0.0, 50.0),
    "hfo_peak_freq_mean": (80.0, 500.0),
    "Brady_Asymmetry": (0.0, 1.0),
    "Rigidity_Asymmetry": (0.0, 1.0),
    "Tremor_Asymmetry": (0.0, 1.0),
    "years_since_diagnosis": (0.0, 60.0),
    "age": (18.0, 100.0),
}

BATCH_MIN_ROWS: Final[int] = 1
#: Hard request-batch ceiling (DAT-003 / API-004).
BATCH_MAX_ROWS: Final[int] = 256

_REQUIRED_LIMIT_KEYS = frozenset(FEATURES)


def schema_document() -> dict:
    """The full schema contract as a plain dict (JSON-serializable).

    This document IS the versioned schema; its canonical JSON encoding
    is what :func:`schema_hash` fingerprints.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Prespecified 7-feature outcome serving schema (synthetic "
            "demonstration track; research only, not for clinical use)"
        ),
        "features": list(FEATURES),
        "feature_limits": {name: list(limits) for name, limits in FEATURE_LIMITS.items()},
        "target": TARGET,
        "group": GROUP,
        "batch_min_rows": BATCH_MIN_ROWS,
        "batch_max_rows": BATCH_MAX_ROWS,
        "row_completeness": "required — the API does not impute missing features",
    }


def canonical_json() -> str:
    """Canonical JSON encoding of the schema document (sorted keys).

    Canonicalization makes the hash independent of dict ordering, so the
    same contract always fingerprints identically.
    """
    return json.dumps(schema_document(), sort_keys=True, separators=(",", ":"))


def schema_hash() -> str:
    """SHA-256 fingerprint of the canonical schema document.

    Recorded in every artifact manifest and checked at load time: any
    change to the schema (features, limits, bounds, version) invalidates
    existing artifacts — the intended, fail-loud behavior.
    """
    return hashlib.sha256(canonical_json().encode("utf-8")).hexdigest()


def validate_limits() -> None:
    """Fail loudly if the limits table and feature list ever drift apart.

    Called at import time by the API layer; keeps the two structures
    honest without a separate test-only check.
    """
    if set(FEATURE_LIMITS) != _REQUIRED_LIMIT_KEYS:
        raise ValueError(
            "FEATURE_LIMITS keys must match FEATURES exactly: "
            f"limits={sorted(FEATURE_LIMITS)}, features={sorted(FEATURES)}"
        )
    for name, (low, high) in FEATURE_LIMITS.items():
        if not (low <= high):
            raise ValueError(f"Invalid limit range for {name}: [{low}, {high}]")

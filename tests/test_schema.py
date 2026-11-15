"""Canonical serving schema: versioning, hashing, limits, Pydantic round trip.

Pins DAT-001/002/003 for the Phase 2 serving contract: one versioned
definition, the real 7-feature prespecified set (never 108 placeholders),
and documented limits that the API and the manifest both derive from.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from curve_icon_dbso import features as features_mod
from curve_icon_dbso import schema as schema_mod
from curve_icon_dbso import tracking


def test_schema_features_are_the_prespecified_seven():
    assert schema_mod.FEATURES == features_mod.PRESPECIFIED_FEATURES
    assert len(schema_mod.FEATURES) == 7
    # DAT-002: the serving schema is the real prespecified contract.
    assert schema_mod.TARGET == features_mod.TARGET_COLUMN
    assert schema_mod.GROUP == features_mod.GROUP_COLUMN


def test_schema_document_is_json_round_trip_stable():
    doc = schema_mod.schema_document()
    # Round-trip through JSON without loss (the artifact stores exactly this).
    decoded = json.loads(json.dumps(doc))
    assert decoded == doc
    # Canonical encoding is independent of dict insertion order.
    reordered = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    assert schema_mod.canonical_json() == reordered


def test_schema_hash_is_canonical_and_sensitive():
    import curve_icon_dbso.schema as s

    expected = hashlib.sha256(s.canonical_json().encode("utf-8")).hexdigest()
    assert schema_mod.schema_hash() == expected
    assert len(schema_mod.schema_hash()) == 64

    # Any contract change must change the hash (that is what invalidates
    # existing artifacts, by design).
    original_limits = dict(schema_mod.FEATURE_LIMITS)
    original_version = schema_mod.SCHEMA_VERSION
    try:
        s.SCHEMA_VERSION = "9.9.9"
        assert schema_mod.schema_hash() != expected
        s.SCHEMA_VERSION = original_version
        s.FEATURE_LIMITS = {**original_limits, "age": (0.0, 150.0)}
        assert schema_mod.schema_hash() != expected
    finally:
        s.SCHEMA_VERSION = original_version
        s.FEATURE_LIMITS = original_limits


def test_schema_hash_is_distinct_from_the_evaluation_lineage_fingerprint():
    # Both exist on purpose: the Phase 1 evaluation fingerprint covers only
    # features/target/group; the serving hash covers the full schema document
    # (version, limits, batch bounds). They must not silently coincide.
    lineage = tracking.feature_schema_hash(
        schema_mod.FEATURES, schema_mod.TARGET, schema_mod.GROUP
    )
    assert lineage != schema_mod.schema_hash()
    assert len(lineage) == 64


def test_every_feature_has_a_documented_limit():
    assert set(schema_mod.FEATURE_LIMITS) == set(schema_mod.FEATURES)
    for name, (low, high) in schema_mod.FEATURE_LIMITS.items():
        assert low <= high, f"inverted limit for {name}"
    # Mathematically bounded asymmetry indices: [0, 1].
    for name in ("Brady_Asymmetry", "Rigidity_Asymmetry", "Tremor_Asymmetry"):
        assert schema_mod.FEATURE_LIMITS[name] == (0.0, 1.0)
    # HFO peak frequency must stay inside the extraction band.
    assert schema_mod.FEATURE_LIMITS["hfo_peak_freq_mean"] == (80.0, 500.0)


def test_validate_limits_rejects_drift(monkeypatch):
    schema_mod.validate_limits()  # healthy state passes
    monkeypatch.setattr(schema_mod, "FEATURE_LIMITS", {**schema_mod.FEATURE_LIMITS, "extra": (0, 1)})
    with pytest.raises(ValueError, match="must match FEATURES"):
        schema_mod.validate_limits()


def test_batch_bounds_match_the_api_contract():
    assert (schema_mod.BATCH_MIN_ROWS, schema_mod.BATCH_MAX_ROWS) == (1, 256)


# ------------------------------------------------------------------ models ---


def _valid_row() -> dict[str, float]:
    return {
        "hfo_power_mean": 1.2,
        "hfo_peak_freq_mean": 245.0,
        "Brady_Asymmetry": 0.25,
        "Rigidity_Asymmetry": 0.1,
        "Tremor_Asymmetry": 0.0,
        "years_since_diagnosis": 11.0,
        "age": 62.0,
    }


def test_row_model_accepts_valid_and_integer_values():
    from curve_icon_dbso.api.schemas import FeatureRow

    parsed = FeatureRow.model_validate(_valid_row())
    assert parsed.model_dump() == _valid_row()
    # Integers are valid JSON numbers for a float field: they parse to their
    # own float values (not silently rounded to some other valid row).
    ints = {k: int(v) for k, v in _valid_row().items()}
    assert FeatureRow.model_validate(ints).model_dump() == {k: float(v) for k, v in ints.items()}


def test_row_model_has_exactly_the_schema_fields():
    from curve_icon_dbso.api.schemas import FeatureRow

    assert list(FeatureRow.model_fields) == schema_mod.FEATURES


@pytest.mark.parametrize(
    "mutation,expected_class",
    [
        ({"age": "62"}, "invalid_type"),           # numeric string rejected
        ({"age": True}, "invalid_type"),           # bool is not a number
        ({"age": 150.0}, "out_of_range"),          # above the limit
        ({"age": 10.0}, "out_of_range"),           # below the limit
        ({"age": None}, "invalid_type"),           # null rejected: no imputation
        ({"subject": "X"}, "unknown_field"),       # identifiers rejected (API-006)
        ({"hfo_band_ratio": 1.0}, "unknown_field"),  # unknown feature rejected
    ],
)
def test_row_model_rejects_malformed_values(mutation, expected_class):
    from curve_icon_dbso.api.schemas import FeatureRow

    payload = {**_valid_row(), **mutation}
    with pytest.raises(ValidationError) as excinfo:
        FeatureRow.model_validate(payload)
    codes = {e["type"] for e in excinfo.value.errors()}
    expected_code = {
        "invalid_type": {"float_type", "float_parsing"},
        "out_of_range": {"greater_than_equal", "less_than_equal"},
        "unknown_field": {"extra_forbidden"},
    }[expected_class]
    assert codes & expected_code, f"expected {expected_code}, got {codes}"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_row_model_rejects_non_finite_values(bad):
    from curve_icon_dbso.api.schemas import FeatureRow

    payload = {**_valid_row(), "age": bad}
    with pytest.raises(ValidationError) as excinfo:
        FeatureRow.model_validate(payload)
    assert any(e["type"] == "finite_number" for e in excinfo.value.errors())


def test_request_model_enforces_batch_bounds():
    from curve_icon_dbso.api.payloads import synthetic_payload_rows
    from curve_icon_dbso.api.schemas import PredictRequest

    single = PredictRequest(rows=synthetic_payload_rows(1))
    assert len(single.rows) == 1  # minimum batch
    assert len(PredictRequest(rows=synthetic_payload_rows(256)).rows) == 256  # maximum
    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(rows=[])
    assert any(e["type"] == "too_short" for e in excinfo.value.errors())
    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(rows=synthetic_payload_rows(257))
    assert any(e["type"] == "too_long" for e in excinfo.value.errors())


def test_request_model_rejects_unknown_top_level_fields():
    from curve_icon_dbso.api.payloads import synthetic_payload_rows
    from curve_icon_dbso.api.schemas import PredictRequest

    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(rows=synthetic_payload_rows(1), subject_id="S01")
    assert any(e["type"] == "extra_forbidden" for e in excinfo.value.errors())

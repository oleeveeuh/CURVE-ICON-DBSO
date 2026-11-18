"""The deploy smoke-test fixture must be schema-valid synthetic content.

deploy.yml POSTs tests/fixtures/synthetic_predict_request.json through the
approved ingress as the deployment verification. This pins that fixture to
the canonical schema (DAT-001..003) and to synthetic provenance, so a
schema change or a non-synthetic payload cannot sneak into the deployment
smoke test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_predict_request.json"


def _rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())["rows"]


def test_fixture_parses_as_an_object_with_rows():
    data = json.loads(FIXTURE.read_text())
    assert set(data.keys()) == {"rows"}
    assert 1 <= len(data["rows"]) <= 256


def test_fixture_rows_are_schema_valid():
    from curve_icon_dbso.api.schemas import PredictRequest
    from curve_icon_dbso.schema import FEATURES

    request = PredictRequest.model_validate({"rows": _rows()})
    assert len(request.rows) == len(_rows())
    for row in request.rows:
        assert set(row.model_dump().keys()) == set(FEATURES)


def test_fixture_matches_the_deterministic_synthetic_generator():
    """Pin provenance while tolerating platform-level floating-point ULP noise."""
    from curve_icon_dbso.api.payloads import synthetic_payload_rows

    actual_rows = _rows()
    expected_rows = synthetic_payload_rows(4)
    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        assert actual.keys() == expected.keys()
        for name in actual:
            assert math.isclose(
                actual[name], expected[name], rel_tol=1e-12, abs_tol=1e-12
            ), (
                f"fixture field {name!r} diverged from the deterministic synthetic "
                "generator beyond platform-level floating-point noise"
            )

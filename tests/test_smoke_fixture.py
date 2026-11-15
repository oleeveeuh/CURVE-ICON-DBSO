"""The deploy smoke-test fixture must be schema-valid synthetic content.

deploy.yml POSTs tests/fixtures/synthetic_predict_request.json through the
approved ingress as the deployment verification. This pins that fixture to
the canonical schema (DAT-001..003) and to synthetic provenance, so a
schema change or a non-synthetic payload cannot sneak into the deployment
smoke test.
"""

from __future__ import annotations

import json
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
    """Provenance: the fixture must be byte-identical to what the public
    synthetic payload generator produces — no other origin is acceptable."""
    from curve_icon_dbso.api.payloads import synthetic_payload_rows

    expected = json.dumps(
        {"rows": synthetic_payload_rows(4)}, indent=2, sort_keys=True
    ) + "\n"
    assert FIXTURE.read_text() == expected, (
        "fixture diverged from synthetic_payload_rows(4); regenerate it from "
        "the synthetic generator (never hand-edit)"
    )

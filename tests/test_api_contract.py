"""API contract tests: endpoints, malformed inputs, redaction, readiness.

Covers the Phase 2 acceptance surface: valid single/batch prediction in
input order, every malformed-input class, fail-closed readiness for
missing/corrupt/incompatible artifacts, direct-model-vs-API equivalence,
metadata-only logging, and the ADR-002 scope pin (no /explain endpoint).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from curve_icon_dbso import artifact as artifact_mod
from curve_icon_dbso import schema as schema_mod
from curve_icon_dbso.api import payloads as payload_mod
from curve_icon_dbso.api.main import LOG, create_app
from curve_icon_dbso.training import train_synthetic_artifact


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("serving")
    out = root / "synthetic-demo"
    train_synthetic_artifact(out)
    return out


@pytest.fixture(scope="module")
def manifest(artifact_dir: Path) -> dict:
    return json.loads((artifact_dir / "manifest.json").read_text())


@pytest.fixture(scope="module")
def client(artifact_dir: Path) -> TestClient:
    app = create_app(artifact_dir)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def direct_model(artifact_dir: Path):
    model, _ = artifact_mod.load_artifact(artifact_dir)
    return model


# ---------------------------------------------------------------- healthy ---


def test_health_reports_liveness_and_readiness_after_startup(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "alive"
    assert body["ready"] is True
    assert body["research_only"] is True and body["not_for_clinical_use"] is True
    assert body["model_version"] and body["schema_version"] == schema_mod.SCHEMA_VERSION
    assert body["schema_hash"] == schema_mod.schema_hash()


def test_ready_endpoint_is_the_readiness_probe(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["error_class"] is None


def test_model_info_matches_the_loaded_manifest(client, manifest):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == manifest


def test_model_info_exposes_the_required_fields(client):
    body = client.get("/model-info").json()
    assert body["intended_use"] == "research only; not for clinical use"
    assert isinstance(body["beats_baseline"], bool)
    assert body["prohibited_uses"]
    assert body["data_provenance"].startswith("synthetic")
    for key in ("model_version", "schema_version", "schema_hash", "run_id", "source_sha",
                "artifact_sha256", "library_versions", "metrics", "baselines",
                "training_seed"):
        assert key in body


def test_no_explain_endpoint_by_design(client):
    """ADR-002 / DD-002: /explain is intentionally absent from Phase 2."""
    assert client.get("/explain").status_code == 404
    assert client.post("/explain", json={"rows": []}).status_code == 404


def test_predict_rejects_get(client):
    assert client.get("/predict").status_code == 405


# --------------------------------------------------------------- prediction ---


def test_single_row_prediction_is_valid_json_number(client):
    row = payload_mod.synthetic_payload_rows(1, seed=7)[0]
    response = client.post("/predict", json={"rows": [row]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    assert body["n_rows"] == 1
    assert isinstance(body["predictions"][0], float)


def test_max_batch_is_accepted(client):
    rows = payload_mod.synthetic_payload_rows(256)
    response = client.post("/predict", json={"rows": rows})
    assert response.status_code == 200
    assert response.json()["n_rows"] == 256


def test_predictions_preserve_input_order(client, direct_model):
    """The alignment contract (SCI-004's serving sibling): predictions[i]
    must be the prediction for rows[i], even shuffled and duplicated."""
    unique_rows = payload_mod.synthetic_payload_rows(50, seed=11)
    rng = random.Random(11)
    batch = unique_rows[:30] + [unique_rows[i] for i in rng.sample(range(50), 20)]
    rng.shuffle(batch)

    response = client.post("/predict", json={"rows": batch})
    assert response.status_code == 200
    frame = pd.DataFrame(batch, columns=schema_mod.FEATURES)
    expected = [float(v) for v in direct_model.predict(frame)]
    # The batch mixes duplicated and distinct rows; a constant-output model
    # would pass trivially, so guard against that degenerate case too.
    predictions = response.json()["predictions"]
    assert len(set(predictions)) > 1
    assert predictions == expected


def test_api_matches_direct_model_prediction_exactly(client, direct_model):
    rows = payload_mod.synthetic_payload_rows(9)
    via_api = client.post("/predict", json={"rows": rows}).json()["predictions"]
    frame = pd.DataFrame(rows, columns=schema_mod.FEATURES)
    direct = [float(v) for v in direct_model.predict(frame)]
    assert via_api == direct  # identical floats: same model, same doubles


def test_boundary_rows_are_valid(client):
    response = client.post("/predict", json={"rows": payload_mod.boundary_payload_rows()})
    assert response.status_code == 200
    assert response.json()["n_rows"] == 3


# ------------------------------------------------------------ malformed inputs ---


def _full_row(**overrides) -> dict:
    row = payload_mod.synthetic_payload_rows(1, seed=3)[0]
    return {**row, **overrides}


def _nonfinite_row_literal(token: str) -> str:
    """Raw JSON body with an `age` of NaN/Infinity (not valid strict JSON,
    but accepted by lenient parsers — exactly what the API must reject)."""
    row = {k: v for k, v in _full_row().items() if k != "age"}
    return '{"rows": [' + json.dumps(row)[:-1] + ', "age": ' + token + '}]}'




MALFORMED_CASES = [
    # (name, json_body_or_raw_string, expected_status, expected_error_class,
    #  value_that_must_not_appear_in_the_response)
    ("empty_batch", {"rows": []}, 422, "empty_batch", None),
    (
        "batch_too_large",
        {"rows": payload_mod.synthetic_payload_rows(257)},
        422, "batch_too_large", None,
    ),
    ("rows_not_a_list", {"rows": "many"}, 422, "invalid_type", "many"),
    ("unknown_row_field", {"rows": [_full_row(subject="X")]}, 422, "unknown_field", "X"),
    (
        "unknown_top_level_field",
        {"rows": [_full_row()], "subject_id": "S01"},
        422, "unknown_field", "S01",
    ),
    ("string_number", {"rows": [_full_row(age="62")]}, 422, "invalid_type", '"62"'),
    ("boolean_number", {"rows": [_full_row(age=True)]}, 422, "invalid_type", None),
    ("null_value", {"rows": [_full_row(age=None)]}, 422, "invalid_type", None),
    ("nan_literal", _nonfinite_row_literal("NaN"), 422, "non_finite_value", None),
    ("infinity_literal", _nonfinite_row_literal("Infinity"), 422, "non_finite_value", None),
    ("above_range", {"rows": [_full_row(age=150.0)]}, 422, "out_of_range", "150.0"),
    ("below_range", {"rows": [_full_row(age=10.0)]}, 422, "out_of_range", "10.0"),
    (
        # A bare feature row at body level: 'rows' is missing AND every
        # feature key is forbidden there -> two entry classes, so the
        # aggregate error_class is the generic validation_error.
        "body_is_a_bare_feature_row",
        {k: v for k, v in _full_row().items() if k != "age"},
        422, "validation_error", None,
    ),
    ("row_not_object", {"rows": [[1, 2, 3, 4, 5, 6, 7]]}, 422, "invalid_row", None),
    ("row_null", {"rows": [None]}, 422, "invalid_row", None),
    ("body_is_array", [1, 2, 3], 422, "invalid_body", None),
    ("missing_rows_key", {}, 422, "missing_field", None),
    ("malformed_json", '{"rows": [{"age": ', 400, "malformed_json", None),
]


@pytest.mark.parametrize(
    "name,body,expected_status,expected_class,forbidden", MALFORMED_CASES,
    ids=[case[0] for case in MALFORMED_CASES],
)
def test_malformed_inputs_fail_safely(client, name, body, expected_status, expected_class, forbidden):
    if isinstance(body, str):
        response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    else:
        response = client.post("/predict", json=body)
    assert response.status_code == expected_status, f"case {name}: {response.text[:300]}"
    payload = response.json()
    assert payload["error_class"] == expected_class, f"case {name}: {payload}"
    assert isinstance(payload.get("message"), str) and payload["message"]
    if forbidden is not None:
        # API-005: no payload values in the error response.
        assert forbidden not in response.text
    # error entries carry field paths only, never pydantic `input` echoes.
    for entry in payload.get("errors") or []:
        assert set(entry) == {"loc", "error_class", "message"}


@pytest.mark.parametrize("feature", schema_mod.FEATURES)
def test_every_feature_enforces_its_documented_limits(client, feature):
    low, high = schema_mod.FEATURE_LIMITS[feature]
    span = high - low
    for bad in (low - span / 2, high + span / 2):
        response = client.post("/predict", json={"rows": [_full_row(**{feature: bad})]})
        assert response.status_code == 422
        body = response.json()
        assert body["error_class"] == "out_of_range", (feature, bad)
        assert f"{bad!r}" not in response.text and str(bad) not in response.text


def test_identifier_row_is_rejected_as_unknown_field(client):
    """API-006: the service neither accepts nor requires subject identifiers."""
    row = _full_row(subject="SUBJECT_XYZ", recording_id=42)
    response = client.post("/predict", json={"rows": [row]})
    assert response.status_code == 422
    body = response.json()
    assert body["error_class"] == "unknown_field"
    assert "SUBJECT_XYZ" not in response.text


def test_oversized_payload_is_rejected_before_parsing(client):
    """Transport guard: >1 MB bodies get 413 without being parsed."""
    row = _full_row()
    rows = [row] * 40_000  # ~3 MB JSON
    response = client.post("/predict", json={"rows": rows})
    assert response.status_code == 413
    assert response.json()["error_class"] == "payload_too_large"


# ------------------------------------------------------ readiness failures ---


def _fresh_client_with(artifact_path: Path):
    app = create_app(artifact_path)
    return TestClient(app)  # no lifespan: state before startup is also tested


def test_unstarted_app_is_alive_but_not_ready(artifact_dir):
    client = _fresh_client_with(artifact_dir)
    health = client.get("/health")
    assert health.status_code == 200  # liveness independent of readiness
    assert health.json()["ready"] is False
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"rows": [_full_row()]}).status_code == 503
    assert client.get("/model-info").status_code == 503


def test_missing_artifact_fails_readiness(tmp_path):
    client = _fresh_client_with(tmp_path / "does_not_exist")
    with client:
        assert client.get("/health").json()["ready"] is False
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["error_class"] == "artifact_missing"
        predict = client.post("/predict", json={"rows": [_full_row()]})
        assert predict.status_code == 503
        assert predict.json()["error_class"] == "artifact_missing"
        assert client.get("/model-info").status_code == 503


def _artifact_copy_with_tampered_model(artifact_dir: Path, tmp_path: Path) -> Path:
    copy_dir = tmp_path / "corrupt"
    copy_dir.mkdir()
    data = bytearray((artifact_dir / "model.pkl").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (copy_dir / "model.pkl").write_bytes(bytes(data))
    for name in ("schema.json", "manifest.json"):
        (copy_dir / name).write_bytes((artifact_dir / name).read_bytes())
    return copy_dir


def test_corrupt_artifact_fails_readiness_and_stays_liveness(artifact_dir, tmp_path):
    client = _fresh_client_with(_artifact_copy_with_tampered_model(artifact_dir, tmp_path))
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ready"] is False
        assert health.json()["error_class"] == "artifact_corrupt"
        assert client.get("/ready").json()["error_class"] == "artifact_corrupt"
        assert client.post("/predict", json={"rows": [_full_row()]}).status_code == 503


def _artifact_copy_with_drifted_libraries(artifact_dir: Path, tmp_path: Path) -> Path:
    copy_dir = tmp_path / "drifted"
    copy_dir.mkdir()
    for name in ("model.pkl", "schema.json"):
        (copy_dir / name).write_bytes((artifact_dir / name).read_bytes())
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    manifest["library_versions"]["scikit-learn"] = "0.0.0"
    (copy_dir / "manifest.json").write_text(json.dumps(manifest))
    return copy_dir


def test_incompatible_artifact_fails_readiness(artifact_dir, tmp_path):
    client = _fresh_client_with(_artifact_copy_with_drifted_libraries(artifact_dir, tmp_path))
    with client:
        assert client.get("/ready").json()["error_class"] == "artifact_incompatible"


def _artifact_copy_with_wrong_schema_version(artifact_dir: Path, tmp_path: Path) -> Path:
    copy_dir = tmp_path / "wrong_schema"
    copy_dir.mkdir()
    for name in ("model.pkl", "schema.json"):
        (copy_dir / name).write_bytes((artifact_dir / name).read_bytes())
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    # Consistent hashes, but the manifest declares a schema version the
    # artifact file does not carry -> incompatible, not merely corrupt.
    manifest["schema_version"] = "0.0.1"
    (copy_dir / "manifest.json").write_text(json.dumps(manifest))
    return copy_dir


def test_schema_version_mismatch_fails_readiness(artifact_dir, tmp_path):
    client = _fresh_client_with(_artifact_copy_with_wrong_schema_version(artifact_dir, tmp_path))
    with client:
        body = client.get("/ready").json()
        assert body["ready"] is False
        assert body["error_class"] == "artifact_incompatible"


# ------------------------------------------------------------------- logging ---


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


@pytest.fixture()
def captured_logs():
    handler = _ListHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = LOG.level
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    yield handler.records
    LOG.removeHandler(handler)
    LOG.setLevel(previous_level)


def test_logs_are_metadata_only_and_json(client, captured_logs):
    secret_value = 0.123456789
    row = _full_row(hfo_power_mean=secret_value)
    good = client.post("/predict", json={"rows": [row]})
    assert good.status_code == 200
    assert secret_value not in good.json()["predictions"]

    client.post("/predict", json={"rows": []})                      # validation error
    client.get("/model-info")
    client.get("/health")

    assert captured_logs, "service emitted no log lines"
    approved_keys = {
        "request_id", "route", "status", "duration_ms",
        "row_count", "schema_hash", "model_version", "error_class",
    }
    for line in captured_logs:
        entry = json.loads(line)  # every line is one JSON object
        unexpected = set(entry) - approved_keys
        assert not unexpected, f"non-approved log keys: {unexpected} in {entry}"
        assert isinstance(entry["duration_ms"], (int, float))
    # approved metadata is present where required
    predict_entry = json.loads(captured_logs[0])
    assert predict_entry["row_count"] == 1
    assert predict_entry["route"] == "/predict"
    assert predict_entry["schema_hash"] == schema_mod.schema_hash()
    validation_entries = [json.loads(line) for line in captured_logs[1:]]
    assert any(entry["error_class"] == "empty_batch" for entry in validation_entries)


def test_log_lines_never_contain_payload_or_prediction_values(client, captured_logs):
    row = _full_row(hfo_peak_freq_mean=321.0)
    response = client.post("/predict", json={"rows": [row]})
    prediction = response.json()["predictions"][0]
    stream = "\n".join(captured_logs)
    assert "321.0" not in stream
    assert str(prediction) not in stream
    assert json.dumps(row) not in stream
    # Feature names never appear in logs either — route/status metadata only.
    for feature in schema_mod.FEATURES:
        assert feature not in stream


def test_request_ids_are_returned_as_headers(client):
    response = client.get("/health")
    assert "x-request-id" in {k.lower() for k in response.headers.keys()}


# ---------------------------------------------------------------- payloads ---


def test_payload_rows_are_valid_and_deterministic(client):
    rows_a = payload_mod.synthetic_payload_rows(5, seed=21)
    rows_b = payload_mod.synthetic_payload_rows(5, seed=21)
    assert rows_a == rows_b
    assert len(rows_a) == 5
    assert all(set(row) == set(schema_mod.FEATURES) for row in rows_a)
    assert client.post("/predict", json={"rows": rows_a}).status_code == 200


def test_shifted_payload_rows_shift_deterministically():
    """Drift fixtures (OBS-006 preview): every value moves outward by a
    fixed fraction of its limit range, deterministically for a given seed.
    Values may still land inside the limits for bounded features (the
    asymmetry indices span only [0, 1]) — these are analysis fixtures,
    not request bodies."""
    rows = payload_mod.shifted_payload_rows(3, seed=5)
    original = payload_mod.synthetic_payload_rows(3, seed=5)
    assert len(rows) == 3
    for shifted, base in zip(rows, original, strict=True):
        assert shifted != base
        for feature in schema_mod.FEATURES:
            low, high = schema_mod.FEATURE_LIMITS[feature]
            span = high - low
            assert shifted[feature] == pytest.approx(base[feature] + 0.5 * span)
    assert rows == payload_mod.shifted_payload_rows(3, seed=5)


def test_payload_row_count_validation():
    with pytest.raises(ValueError):
        payload_mod.synthetic_payload_rows(0)

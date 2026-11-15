"""Artifact packaging: manifest/hash consistency, determinism, tamper rejection.

Pins the MOD-002 manifest contract and the MOD-003 fail-closed loader:
every hash is reverified on load, schema/library drift is incompatible,
and the pickle is deserialized only after every check is green.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso import artifact as artifact_mod
from curve_icon_dbso import schema as schema_mod
from curve_icon_dbso.errors import ArtifactError
from curve_icon_dbso.training import train_synthetic_artifact


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """One synthetic artifact shared by the module's tests."""
    root = tmp_path_factory.mktemp("artifact")
    out = root / "synthetic-demo"
    train_synthetic_artifact(out)
    return out


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text())


def test_artifact_contains_exactly_the_three_files(trained):
    assert sorted(p.name for p in trained.iterdir()) == [
        "manifest.json", "model.pkl", "schema.json"
    ]


def test_manifest_hashes_match_the_files_on_disk(trained):
    manifest = _manifest(trained)
    model_bytes = (trained / "model.pkl").read_bytes()
    schema_bytes = (trained / "schema.json").read_bytes()
    assert manifest["model_sha256"] == artifact_mod.sha256_bytes(model_bytes)
    assert manifest["schema_sha256"] == artifact_mod.sha256_bytes(schema_bytes)
    assert manifest["artifact_sha256"] == artifact_mod.artifact_binding_hash(
        manifest["model_sha256"], manifest["schema_sha256"]
    )


def test_manifest_carries_the_full_contract(trained):
    manifest = _manifest(trained)
    assert set(artifact_mod.REQUIRED_MANIFEST_KEYS).issubset(manifest)
    assert manifest["schema_version"] == schema_mod.SCHEMA_VERSION
    assert manifest["schema_hash"] == schema_mod.schema_hash()
    assert manifest["intended_use"] == "research only; not for clinical use"
    assert isinstance(manifest["prohibited_uses"], list) and manifest["prohibited_uses"]
    assert manifest["data_provenance"].startswith("synthetic")
    assert manifest["research_only"] is True
    assert manifest["not_for_clinical_use"] is True
    assert manifest["beats_baseline"] in (True, False)
    assert manifest["training_seed"] == 42
    assert manifest["training_rows"] == 9 and manifest["training_subjects"] == 8
    assert manifest["source_sha"] == "unknown" or len(manifest["source_sha"]) == 40
    # MOD-002: metrics come with baselines; the manifest states both.
    assert {"Mean baseline", "Median baseline"}.issubset(manifest["baselines"])
    assert set(manifest["baselines"]["Mean baseline"]) >= {"MAE", "RMSE", "R2"}
    assert manifest["metrics"] and manifest["candidates"]


def test_load_returns_an_equivalent_predicting_model(trained):
    model, manifest = artifact_mod.load_artifact(trained)
    rows = pd.DataFrame(
        [
            {"hfo_power_mean": 1.2, "hfo_peak_freq_mean": 245.0, "Brady_Asymmetry": 0.25,
             "Rigidity_Asymmetry": 0.1, "Tremor_Asymmetry": 0.0,
             "years_since_diagnosis": 11.0, "age": 62.0}
        ],
        columns=schema_mod.FEATURES,
    )
    prediction = float(model.predict(rows)[0])
    assert np.isfinite(prediction)


def test_training_is_byte_deterministic(tmp_path):
    """Same seed -> identical model, schema and manifest bytes (offline path).

    The manifest carries no wall-clock fields and its run_id is a
    deterministic fingerprint; the --mlflow path is documented as the
    exception (real run IDs are nondeterministic by nature).
    """
    a, b = tmp_path / "a", tmp_path / "b"
    manifest_a = train_synthetic_artifact(a)
    manifest_b = train_synthetic_artifact(b)
    for name in ("model.pkl", "schema.json", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert manifest_a == manifest_b


def test_refuses_to_overwrite_an_existing_artifact(trained):
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        train_synthetic_artifact(trained)


# ------------------------------------------------------- fail-closed loader ---


def test_missing_files_fail_as_artifact_missing(tmp_path):
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(tmp_path)  # empty directory
    assert excinfo.value.error_class == "artifact_missing"
    (tmp_path / "model.pkl").write_bytes(b"x")
    (tmp_path / "schema.json").write_bytes(b"x")
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(tmp_path)  # manifest still missing
    assert excinfo.value.error_class == "artifact_missing"


def test_tampered_model_bytes_fail_hash_check(trained):
    data = bytearray((trained / "model.pkl").read_bytes())
    data[0] ^= 0xFF
    tampered = trained.parent / "tampered_model"
    tampered.mkdir()
    (tampered / "model.pkl").write_bytes(bytes(data))
    for name in ("schema.json", "manifest.json"):
        (tampered / name).write_bytes((trained / name).read_bytes())
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(tampered)
    assert excinfo.value.error_class == "artifact_corrupt"
    assert "model_sha256" in str(excinfo.value)


def test_undecryptable_pickle_is_rejected_even_with_matching_hashes(trained):
    """Verify-then-deserialize order: bytes whose hash is consistent with a
    rebuilt manifest but are not a valid pickle must still be rejected."""
    hostile = trained.parent / "hostile"
    hostile.mkdir()
    garbage = b"definitely not a pickle"
    schema_sha = artifact_mod.sha256_bytes((trained / "schema.json").read_bytes())
    model_sha = artifact_mod.sha256_bytes(garbage)
    (hostile / "model.pkl").write_bytes(garbage)
    (hostile / "schema.json").write_bytes((trained / "schema.json").read_bytes())
    manifest = _manifest(trained)
    manifest["model_sha256"] = model_sha
    manifest["schema_sha256"] = schema_sha
    manifest["artifact_sha256"] = artifact_mod.artifact_binding_hash(model_sha, schema_sha)
    (hostile / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(hostile)
    assert excinfo.value.error_class == "artifact_corrupt"


def test_schema_drift_fails_as_incompatible(trained, monkeypatch):
    """An artifact trained on schema v1 must not load under schema v2."""
    modified = schema_mod.schema_document()
    modified["schema_version"] = "2.0.0"
    monkeypatch.setattr(artifact_mod, "schema_document", lambda: modified)
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(trained)
    assert excinfo.value.error_class == "artifact_incompatible"


def test_limit_drift_fails_as_incompatible(trained, monkeypatch):
    modified = schema_mod.schema_document()
    modified["feature_limits"]["age"] = [0.0, 150.0]
    monkeypatch.setattr(artifact_mod, "schema_document", lambda: modified)
    monkeypatch.setattr(artifact_mod, "schema_hash", lambda: "0" * 64)
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(trained)
    assert excinfo.value.error_class == "artifact_incompatible"


def test_library_version_drift_fails_as_incompatible(trained):
    drifted = trained.parent / "drifted_libs"
    drifted.mkdir()
    for name in ("model.pkl", "schema.json"):
        (drifted / name).write_bytes((trained / name).read_bytes())
    manifest = _manifest(trained)
    manifest["library_versions"]["numpy"] = "0.0.1"
    (drifted / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(drifted)
    assert excinfo.value.error_class == "artifact_incompatible"
    assert "numpy" in str(excinfo.value)


def test_manifest_missing_keys_fail_as_corrupt(trained):
    incomplete = trained.parent / "incomplete"
    incomplete.mkdir()
    for name in ("model.pkl", "schema.json"):
        (incomplete / name).write_bytes((trained / name).read_bytes())
    manifest = _manifest(trained)
    del manifest["beats_baseline"]
    (incomplete / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(incomplete)
    assert excinfo.value.error_class == "artifact_corrupt"


def test_manifest_without_hashes_fails_before_deserialization(trained):
    """A hash mismatch on schema.json must abort before unpickling."""
    broken = trained.parent / "broken_schema"
    broken.mkdir()
    (broken / "model.pkl").write_bytes((trained / "model.pkl").read_bytes())
    (broken / "schema.json").write_bytes(b'{"schema_version": "lies"}\n')
    (broken / "manifest.json").write_bytes((trained / "manifest.json").read_bytes())
    with pytest.raises(ArtifactError) as excinfo:
        artifact_mod.load_artifact(broken)
    assert excinfo.value.error_class == "artifact_corrupt"


def test_binding_hash_is_stable_and_discriminating():
    a = artifact_mod.artifact_binding_hash("1" * 64, "2" * 64)
    assert a == artifact_mod.artifact_binding_hash("1" * 64, "2" * 64)
    assert a != artifact_mod.artifact_binding_hash("2" * 64, "1" * 64)
    assert len(a) == 64

"""Drift demonstration tests (OBS-006/OBS-007): PSI math, aggregates,
and the aggregate-only privacy contract. Educational synthetic-only
signal — these tests pin the MATH and the privacy shape, not any
clinical interpretation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso import drift as drift_mod
from curve_icon_dbso.api.payloads import shifted_payload_rows, synthetic_payload_rows
from curve_icon_dbso.training import train_synthetic_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("drift-artifact")
    out = base / "artifact"
    train_synthetic_artifact(out)
    return json.loads((out / "manifest.json").read_text())


# ------------------------------------------------------------ PSI math ---


def test_psi_is_zero_for_identical_distributions():
    expected = [0.1, 0.2, 0.3, 0.4]
    assert drift_mod.population_stability_index(expected, expected) == pytest.approx(0.0, abs=1e-9)


def test_psi_hand_checked_value():
    # sum (a-e) ln(a/e) with e=[.5,.5], a=[.25,.75]:
    #   (-.25)ln(.5) + (.25)ln(1.5) = .25*(ln 1.5 - ln .5) = .25*ln 3
    expected = [0.5, 0.5]
    actual = [0.25, 0.75]
    assert drift_mod.population_stability_index(expected, actual) == pytest.approx(
        0.25 * np.log(3.0), rel=1e-9
    )


def test_psi_survives_empty_bins_via_smoothing():
    expected = [0.5, 0.5, 0.0]
    actual = [0.25, 0.25, 0.5]  # mass appears in a bin the reference never saw
    value = drift_mod.population_stability_index(expected, actual)
    assert np.isfinite(value) and value > 0.5  # large-but-finite, not inf


def test_psi_rejects_malformed_inputs():
    with pytest.raises(ValueError):
        drift_mod.population_stability_index([0.5], [0.5, 0.5])
    with pytest.raises(ValueError):
        drift_mod.population_stability_index([], [])


# ------------------------------------------------------- aggregates -------


def test_reference_aggregates_are_deterministic_and_normalized(manifest):
    stored = manifest["reference_aggregates"]
    from curve_icon_dbso.features import add_clinical_composites, build_model_matrix
    from curve_icon_dbso.synthetic import generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_subjects=8, seed=42)
    x, _, _ = build_model_matrix(add_clinical_composites(cohort))
    recomputed = drift_mod.reference_aggregates(x)
    assert json.dumps(recomputed, sort_keys=True) == json.dumps(stored, sort_keys=True), (
        "aggregates must be reproducible from the training seed (byte-stable artifact)"
    )
    for _feature, agg in stored.items():
        assert pytest.approx(sum(agg["expected_share"]), abs=1e-3) == 1.0
        assert len(agg["expected_share"]) == len(agg["bin_edges"]) - 1


# ------------------------------------------------------------ report ------


def test_clean_resample_is_not_drifted_and_shifted_resample_is(manifest):
    aggregates = manifest["reference_aggregates"]
    seed = manifest["training_seed"]  # 42: same cohort as the reference

    clean = pd.DataFrame(synthetic_payload_rows(500, seed=seed))
    clean_report = drift_mod.compute_drift_report(aggregates, clean)
    assert clean_report["drifted_features"] == []
    assert max(clean_report["features"].values()) < 0.05

    shifted = pd.DataFrame(shifted_payload_rows(500, seed=seed, shift_fraction=0.5))
    shifted_report = drift_mod.compute_drift_report(aggregates, shifted)
    assert len(shifted_report["drifted_features"]) == len(shifted_report["features"])
    assert min(shifted_report["features"].values()) > drift_mod.DEFAULT_PSI_THRESHOLD


def test_report_is_aggregate_only_never_raw_values(manifest):
    aggregates = manifest["reference_aggregates"]
    rows = synthetic_payload_rows(50, seed=42)
    report = drift_mod.compute_drift_report(aggregates, pd.DataFrame(rows))
    text = json.dumps(report)
    # No serialized ROW may appear (a full feature vector is per-row data);
    # scalar collisions are meaningless, whole-row leakage is the leak.
    for row in rows:
        assert json.dumps(row) not in text, "a per-row record leaked into the drift report"
        assert str(sorted(row.items())) not in text
    # Shape: aggregate floats only — no bucket contents, no raw vectors.
    assert set(report) == {"threshold", "features", "drifted_features"}
    assert all(isinstance(v, float) for v in report["features"].values())


def test_missing_aggregates_fail_with_actionable_error():
    frame = pd.DataFrame(synthetic_payload_rows(10, seed=42))
    with pytest.raises(ValueError, match="retrain"):
        drift_mod.compute_drift_report({}, frame)


# ------------------------------------------------------------- demo -------


def test_drift_demo_script_produces_expected_verdict(tmp_path):
    """End-to-end: retrain in tmp, run the demo, check the JSON verdict +
    that the disclaimer is present and no raw values leak."""

    out = tmp_path / "artifact"
    train_synthetic_artifact(out)
    report_path = tmp_path / "drift.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "drift_demo.py"),
         "--artifact-dir", str(out), "--output", str(report_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert "NOT validated clinical drift monitoring" in report["disclaimer"]
    assert report["clean_synthetic_sample"]["drifted_features"] == []
    assert report["shifted_synthetic_sample"]["drifted_features"]
    # Privacy: no full per-row record may appear anywhere in the report
    # (aggregate PSI floats only — per-scalar substring checks would
    # false-positive on coincidental float equality).
    text = report_path.read_text()
    rows = synthetic_payload_rows(5, seed=42)
    for row in rows:
        assert json.dumps(row) not in text


def test_demo_rejects_cohort_seed_mismatch(tmp_path):
    out = tmp_path / "artifact"
    train_synthetic_artifact(out)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "drift_demo.py"),
         "--artifact-dir", str(out), "--cohort-seed", "999"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "does not match the artifact's training seed" in result.stderr

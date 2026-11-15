"""Experiment lineage: provenance fingerprints, lazy MLflow, local logging."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from curve_icon_dbso import tracking

mlflow = pytest.importorskip("mlflow", reason="optional 'tracking' dependency group")

from curve_icon_dbso import features as features_mod  # noqa: E402
from curve_icon_dbso.evaluation import (  # noqa: E402
    baseline_models,
    default_models,
    run_group_cv,
)
from curve_icon_dbso.synthetic import generate_synthetic_cohort  # noqa: E402


def test_feature_schema_hash_is_stable_and_sensitive():
    a = tracking.feature_schema_hash(["f1", "f2"], "t", "g")
    assert a == tracking.feature_schema_hash(["f1", "f2"], "t", "g")
    assert a != tracking.feature_schema_hash(["f2", "f1"], "t", "g")
    assert a != tracking.feature_schema_hash(["f1", "f2"], "other", "g")
    assert len(a) == 64


def test_outcome_and_kinematic_schema_hashes_differ():
    assert tracking.outcome_schema_hash() != tracking.kinematic_schema_hash()


def test_git_sha_is_a_fingerprint_or_honest_unknown(tmp_path):
    sha = tracking.git_sha()
    assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))
    # Outside any repository the fingerprint must say unknown, not guess.
    assert tracking.git_sha(tmp_path) == "unknown"


def test_lock_hash_matches_file_content():
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    expected = tracking.file_sha256(lock)
    assert tracking.lock_hash() == expected


def test_stage_validation_is_strict():
    assert tracking.validate_stage("exploratory") == "exploratory"
    assert tracking.validate_stage("confirmatory") == "confirmatory"
    with pytest.raises(Exception, match="analysis stage"):
        tracking.validate_stage("pilot")


def test_core_package_import_does_not_pull_mlflow():
    """MOD-004: MLflow stays optional — importing tracking must not import it."""
    code = (
        "import sys; import curve_icon_dbso.tracking; "
        "assert 'mlflow' not in sys.modules, 'tracking.py eagerly imported mlflow'; "
        "print('lazy-ok')"
    )
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(repo_root), check=True,
    )
    assert "lazy-ok" in result.stdout


def test_log_evaluation_run_requires_mlflow(monkeypatch):
    """With the dependency blocked, the call fails with install guidance."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match="tracking"):
        tracking.log_evaluation_run(
            run_name="x", config={}, model_params={}, aggregate_metrics={},
            provenance="synthetic", seed=1, stage="exploratory",
            git_sha_value="unknown", lock_hash_value="absent",
            schema_hash_value="0" * 64, target="t",
        )


def test_full_run_logs_complete_lineage(tmp_path):
    """End-to-end: run the synthetic demo CV and log it; assert every
    REP-005/MOD-004 field is present and readable from the store."""
    cohort = generate_synthetic_cohort()
    enriched = features_mod.add_clinical_composites(cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    models = {**baseline_models(), **default_models()}
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)

    run_id = tracking.log_evaluation_run(
        run_name="lineage-test",
        config={"n_subjects": 8, "splitter": "leave-one-subject-out"},
        model_params={"Random Forest": {"n_estimators": 200}},
        aggregate_metrics={
            r.model_name: {k: float(v) for k, v in r.metrics.items()} for r in results
        },
        fold_metrics={
            r.model_name: tracking.fold_metrics_from_result(r) for r in results
        },
        bootstrap_summary={
            "Random Forest": {"MAE": {"metric": "MAE", "point": 1.0, "ci_low": 0.5,
                                      "ci_high": 1.5, "level": 0.95, "n_resamples": 10,
                                      "seed": 1, "n_subjects": 8}}
        },
        permutation_summary={
            "Random Forest": {"MAE": {"metric": "MAE", "direction": "lower",
                                      "observed": 1.0, "p_value": 0.5,
                                      "null_mean": 2.0, "null_median": 2.0,
                                      "n_permutations": 10, "seed": 1}}
        },
        provenance="synthetic — lineage test",
        seed=42,
        stage="exploratory",
        git_sha_value="0" * 40,
        lock_hash_value="1" * 64,
        schema_hash_value="2" * 64,
        target="updrs_improvement",
        artifacts={},
        tracking_uri=f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}",
    )
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    run = client.get_run(run_id)
    params, tags = run.data.params, run.data.tags
    assert params["git_sha"] == "0" * 40
    assert params["lock_hash"] == "1" * 64
    assert params["feature_schema_hash"] == "2" * 64
    assert params["analysis_stage"] == "exploratory"
    assert params["target"] == "updrs_improvement"
    assert params["model.Random_Forest.n_estimators"] == "200"
    assert any(key.startswith("Random_Forest/") for key in run.data.metrics)
    assert any("fold" in key for key in run.data.metrics)
    assert any("bootstrap_MAE_ci_low" in key for key in run.data.metrics)
    assert any("permutation_MAE_p" in key for key in run.data.metrics)
    assert tags["research_only"] == "true"
    assert tags["not_for_clinical_use"] == "true"


def test_demo_mlflow_path_accepts_result_objects(tmp_path, monkeypatch):
    """Regression: the demo logs BootstrapCI / PermutationMetricResult objects,
    not plain dicts — logging must coerce them (this broke once)."""
    monkeypatch.chdir(tmp_path)  # default store lands in the tmp cwd, not the repo
    from curve_icon_dbso import demo as demo_mod

    exit_code = demo_mod.main([
        "--output-dir", str(tmp_path / "out"), "--seed", "1",
        "--permutations", "3", "--bootstrap-resamples", "50", "--mlflow",
    ])
    assert exit_code == 0
    store = tmp_path / "mlruns" / "mlflow.db"
    assert store.exists()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{store}")
    runs = client.search_runs(experiment_ids=["0"])
    assert runs and runs[0].info.status == "FINISHED"
    assert any("bootstrap_MAE_ci_low" in k for k in runs[0].data.metrics)
    assert any("permutation_MAE_p" in k for k in runs[0].data.metrics)

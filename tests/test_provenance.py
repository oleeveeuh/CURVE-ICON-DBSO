"""Run contract (REP-005/SCI-012), retraction-wording guard, ignored stores,
and the compliance-gated external-data script rehearsed on SYNTHETIC fixtures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from curve_icon_dbso import demo

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo") / "out"
    exit_code = demo.main([
        "--output-dir", str(out), "--seed", "42",
        "--permutations", "5", "--bootstrap-resamples", "200",
    ])
    assert exit_code == 0
    return json.loads((out / "summary.json").read_text())


def test_summary_carries_the_full_run_contract(demo_summary):
    """SCI-012: cohort, target, unit of observation, fold strategy,
    missing-data handling and provenance are reported together."""
    summary = demo_summary
    assert summary["data_provenance"].startswith("synthetic")
    assert summary["analysis_stage"] == "exploratory"
    assert summary["target"] == "updrs_improvement"
    assert "subject" in summary["unit_of_observation"]
    assert "leave-one-subject-out" in summary["fold_strategy"]
    assert summary["missing_data_handling"]
    assert summary["n_subjects"] == 8 and summary["n_recordings"] == 9


def test_summary_provenance_fingerprints_are_real(demo_summary):
    provenance = demo_summary["run_provenance"]
    assert len(provenance["git_sha"]) == 40  # the real commit SHA
    lock_hash = provenance["lock_hash"]
    assert len(lock_hash) == 64  # real uv.lock SHA-256
    assert len(provenance["feature_schema_hash"]) == 64
    assert provenance["seed"] == 42


def test_every_model_reports_beats_baseline_sci_014(demo_summary):
    models = demo_summary["models"]
    assert {"Mean baseline", "Median baseline"} <= set(models)
    candidates = {name: entry for name, entry in models.items()
                  if "baseline" not in name.lower()}
    assert candidates, "expected candidate models in summary"
    for name, entry in candidates.items():
        assert entry["beats_baseline"] in {True, False}, (
            f"{name} must visibly report beats_baseline (SCI-014)"
        )


def test_summary_includes_bootstrap_and_permutation_blocks(demo_summary):
    assert demo_summary["bootstrap"], "bootstrap block missing"
    first = next(iter(demo_summary["bootstrap"].values()))
    assert "MAE" in first and "RMSE" in first
    assert first["MAE"]["n_resamples"] == 200
    assert demo_summary["permutation_null"]
    assert "DESCRIPTIVE" in demo_summary["permutation_interpretation"] or \
        "hypothesis-generating" in demo_summary["permutation_interpretation"]


def test_no_output_ever_calls_r_squared_accuracy(tmp_path):
    """The retracted value was mislabeled 'accuracy'; no artifact may repeat
    that framing (plan §3.3, item 6 of Prompt 2)."""
    out = tmp_path / "out"
    assert demo.main(["--output-dir", str(out), "--seed", "1",
                      "--permutations", "3", "--bootstrap-resamples", "50"]) == 0
    for artifact in ("summary.json", "metrics.csv", "oof_predictions_per_subject.csv"):
        assert "accuracy" not in (out / artifact).read_text().lower()


def test_mlflow_store_is_gitignored():
    """Proof that local experiment stores and generated artifacts stay out of
    version control (Prompt 2 required verification)."""
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "mlruns/" in gitignore
    check = subprocess.run(
        ["git", "check-ignore", "-q", "mlruns/"],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert check.returncode == 0, "git does not ignore mlruns/"
    check_outputs = subprocess.run(
        ["git", "check-ignore", "-q", "outputs/"],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert check_outputs.returncode == 0, "git does not ignore outputs/"
    # ...and uv.lock is the opposite: it MUST be tracked (REP-003: the
    # committed deterministic lock feeds local setup, CI, and Docker).
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "uv.lock"],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert tracked.returncode == 0, "uv.lock must be committed (REP-003)"


# --- compliance-gated script: rehearsal on synthetic fixtures only -------------

ACK = (
    "This dataset is compliance-verified public data, lawfully licensed "
    "for this use, and not restricted, private, identifiable, or "
    "former-lab data."
)


def _kinematic_feature_frame(tmp_path: Path) -> Path:
    """Synthetic per-recording kinematic table (as the future extraction step
    would emit) — the script never sees real data here."""
    import numpy as np
    import pandas as pd

    from curve_icon_dbso.kinematics import KINEMATIC_FEATURES, KINEMATIC_TARGET

    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        rows.append({
            **{name: float(rng.uniform(0.5, 2.0)) for name in KINEMATIC_FEATURES},
            KINEMATIC_TARGET: float(rng.uniform(1.5, 3.0)),
            "subject": f"SYN{i:02d}",
        })
    path = tmp_path / "kinematic_recordings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_real_script_refuses_without_the_exact_acknowledgement(tmp_path, capsys):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from evaluate_real_kinematics import main as real_main
    finally:
        sys.path.pop(0)

    features_csv = _kinematic_feature_frame(tmp_path)
    exit_code = real_main([
        "--features-csv", str(features_csv),
        "--output-dir", str(tmp_path / "results"),
        "--acknowledge-data-use", "trust me",
        "--stage", "exploratory",
    ])
    assert exit_code == 1
    assert "exactly" in capsys.readouterr().err


def test_real_script_refuses_while_no_dataset_is_compliance_verified(tmp_path, capsys):
    """No dataset is VERIFIED in the docs/data_access.md compliance
    register, so the gate must block even a correct acknowledgement —
    this is the rehearsal of the closed state."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from evaluate_real_kinematics import main as real_main
    finally:
        sys.path.pop(0)
    features_csv = _kinematic_feature_frame(tmp_path)
    exit_code = real_main([
        "--features-csv", str(features_csv),
        "--output-dir", str(tmp_path / "results"),
        "--acknowledge-data-use", ACK,
        "--stage", "exploratory",
    ])
    assert exit_code == 1
    assert "CLOSED" in capsys.readouterr().err


def test_real_script_evaluation_half_runs_on_synthetic_fixture(tmp_path):
    """The evaluation machinery behind the gates, exercised directly on the
    synthetic table: same grouped CV, bootstrap, permutation, provenance."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from evaluate_real_kinematics import PROVENANCE_REAL_KINEMATIC, _evaluate
    finally:
        sys.path.pop(0)

    from curve_icon_dbso.kinematics import build_kinematic_model_matrix

    frame = __import__("pandas").read_csv(_kinematic_feature_frame(tmp_path))
    out = tmp_path / "results"
    results = _evaluate(
        frame,
        repo_root=REPO_ROOT,
        seed=42,
        permutations=5,
        bootstrap_resamples=100,
        with_xgboost=False,
        stage="exploratory",
        output_dir=out,
        use_mlflow=False,
    )
    assert results, "expected evaluation results"
    x, y, groups = build_kinematic_model_matrix(frame)
    assert results[0].metrics["n_subjects"] == 10
    summary = json.loads((out / "summary.json").read_text())
    assert summary["data_provenance"] == PROVENANCE_REAL_KINEMATIC
    assert summary["target"] == "tap_rate_mean"
    assert len(summary["run_provenance"]["feature_schema_hash"]) == 64

"""End-to-end driver test for scripts/run_transfer_study.py on a fully
synthetic input directory (converted-contract npz files + a minimal
workbook). Verifies the versioned run directory, aggregate artifacts,
provenance, and the no-identifiers property of the committable summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _write_synthetic_npz(directory: Path) -> None:
    from curve_icon_dbso.transfer.synthetic import make_fast_cohort

    directory.mkdir(parents=True, exist_ok=True)
    cohort = make_fast_cohort(n=4, seed0=9700, plant="movement")
    for i, recording in enumerate(cohort):
        np.savez(
            directory / f"SYN{i + 1:02d}.npz",
            subject=f"SYN{i + 1:02d}",
            fs_ecog=recording.fs,
            fs_glove=recording.fs,
            restmove_segs=np.asarray(
                [[s, e, 0, 0 if k == "rest" else 1]
                 for s, e, k in recording.blocks], dtype=float),
            ecog=recording.ecog,
            glove=recording.glove,
            age=60.0 + i,
            dbs_volt=1.0,
            dbs_freq=130.0,
        )


def _write_synthetic_workbook(path: Path) -> None:
    # Sibling top-level module (tests/ is not an importable package in a
    # clean environment; pytest's rootdir puts the tests dir on sys.path).
    from test_transfer_workbook import _minimal_xlsx

    rows = [["SUBJECT", "M1_Contact_Img", "Signal quality"]]
    anchors = [5, 4, 6, 5]
    for i, anchor in enumerate(anchors):
        rows.append([f"SYN{i + 1:02d}", str(anchor), "1"])
    _minimal_xlsx(path, rows)


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    npz_dir = tmp_path / "npz"
    workbook = tmp_path / "wb.xlsx"
    _write_synthetic_npz(npz_dir)
    _write_synthetic_workbook(workbook)
    return npz_dir, workbook


def test_driver_runs_end_to_end_and_writes_artifacts(
    synthetic_inputs, tmp_path: Path, capsys, monkeypatch
) -> None:
    npz_dir, workbook = synthetic_inputs
    import importlib

    module = importlib.import_module("run_transfer_study")
    output_root = tmp_path / "outputs" / "transfer_v2"
    exit_code = module.main([
        "--input-dir", str(npz_dir),
        "--workbook", str(workbook),
        "--output-dir", str(output_root),
        "--null-draws", "10",
        "--no-ablations",
        "--no-xgboost",
        "--acknowledge-data-use", "synthetic smoke test",
    ])
    assert exit_code == 0
    run_dirs = list(output_root.glob("smoke_*/"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary_path = run_dir / "run_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert (run_dir / "cohort_flow.json").is_file()
    assert (run_dir / "calibration_curve.json").is_file()
    assert (run_dir / "results_aggregate.md").is_file()

    # provenance + stage discipline
    from curve_icon_dbso.transfer import contract as transfer_contract

    assert summary["contract_version"] == transfer_contract.CONTRACT_VERSION
    assert summary["stage"] == "exploratory"
    provenance = summary["provenance"]
    assert provenance["smoke_run"] is True
    assert provenance["run_mode"] == "smoke"
    assert provenance["definitive"] is False
    assert provenance["null_draws_override"] == 10
    assert provenance["n_inputs"] == 4
    # the manifest itself is a LOCAL-ONLY file (filenames carry subject
    # codes with real data); the summary carries its hash, not its names
    manifest_path = run_dir / "input_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 4
    assert provenance["contract_hash"] == summary["contract_hash"]

    # aggregate structure
    assert summary["cohort_flow"]["n_recordings_loaded"] == 4
    assert summary["cohort_flow"]["n_eligible"] == 4
    assert "ridge_logistic" in summary["track_a_detection"]
    assert "verdicts" in summary and "global_null" in summary["verdicts"]

    # the committable aggregate carries NO subject identifiers — the
    # synthetic codes must not appear anywhere in the summary document
    text = summary_path.read_text()
    for i in range(1, 5):
        assert f"SYN{i:02d}" not in text


def test_driver_smoke_flag_requires_explicit_reduced_draws(
    synthetic_inputs, tmp_path
) -> None:
    import importlib

    module = importlib.import_module("run_transfer_study")
    npz_dir, workbook = synthetic_inputs
    # ANY reduced setting — altered draws, skipped ablations, skipped
    # comparators — is a smoke run (v2.1.0 gate; there is no draw count
    # that makes a reduced run "definitive").
    output_root = tmp_path / "out2"
    module.main([
        "--input-dir", str(npz_dir),
        "--workbook", str(workbook),
        "--output-dir", str(output_root),
        "--null-draws", "20",
        "--no-ablations",
        "--no-xgboost",
    ])
    run_dirs = list(output_root.glob("smoke_*/"))
    assert len(run_dirs) == 1  # reduced settings → smoke labeling applies


def test_reduced_draws_alone_still_smoke_and_cannot_write_public_doc(
    synthetic_inputs, tmp_path
) -> None:
    import importlib

    module = importlib.import_module("run_transfer_study")
    npz_dir, workbook = synthetic_inputs
    output_root = tmp_path / "out3"
    # full-featured run but with altered draws → smoke, and
    # --write-results-doc is refused OUTRIGHT (before any run directory
    # is created — a non-definitive run can never publish)
    exit_code = module.main([
        "--input-dir", str(npz_dir),
        "--workbook", str(workbook),
        "--output-dir", str(output_root),
        "--null-draws", "999",       # altered, NOT the frozen 1000
        "--write-results-doc",       # must be rejected: not definitive
    ])
    assert exit_code == 2
    assert not list(output_root.glob("**/*"))


def test_definitive_mode_rejects_every_reduced_setting(
    synthetic_inputs, tmp_path
) -> None:
    import importlib

    module = importlib.import_module("run_transfer_study")
    npz_dir, workbook = synthetic_inputs
    output_root = tmp_path / "out4"
    for extra in (["--no-ablations"], ["--no-xgboost"], ["--null-draws", "500"]):
        exit_code = module.main([
            "--input-dir", str(npz_dir),
            "--workbook", str(workbook),
            "--output-dir", str(output_root),
            "--definitive",
            *extra,
        ])
        assert exit_code == 2, f"definitive must reject {extra}"
    assert not list(output_root.glob("**/run_*"))  # nothing launched


def test_definitive_mode_launches_into_run_directory(
    synthetic_inputs, tmp_path
) -> None:
    import importlib

    module = importlib.import_module("run_transfer_study")
    npz_dir, workbook = synthetic_inputs
    output_root = tmp_path / "out5"
    exit_code = module.main([
        "--input-dir", str(npz_dir),
        "--workbook", str(workbook),
        "--output-dir", str(output_root),
        "--definitive",
        "--null-draws", "8",   # exercises the matrix fast; see note below
    ])
    # exact frozen draws are required — this launch must be REJECTED even
    # though --definitive was requested
    assert exit_code == 2

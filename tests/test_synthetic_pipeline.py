"""End-to-end synthetic pipeline: deterministic and offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from curve_icon_dbso import demo


def _run_into(tmp_path: Path, seed: int = 42) -> Path:
    output_dir = tmp_path / f"out_seed{seed}"
    exit_code = demo.main(["--output-dir", str(output_dir), "--seed", str(seed)])
    assert exit_code == 0
    return output_dir


@pytest.fixture(scope="module")
def demo_outputs(tmp_path_factory) -> Path:
    return _run_into(tmp_path_factory.mktemp("demo"))


def test_demo_writes_expected_outputs(demo_outputs: Path):
    for name in ("metrics.csv", "oof_predictions_per_subject.csv", "summary.json"):
        assert (demo_outputs / name).exists(), f"missing output {name}"


def test_demo_is_deterministic(tmp_path: Path, demo_outputs: Path):
    """Same seed twice → byte-identical metrics and summary. (The second
    run reuses the module-scoped first run's outputs.)"""
    repeat = _run_into(tmp_path, seed=42)
    for name in ("metrics.csv", "summary.json"):
        assert (demo_outputs / name).read_bytes() == (repeat / name).read_bytes(), (
            f"{name} differs between identical runs — randomness is unseeded somewhere"
        )


def test_demo_summary_carries_provenance_and_warning(demo_outputs: Path):
    import json

    summary = json.loads((demo_outputs / "summary.json").read_text())
    assert summary["data_provenance"].startswith("synthetic")
    assert summary["n_subjects"] == 8
    assert summary["n_recordings"] == 9
    assert "n=8" in summary["small_sample_warning"]
    # Baselines must be part of every run so models can be compared to them.
    model_names = set(summary["models"])
    assert {"Mean baseline", "Median baseline"} <= model_names


def test_demo_cli_error_paths(tmp_path: Path, capsys):
    exit_code = demo.main(["--n-subjects", "1", "--output-dir", str(tmp_path / "x")])
    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().err

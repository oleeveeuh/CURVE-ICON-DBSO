"""The kinematic extraction half, rehearsed end to end on synthetic fixtures.

Covers the neutral per-recording .npz contract of
``scripts/build_kinematic_table.py`` (extraction -> QC-filtered feature
table) and proves the emitted table feeds ``evaluate_real_kinematics``
unchanged. Synthetic signals only — the script never sees real data here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.errors import CurveIconError

REPO_ROOT = Path(__file__).resolve().parents[1]

FS_ECOG = 1200.0
FS_GLOVE = 100.0


def _write_recording(path: Path, *, subject: str, with_move: bool = True) -> None:
    """One deterministic synthetic recording in the documented .npz contract.

    REST block: 200 Hz ECoG sine (inside the 80–500 Hz HFO band). MOVE
    block: two glove channels of a 2 Hz tap sinusoid — both channel-blocks
    dominate the in-band PSD, so the 0.20 prominence floor passes and the
    >= 2 channel-block QC rule is satisfied. ``with_move=False`` yields a
    QC failure (no MOVE blocks).
    """
    seconds = 4.0
    n = int(seconds * FS_GLOVE)
    t_glove = np.arange(n) / FS_GLOVE
    # ECoG shares the restmove_segs sample grid (contract: arrays are
    # stored on a common time base); its own rate only affects the
    # HFO band math.
    t_ecog = np.arange(n) / FS_ECOG

    segs = [[0, n // 2, 0, 0]]  # REST: first half
    if with_move:
        segs.append([n // 2, n, 0, 1])  # MOVE: second half
    ecog = np.tile((np.sin(2 * np.pi * 200.0 * t_ecog) + 0.1).reshape(1, -1), (2, 1))
    tap = np.sin(2 * np.pi * 2.0 * t_glove) if with_move else np.zeros(n)
    glove = np.tile((tap + 0.05).reshape(1, -1), (2, 1))

    np.savez(
        path,
        subject=subject,
        fs_ecog=FS_ECOG,
        fs_glove=FS_GLOVE,
        restmove_segs=np.asarray(segs, dtype=float),
        ecog=ecog,
        glove=glove,
        age=61.0,
        dbs_volt=3.0,
        dbs_freq=130.0,
    )


@pytest.fixture(scope="module")
def recordings_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("recordings")
    _write_recording(directory / "rec01.npz", subject="SYN01")
    _write_recording(directory / "rec02.npz", subject="SYN02")
    _write_recording(directory / "rec03.npz", subject="SYN03", with_move=False)
    return directory


def _main():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import build_kinematic_table
    finally:
        sys.path.pop(0)
    return build_kinematic_table


def test_build_table_extracts_and_filters_qc(recordings_dir, capsys):
    builder = _main()
    table = builder.build_table(recordings_dir)
    assert list(table["subject"]) == ["SYN01", "SYN02"]
    assert (table["tap_qc_status"] == "ok").all()
    assert table["tap_rate_mean"].between(1.5, 2.5).all(), (
        "2 Hz synthetic taps must be recovered within the 1 Hz PSD resolution"
    )
    assert table["hfo_power_rest_mean"].notna().all()
    # The QC failure is reported explicitly, never silently dropped.
    captured = capsys.readouterr()
    assert "EXCLUDED" in captured.out and "rec03.npz" in captured.out
    assert "no MOVE blocks" in captured.out


def test_build_table_rejects_missing_dir_and_keys(tmp_path):
    builder = _main()
    with pytest.raises(CurveIconError, match="Input directory not found"):
        builder.build_table(tmp_path / "does-not-exist")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CurveIconError, match="No \\*.npz recordings"):
        builder.build_table(empty)
    bad = tmp_path / "bad"
    bad.mkdir()
    np.savez(bad / "incomplete.npz", subject="SYN09")
    with pytest.raises(CurveIconError, match="missing required keys"):
        builder.build_table(bad)


def test_script_main_runs_and_records_reminder(recordings_dir, tmp_path, capsys):
    main = _main().main
    output_csv = tmp_path / "kinematic_recordings.csv"
    exit_code = main([
        "--input-dir", str(recordings_dir),
        "--output-csv", str(output_csv),
    ])
    assert exit_code == 0
    assert "REMINDER" in capsys.readouterr().err
    assert output_csv.exists()


def test_emitted_table_feeds_the_evaluation_half(recordings_dir, tmp_path):
    """Full-chain rehearsal: .npz recordings -> feature table -> grouped CV
    with baselines, bootstrap, provenance, and the run contract."""
    builder = _main()
    table = builder.build_table(recordings_dir)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from evaluate_real_kinematics import PROVENANCE_REAL_KINEMATIC, _evaluate
    finally:
        sys.path.pop(0)

    out = tmp_path / "results"
    results = _evaluate(
        table,
        repo_root=REPO_ROOT,
        seed=42,
        permutations=0,
        bootstrap_resamples=50,
        with_xgboost=False,
        stage="exploratory",
        output_dir=out,
        use_mlflow=False,
    )
    assert results, "expected evaluation results from the extracted table"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["data_provenance"] == PROVENANCE_REAL_KINEMATIC
    assert summary["target"] == "tap_rate_mean"
    metrics_frame = pd.read_csv(out / "metrics.csv")
    assert len(metrics_frame) == len(results)

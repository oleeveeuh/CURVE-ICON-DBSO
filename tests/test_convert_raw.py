"""The raw-layout converter, rehearsed on a synthetic clone of the format.

Builds a miniature stand-in for the authorized local dataset — same
FieldTrip-style struct layout, channel naming convention (RDBS*/P*/aux),
glove sensors, restmove_segs, and localization workbook columns — with
NO real subject codes or data. Proves conversion -> the .npz contract,
explicit exclusions, and covariate extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from curve_icon_dbso.errors import CurveIconError

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 1200.0
N_SAMPLES = 4000


def _ft_struct(labels: list[str], fs: float, signals: np.ndarray):
    dtype = np.dtype([("hdr", "O"), ("fsample", "O"), ("sampleinfo", "O"),
                      ("trial", "O"), ("time", "O"), ("label", "O"), ("cfg", "O")])
    struct = np.zeros((1, 1), dtype=dtype)
    struct[0, 0]["fsample"] = fs
    struct[0, 0]["label"] = np.array([[name] for name in labels], dtype=object)
    struct[0, 0]["trial"] = signals
    return struct


def _write_raw(root: Path, subject: str, *, with_move: bool = True) -> None:
    t = np.arange(N_SAMPLES) / FS
    ecog = np.vstack([
        np.sin(2 * np.pi * 200.0 * t) + 0.05,   # P0: HFO-band tone
        np.sin(2 * np.pi * 120.0 * t) + 0.05,   # P1: HFO-band tone
        np.sin(2 * np.pi * 40.0 * t),           # RDBS0: not an ECoG strip channel
    ])
    glove = np.vstack([
        np.sin(2 * np.pi * 2.0 * t) + 0.05,
        np.sin(2 * np.pi * 2.0 * t + 0.1) + 0.05,
        np.sin(2 * np.pi * 2.0 * t + 0.2) + 0.05,
        np.sin(2 * np.pi * 2.0 * t + 0.3) + 0.05,
        np.sin(2 * np.pi * 2.0 * t + 0.4) + 0.05,
    ])
    segs = [[0, N_SAMPLES // 2, 0, 0]]
    if with_move:
        segs.append([N_SAMPLES // 2, N_SAMPLES, 0, 1])
    subject_dir = root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        subject_dir / "filtered_downsampled_Data_FT_format.mat",
        {
            "data": _ft_struct(["RDBS0", "P0", "P1"], FS, ecog),
            "data_glove": _ft_struct(
                [f"Glove1Sensor0{i}" for i in range(1, 6)], FS, glove
            ),
            "restmove_segs": np.asarray(segs, dtype=float),
        },
    )


def _write_workbook(path: Path) -> None:
    frame = pd.DataFrame({
        "SUBJECT": ["SUB_A", "SUB_B"],
        "Age": [61.0, np.nan],
        "FREQ STIM": [130.0, 130.0],
        "VOLT STIM": [3.0, 3.0],
    })
    frame.to_excel(path, sheet_name="Sheet1", index=False)


@pytest.fixture(scope="module")
def raw_tree(tmp_path_factory):
    pytest.importorskip("openpyxl")
    root = tmp_path_factory.mktemp("raw")
    _write_raw(root, "SUB_A")
    _write_raw(root, "SUB_B", with_move=False)
    (root / "SUB_C").mkdir()  # no .mat at all
    workbook = root / "ECoG_localization.xlsx"
    _write_workbook(workbook)
    return root, workbook


def _converter():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import convert_raw_to_npz
    finally:
        sys.path.pop(0)
    return convert_raw_to_npz


def test_convert_writes_npz_and_excludes_explicitly(raw_tree, tmp_path, capsys):
    converter = _converter()
    raw_root, workbook = raw_tree
    out = tmp_path / "kinematic_npz"
    summary = converter.convert(raw_root, workbook, out)

    assert list(summary["subject"]) == ["SUB_A"], (
        "SUB_B (covariate missing) and SUB_C (no .mat) are excluded"
    )
    captured = capsys.readouterr()
    assert "EXCLUDED: SUB_B" in captured.out and "missing/NaN" in captured.out
    assert "EXCLUDED: SUB_C" in captured.out and "no '" in captured.out

    with np.load(out / "SUB_A.npz", allow_pickle=False) as data:
        assert str(data["subject"]) == "SUB_A"
        assert data["ecog"].shape == (2, N_SAMPLES)  # only the P* strip
        assert data["glove"].shape == (5, N_SAMPLES)
        assert float(data["fs_ecog"]) == FS and float(data["fs_glove"]) == FS
        assert float(data["age"]) == 61.0
        assert float(data["dbs_volt"]) == 3.0
        assert float(data["dbs_freq"]) == 130.0
        assert data["restmove_segs"].shape == (2, 4)


def test_convert_rejects_missing_inputs(tmp_path):
    converter = _converter()
    with pytest.raises(CurveIconError, match="Raw directory not found"):
        converter.convert(tmp_path / "nope", tmp_path / "wb.xlsx", tmp_path / "out")
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(CurveIconError, match="workbook not found"):
        converter.convert(raw, tmp_path / "wb.xlsx", tmp_path / "out")


def test_convert_emits_tables_the_builder_consumes(raw_tree, tmp_path, capsys):
    """Chain rehearsal: converter -> builder contract on converted files."""
    converter = _converter()
    raw_root, workbook = raw_tree
    out = tmp_path / "npz"
    converter.convert(raw_root, workbook, out)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from build_kinematic_table import build_table
    finally:
        sys.path.pop(0)
    table = build_table(out)
    assert list(table["subject"]) == ["SUB_A"]
    assert table["tap_qc_status"].iloc[0] == "ok"
    assert table["tap_rate_mean"].between(1.5, 2.5).all()

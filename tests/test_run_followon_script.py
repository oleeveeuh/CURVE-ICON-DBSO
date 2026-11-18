"""End-to-end synthetic verification of the follow-on subject-modeling
command (scripts/run_followon_subject_modeling.py): a tiny synthetic
.npz cohort goes through extraction → glove targets → subject
aggregation → frozen arm machinery, plus the versioned-output guarantee.
The real-data command itself is a local-only owner action and is never
executed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.subject_modeling import promotion_table, run_arm

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_npz_cohort(tmp_path, n_subjects: int = 6) -> None:
    """Write a tiny synthetic cohort in the documented per-recording .npz
    contract: REST(2 s) + MOVE(2 s) + REST(2 s) + MOVE(2 s), glove taps at
    a subject-specific rate."""
    fs_ecog, fs_glove = 2000.0, 500.0
    for subject in range(n_subjects):
        n = int(8 * fs_ecog)
        rng = np.random.default_rng(subject)
        t = np.arange(n) / fs_ecog
        # REST: 1/f-ish background; MOVE: same + a 70 Hz state component
        # with a subject-specific amplitude (a cross-subject REST signal).
        background = rng.normal(0, 2.0, n)
        rest_signal = background
        move_signal = background + (2.0 + 0.5 * subject) * np.sin(
            2 * np.pi * 70.0 * t
        )
        ecog = np.vstack([rest_signal, rest_signal, move_signal, move_signal])
        ecog = ecog + rng.normal(0, 0.2, ecog.shape)
        # Glove: two channels tapping at the subject's rate during MOVE.
        # Glove taps at the glove rate, then zero-order-hold onto the
        # shared restmove_segs sample grid (the .npz interchange contract:
        # both arrays share the block grid — bounds are sample-wise).
        ng = int(8 * fs_glove)
        tg = np.arange(ng) / fs_glove
        rate = 1.5 + 0.1 * subject
        taps = np.where((tg % (1 / rate)) < 0.5 / rate,
                        np.sin(2 * np.pi * rate * tg), 0.0)
        hold = n // ng
        taps_grid = np.repeat(taps, hold)[:n]
        glove = (np.vstack([taps_grid, -taps_grid])
                 + 0.01 * rng.standard_normal((2, n)))
        segs = np.array([
            [0, 2 * fs_ecog, 0, 0],
            [2 * fs_ecog, 4 * fs_ecog, 0, 1],
            [4 * fs_ecog, 6 * fs_ecog, 0, 0],
            [6 * fs_ecog, 8 * fs_ecog, 0, 1],
        ], dtype=float)
        np.savez_compressed(
            tmp_path / f"rec_{subject:02d}.npz",
            subject=f"synthetic_rec_{subject:02d}",
            fs_ecog=fs_ecog,
            fs_glove=fs_glove,
            restmove_segs=segs,
            ecog=ecog,
            glove=glove,
            age=50.0 + subject,
            dbs_volt=2.5,
            dbs_freq=100.0,
        )


def _load_module():
    """Load the script as a module (it lives in scripts/, not the package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_followon_subject_modeling",
        str(REPO_ROOT / "scripts" / "run_followon_subject_modeling.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_cohort_flows_through_the_whole_chain(tmp_path):
    _synthetic_npz_cohort(tmp_path)
    module = _load_module()
    frame, qc = module.load_and_extract(tmp_path)
    assert frame.shape[0] == 6
    # Frozen block columns resolved directly from the aggregated frame.
    subjects = module.aggregate_to_subjects(frame)
    assert subjects.shape[0] == 6
    assert "tap_rate_mean" in subjects.columns
    assert subjects["tap_rate_mean"].notna().all()
    arm = run_arm(subjects, "tap_rate_mean", "subject",
                  blocks=("clinical", "rest_spectral", "delta_spectral"),
                  model_names=("dummy_mean", "dummy_median", "ridge_primary"),
                  seed=3)
    for block in ("clinical", "rest_spectral", "delta_spectral"):
        assert len(arm["blocks"][block]) == 3
        for result in arm["blocks"][block]:
            assert np.isfinite(result.aggregate["MAE"])
    table = promotion_table(arm["blocks"]["delta_spectral"],
                            arm["reference_aggregates"])
    assert {"model", "block", "promotes"} <= set(table.columns)


def test_versioned_output_dir_never_overwrites(tmp_path):
    module = _load_module()
    first = module._versioned_output_dir(tmp_path)
    first.mkdir(parents=True)
    second = module._versioned_output_dir(tmp_path)
    second.mkdir(parents=True)
    third = module._versioned_output_dir(tmp_path)
    assert first != second and second != third
    assert not third.exists()


def test_script_refuses_missing_inputs(tmp_path):
    module = _load_module()
    with pytest.raises(Exception, match="input directory not found"):
        module.load_and_extract(tmp_path / "nowhere")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="no .npz recordings"):
        module.load_and_extract(empty)


def test_input_manifest_records_verifiable_sha256_digests(tmp_path):
    """The identical-inputs guarantee between the v1.0.0 and v1.0.1 runs
    rests on these digests — they must be real SHA-256 over the file
    bytes, not asserted prose."""
    import hashlib

    _synthetic_npz_cohort(tmp_path, n_subjects=3)
    module = _load_module()
    manifest = module._input_manifest(tmp_path)
    assert set(manifest) == {f"rec_{i:02d}.npz" for i in range(3)}
    for name, record in manifest.items():
        data = (tmp_path / name).read_bytes()
        assert record["bytes"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    with pytest.raises(Exception, match="no .npz recordings"):
        module._input_manifest(tmp_path / "nowhere")


def test_contract_version_recorded_in_the_script_source():

    source = (REPO_ROOT / "scripts"
              / "run_followon_subject_modeling.py").read_text()
    assert "protocol_hash()" in source and "PROTOCOL_VERSION" in source
    loaded = _load_module()
    assert loaded.PROTOCOL_VERSION == protocol.PROTOCOL_VERSION

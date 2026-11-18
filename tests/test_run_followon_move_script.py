"""End-to-end synthetic verification of the follow-on MOVE command
(scripts/run_followon_move_modeling.py): a tiny synthetic .npz cohort
flows through window building → decoding → null → plots → sanitized
versioned outputs. The real-data command is a local-only owner action
and is never executed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from curve_icon_dbso import protocol

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_followon_move_modeling",
        str(REPO_ROOT / "scripts" / "run_followon_move_modeling.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_npz_cohort(tmp_path: Path, n: int = 3) -> None:
    fs = 1000.0
    for subject in range(n):
        rng = np.random.default_rng(subject)
        total = int(8 * fs)
        segs = np.array([
            [0, int(1 * fs), 0, 0],
            [int(1 * fs), int(7 * fs), 0, 1],
            [int(7 * fs), int(8 * fs), 0, 0],
        ], dtype=float)
        tg = np.arange(total) / fs
        rate = 1.5 + 0.1 * subject
        taps = np.where((tg % (1 / rate)) < 0.5 / rate,
                        np.sin(2 * np.pi * rate * tg), 0.0)
        glove = np.vstack([taps, -taps]) + 0.01 * rng.standard_normal((2, total))
        move = np.zeros(total)
        in_move = slice(int(1 * fs), int(7 * fs))
        move[in_move] = (2.0 + subject) * np.sin(
            2 * np.pi * 70.0 * tg[in_move]
        )
        ecog = (np.vstack([rng.normal(0, 0.5, total) + move * g
                           for g in (0.9, 1.0, 1.1)]))
        np.savez_compressed(
            tmp_path / f"move_rec_{subject:02d}.npz",
            subject=f"synthetic_move_rec_{subject:02d}",
            fs_ecog=fs,
            fs_glove=fs,
            restmove_segs=segs,
            ecog=ecog,
            glove=glove,
        )


def test_synthetic_cohort_flows_through_the_move_chain(tmp_path):
    _synthetic_npz_cohort(tmp_path)
    module = _load_module()
    datasets, qc = module.load_window_datasets(tmp_path)
    assert len(datasets) == 3
    # 6 s MOVE @ 100 ms updates = 60 bins; the first bins (no full
    # in-block window) and the last bins (no far-lag context inside the
    # block) are dropped by the frozen causal rules.
    assert all(40 <= d.features.shape[0] <= 60 for d in datasets)
    assert all(d.qc['dropped_windows'] >= 5 for d in datasets)
    dataset = module.merge_datasets(datasets)
    # Subject positions, never participant codes.
    assert set(dataset.groups.tolist()) == {"pos000", "pos001", "pos002"}
    assert "synthetic" not in " ".join(dataset.groups.tolist())
    results = module.run_move_cv(dataset, model_names=("lagged_ridge_primary",))
    assert results[0].summary["n_subjects"] == 3
    null = module.move_circular_shift_null(dataset, n_shifts=2, seed=1)
    assert null["n_shifts"] == 2
    plots = module.plot_move_results(
        results, null=null, out_dir=tmp_path / "plots",
        lag_ablation={"full": results[0]},
        block_ablation={"multiband_power": results[0]},
    )
    assert Path(plots["observed_vs_null"]).exists()


def test_versioned_output_dir_never_overwrites(tmp_path):
    module = _load_module()
    first = module._versioned_output_dir(tmp_path)
    first.mkdir(parents=True)
    second = module._versioned_output_dir(tmp_path)
    second.mkdir(parents=True)
    third = module._versioned_output_dir(tmp_path)
    assert first != second != third
    assert not third.exists()


def test_script_refuses_missing_inputs(tmp_path):
    module = _load_module()
    with pytest.raises(Exception, match="input directory not found"):
        module.load_window_datasets(tmp_path / "nowhere")


def test_input_manifest_records_verifiable_sha256_digests(tmp_path):
    import hashlib

    _synthetic_npz_cohort(tmp_path, n=2)
    module = _load_module()
    manifest = module._input_manifest(tmp_path)
    assert set(manifest) == {"move_rec_00.npz", "move_rec_01.npz"}
    for name, record in manifest.items():
        data = (tmp_path / name).read_bytes()
        assert record["bytes"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    with pytest.raises(Exception, match="no .npz recordings"):
        module._input_manifest(tmp_path / "nowhere")


def test_dmd_comparison_runs_end_to_end(tmp_path):
    _synthetic_npz_cohort(tmp_path, n=3)
    module = _load_module()
    datasets, _qc = module.load_window_datasets(tmp_path, capture_windows=True)
    dataset = module.merge_datasets(datasets)
    merged_windows: list = []
    for per_recording in datasets:
        merged_windows.extend(per_recording.windows or [])
    dmd_frame, diagnostics = module.dmd_feature_frame(merged_windows)
    assert diagnostics["n_windows"] == dataset.features.shape[0]
    assert diagnostics["per_window_seconds"] > 0
    comparison = module.run_representation_comparison(
        dataset, dmd_frame, model_name="lagged_ridge_primary",
    )
    assert set(comparison) == {"multiband_power", "dmd_nodes",
                               "combined_power_plus_dmd_nodes"}
    for _representation, result in comparison.items():
        assert result.summary["n_windows"] == dataset.features.shape[0]


def test_script_records_the_frozen_contract():
    module = _load_module()
    assert module.PROTOCOL_VERSION == protocol.PROTOCOL_VERSION
    source = (REPO_ROOT / "scripts"
              / "run_followon_move_modeling.py").read_text()
    assert "protocol_hash()" in source
    assert "never overwrites" in source

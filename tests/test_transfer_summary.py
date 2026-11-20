"""Public-artifact privacy tests (post-definitive-run review, 2026-11-20):
the committable summary and public report must never carry per-subject,
held-out, or ordinal-position FIELDS — a recursive oracle rejects them
wherever they hide."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from curve_icon_dbso.transfer.pipeline import (  # noqa: E402
    RegressionArtifacts,
    StudyArtifacts,
)
from curve_icon_dbso.transfer.summary import (  # noqa: E402
    _contains_local_only_key,
    build_run_summary,
    strip_local_only,
)


def test_strip_local_only_removes_planted_fields_at_any_depth() -> None:
    planted = {
        "ablations": {
            "regression_window_1s": {
                "median_delta_mae": 0.0,
                "per_subject_delta_mae": {"0": 0.0, "1": -0.01},
            },
            "contrast_family": {
                "members": {"x": {"n_pairs": 31, "held_out_positions": [0, 1]}},
            },
        },
        "cohort_flow": {
            "n_eligible": 31,
            "eligible_ordinal_positions": [1, 2],
            "exclusions_by_reason": {"x": [3]},
        },
        "nested": [{"per_subject_r": [0.1]}],
    }
    cleaned = strip_local_only(planted)
    oracle = _contains_local_only_key(cleaned)
    assert oracle == [], f"local-only fields survived: {oracle}"
    # aggregates pass through untouched
    assert cleaned["ablations"]["regression_window_1s"]["median_delta_mae"] == 0.0
    assert cleaned["cohort_flow"]["n_eligible"] == 31


def test_oracle_finds_local_only_fields() -> None:
    assert _contains_local_only_key({"per_subject_delta_mae": {}}) != []
    assert _contains_local_only_key({"held_out_position": 0}) != []
    assert _contains_local_only_key({"a": {"b": [{"ordinal_positions": []}]}}) != []
    assert _contains_local_only_key({"median_delta_mae": 0.0, "n": 31}) == []


def test_frozen_statistic_name_allowlisted_but_payloads_still_scanned() -> None:
    """``per_subject_spread`` is a frozen secondary-family NAME whose
    payload is a cohort aggregate — allowlisted. But a nested per-subject
    map under it is still caught (the scan recurses through allowlisted
    keys)."""
    clean = {"per_subject_spread": {"observed": 0.4, "null_median": 0.0,
                                    "p_raw": 0.3}}
    assert _contains_local_only_key(clean) == []
    dirty = {"per_subject_spread": {"per_subject_values": [0.1, 0.2]}}
    assert _contains_local_only_key(dirty) != []


def test_public_summary_of_a_planted_artifact_tree_is_clean() -> None:
    artifacts = StudyArtifacts(
        cohort_flow={"n_eligible": 3, "eligible_ordinal_positions": [1, 2, 3]},
        track_a={"results": {}},
        tracks_bc=RegressionArtifacts(
            fewshot={"fewshot_30": {"per_subject_delta_mae": {"0": 0.0}}},
        ),
        ablations={"regression_window_1s": {
            "median_delta_mae": 0.0,
            "per_subject_delta_mae": {"0": 0.01},
        }},
    )
    summary = build_run_summary(artifacts)
    assert _contains_local_only_key(summary) == []


def test_real_run_report_and_summary_carry_no_subject_fields(
    tmp_path,
) -> None:
    """End-to-end: a driver run on synthetic inputs must produce a public
    summary and report with zero local-only fields anywhere (the
    definitive-run review found keyed per-subject delta-MAE maps in
    exactly these artifacts)."""
    import importlib

    from test_run_transfer_script import (
        _write_synthetic_npz,
        _write_synthetic_workbook,
    )

    module = importlib.import_module("run_transfer_study")
    npz_dir = tmp_path / "npz"
    workbook = tmp_path / "wb.xlsx"
    _write_synthetic_npz(npz_dir)
    _write_synthetic_workbook(workbook)
    output_root = tmp_path / "out_privacy"
    exit_code = module.main([
        "--input-dir", str(npz_dir),
        "--workbook", str(workbook),
        "--output-dir", str(output_root),
        "--null-draws", "8",
        "--no-xgboost",          # ablations ON: they carried the leak
    ])
    assert exit_code == 0
    run_dir = next(output_root.glob("smoke_*/"))
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert _contains_local_only_key(summary) == []
    doc_text = (run_dir / "results_aggregate.md").read_text()
    assert "per_subject" not in doc_text
    assert "held_out" not in doc_text

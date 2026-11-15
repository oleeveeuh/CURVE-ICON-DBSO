"""Optional XGBoost parity exercise (DD-003): fixed params, same machinery,
never a default, and clean skips when the optional group is absent."""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("xgboost", reason="optional 'analysis' dependency group")

from curve_icon_dbso import demo
from curve_icon_dbso import features as features_mod  # noqa: E402
from curve_icon_dbso.evaluation import (  # noqa: E402
    XGB_FIXED_PARAMS,
    baseline_models,
    default_models,
    run_group_cv,
    xgboost_model,
)
from curve_icon_dbso.synthetic import generate_synthetic_cohort  # noqa: E402


def test_xgboost_is_not_a_default_model():
    assert "XGBoost" not in default_models()
    assert "XGBoost" not in baseline_models()


def test_xgboost_configuration_is_fixed_and_documented():
    params = xgboost_model().get_params()
    for key, value in XGB_FIXED_PARAMS.items():
        assert params[key] == value, f"fixed parameter drifted: {key}"
    assert params["random_state"] == XGB_FIXED_PARAMS["random_state"]


def test_xgboost_runs_the_identical_grouped_cv_machinery():
    cohort = generate_synthetic_cohort()
    enriched = features_mod.add_clinical_composites(cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    models = {**baseline_models(), "XGBoost": xgboost_model()}
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)
    by_name = {r.model_name: r for r in results}
    assert "XGBoost" in by_name
    # Same folds as every other model: exactly 8 held-out subjects and a
    # complete (no gap) out-of-fold prediction vector.
    assert by_name["XGBoost"].metrics["n_subjects"] == 8
    assert np.isfinite(by_name["XGBoost"].oof_predictions).all()
    assert "Mean baseline" in by_name  # baselines ran through the same procedure


def test_xgboost_comparison_is_deterministic_for_fixed_seed():
    cohort = generate_synthetic_cohort()
    enriched = features_mod.add_clinical_composites(cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    models = {"XGBoost": xgboost_model()}
    first = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)[0]
    second = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)[0]
    assert first.metrics == second.metrics


def test_missing_dependency_fails_with_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match="analysis"):
        xgboost_model()


def test_demo_includes_xgboost_only_when_asked(tmp_path):
    out = tmp_path / "out"
    assert demo.main(["--output-dir", str(out), "--seed", "1",
                      "--permutations", "3", "--bootstrap-resamples", "50"]) == 0
    import json

    summary = json.loads((out / "summary.json").read_text())
    assert "XGBoost" not in summary["models"]

    out_xgb = tmp_path / "out_xgb"
    assert demo.main(["--output-dir", str(out_xgb), "--seed", "1",
                      "--permutations", "3", "--bootstrap-resamples", "50",
                      "--with-xgboost"]) == 0
    summary = json.loads((out_xgb / "summary.json").read_text())
    assert "XGBoost" in summary["models"]
    assert summary["models"]["XGBoost"]["beats_baseline"] in {True, False}

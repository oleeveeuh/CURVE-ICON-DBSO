"""Malformed input must produce actionable validation errors."""

from __future__ import annotations

import pandas as pd
import pytest

from curve_icon_dbso import features as features_mod
from curve_icon_dbso.config import DemoConfig, HFOConfig, load_config
from curve_icon_dbso.errors import DataValidationError


def test_missing_subscore_columns_are_listed(tmp_path):
    """Dropping one subscore column must name the missing column and the
    remedy, not silently skip the feature family (historical behavior)."""
    from curve_icon_dbso.synthetic import generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_subjects=4, seed=1)
    broken = cohort.drop(columns=["PS_R"])
    with pytest.raises(DataValidationError, match=r"missing UPDRS subscore.*PS_R"):
        features_mod.add_clinical_composites(broken)


def test_missing_target_column_is_named():
    df = pd.DataFrame({"subject": ["S1", "S2"], "age": [50.0, 60.0]})
    with pytest.raises(DataValidationError, match="updrs_improvement"):
        features_mod.build_model_matrix(df)


def test_non_numeric_target_rejected():
    df = pd.DataFrame({
        "subject": ["S1", "S2"],
        "age": [50.0, 60.0],
        "years_since_diagnosis": [10.0, 12.0],
        "hfo_power_mean": [1.0, 2.0],
        "hfo_peak_freq_mean": [200.0, 210.0],
        "Brady_Asymmetry": [0.1, 0.2],
        "Rigidity_Asymmetry": [0.1, 0.2],
        "Tremor_Asymmetry": [0.1, 0.2],
        "updrs_improvement": ["30%", "40%"],  # strings, not numbers
    })
    with pytest.raises(DataValidationError, match="numeric"):
        features_mod.build_model_matrix(df)


def test_invalid_hfo_band_rejected():
    with pytest.raises(DataValidationError, match="band is invalid"):
        HFOConfig(low_hz=500.0, high_hz=80.0)


def test_config_unknown_keys_rejected(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 42\nnot_a_real_key: 1\n")
    with pytest.raises(DataValidationError, match="not_a_real_key"):
        load_config(config_file)


def test_config_missing_file_named(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_valid_config_round_trips(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 7\nn_subjects: 6\nhfo:\n  low_hz: 80\n  high_hz: 250\n")
    config = load_config(config_file)
    assert config.seed == 7
    assert config.n_subjects == 6
    assert config.hfo.high_hz == 250.0
    assert config == DemoConfig(seed=7, n_subjects=6, hfo=HFOConfig(low_hz=80, high_hz=250))

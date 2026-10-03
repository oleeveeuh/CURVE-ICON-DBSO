"""Shared fixtures: small deterministic datasets for offline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.synthetic import generate_synthetic_cohort


@pytest.fixture()
def synthetic_cohort() -> pd.DataFrame:
    """The standard 8-subject / 9-recording synthetic cohort."""
    return generate_synthetic_cohort(n_subjects=8, seed=42)


@pytest.fixture()
def simple_grouped_data():
    """Tiny grouped regression problem with NONSEQUENTIAL fold layout.

    Groups are interleaved (A, B, A, C, B, C, ...) so that every
    LeaveOneGroupOut test fold has noncontiguous indices — exactly the
    situation that broke the historical prediction-saving code.
    """
    rng = np.random.default_rng(7)
    groups = np.array(list("ABACBCABCBAC"))  # 12 rows, 3 subjects, interleaved
    x = pd.DataFrame({
        "f1": rng.normal(size=12),
        "f2": rng.normal(size=12),
    })
    y = np.array([10.0, 20.0, 12.0, 30.0, 22.0, 32.0, 11.0, 21.0, 23.0, 33.0, 13.0, 31.0])
    return x, y, groups

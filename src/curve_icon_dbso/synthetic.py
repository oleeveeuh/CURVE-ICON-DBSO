"""Deterministic synthetic cohort generation.

This module exists so the pipeline can be exercised end-to-end without
any access to restricted clinical data. Its output is **verification
data only**: it must never be presented as study results, and every
artifact produced from it is labeled as synthetic.

The generator reproduces the *structure* of the historical pilot cohort
(eight subjects, nine recordings — one subject contributes two
recordings) without reproducing any participant's values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# UPDRS part-III subscores used by the feature module.
_SUBSCORE_COLUMNS = [
    "Finger_Tap_R", "Finger_Tap_L",
    "Hand_Movement_R", "Hand_Movement_L",
    "PS_R", "PS_L",
    "Toe_Tapping_R", "Toe_Tapping_L",
    "Leg_Agility_R", "Leg_Agility_L",
    "Rigidity_RUE", "Rigidity_LUE", "Rigidity_RLE", "Rigidity_LLE",
    "Rest_Tremor_RUE", "Rest_Tremor_LUE", "Rest_Tremor_RLE", "Rest_Tremor_LLE",
    "Postural_Tremor_R_Hand", "Postural_Tremor_L_Hand",
    "Kinetic_Tremor_R_Hand", "Kinetic_Tremor_L_Hand",
]


def generate_synthetic_cohort(n_subjects: int = 8, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic cohort of UPDRS-style rows.

    One subject (the second, matching the historical cohort shape) gets
    two recordings, giving ``n_subjects + 1`` rows in total. The target
    ``updrs_improvement`` is produced by a fixed linear mixing of a few
    features plus noise, so corrected leave-one-subject-out evaluation
    has something learnable but imperfect to find.

    Args:
        n_subjects: Number of unique subjects (>= 2).
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        DataFrame with columns ``subject``, ``recording_id``,
        ``age``, ``years_since_diagnosis``, the UPDRS subscores,
        ``hfo_power_mean``, ``hfo_peak_freq_mean`` and the target
        ``updrs_improvement`` (percent).
    """
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")

    rng = np.random.default_rng(seed)

    rows = []
    for subject_idx in range(n_subjects):
        n_recordings = 2 if subject_idx == 1 else 1
        # Subject-level traits shared by all recordings of one subject.
        age = int(rng.integers(45, 75))
        duration = float(rng.integers(5, 20))
        hfo_power = float(rng.uniform(0.5, 2.0))
        hfo_peak = float(rng.uniform(150.0, 350.0))
        bradykinesia_severity = float(rng.uniform(0.0, 1.0))

        for rec_idx in range(1, n_recordings + 1):
            subscores = _synthetic_subscores(rng, bradykinesia_severity)
            # Fixed linear signal + noise; coefficients chosen so no
            # single feature trivially determines the target.
            true_signal = (
                25.0
                + 6.0 * (hfo_power - 1.25)
                - 0.25 * (age - 60.0)
                - 0.4 * (duration - 12.0)
                + 8.0 * bradykinesia_severity
            )
            noise = float(rng.normal(0.0, 6.0))
            improvement = float(np.clip(true_signal + noise, 0.0, 60.0))

            row = {
                "subject": f"SYN{subject_idx + 1:02d}",
                "recording_id": rec_idx,
                "age": age,
                "years_since_diagnosis": duration,
                "hfo_power_mean": hfo_power,
                "hfo_peak_freq_mean": hfo_peak,
                **subscores,
                "updrs_improvement": improvement,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _synthetic_subscores(rng: np.random.Generator, severity: float) -> dict[str, int]:
    """Draw integer UPDRS-style subscores (0-4) biased by ``severity``."""
    subscores = {}
    for col in _SUBSCORE_COLUMNS:
        lam = 4.0 * severity
        subscores[col] = int(np.clip(round(rng.poisson(lam)), 0, 4))
    return subscores

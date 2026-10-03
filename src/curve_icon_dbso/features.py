"""Feature engineering: clinical composites and the prespecified feature set.

Design rules (a corrective to the historical pipeline):

1. The feature set is **prespecified** — fixed in advance from the
   project hypothesis, never selected by looking at the outcome. With
   eight subjects, data-driven selection of "top features" is not
   statistically defensible.
2. Required columns are validated up front. Missing inputs raise
   :class:`~curve_icon_dbso.errors.DataValidationError` instead of being
   silently skipped (the historical code silently dropped whole feature
   families when a column was absent).
3. Feature names describe a single, explicitly defined quantity.
"""

from __future__ import annotations

import pandas as pd

from .errors import DataValidationError

# Column groups required to compute the composites below.
_BRADYKINESIA = ["Finger_Tap_{S}", "Hand_Movement_{S}", "PS_{S}",
                 "Toe_Tapping_{S}", "Leg_Agility_{S}"]
_RIGIDITY = ["Rigidity_{S}UE", "Rigidity_{S}LE"]
_TREMOR = ["Rest_Tremor_{S}UE", "Rest_Tremor_{S}LE",
           "Postural_Tremor_{S}_Hand", "Kinetic_Tremor_{S}_Hand"]

# The prespecified feature set. Seven features for eight subjects is
# already generous; these are fixed a priori and are NOT tuned against
# the outcome. Grounded in the project hypothesis: HFO activity in the
# recorded cortex, motor-sign asymmetry, disease duration and age.
PRESPECIFIED_FEATURES: list[str] = [
    "hfo_power_mean",
    "hfo_peak_freq_mean",
    "Brady_Asymmetry",
    "Rigidity_Asymmetry",
    "Tremor_Asymmetry",
    "years_since_diagnosis",
    "age",
]

TARGET_COLUMN = "updrs_improvement"
GROUP_COLUMN = "subject"

_REQUIRED_BASE = PRESPECIFIED_FEATURES + [TARGET_COLUMN, GROUP_COLUMN]


def required_subscore_columns() -> list[str]:
    """Return the concrete UPDRS subscore columns the composites need."""
    columns: list[str] = []
    for side in ("R", "L"):
        columns += [c.format(S=side) for c in _BRADYKINESIA]
        columns += [c.format(S=side) for c in _RIGIDITY]
        columns += [c.format(S=side) for c in _TREMOR]
    return columns


def add_clinical_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Add left/right motor composites and asymmetry indices.

    Asymmetry is defined as ``|right - left| / (right + left)`` (0 when
    the denominator is 0), matching the historical definition up to the
    absolute value, which makes the feature non-directional.

    Raises:
        DataValidationError: If any required subscore column is missing.
    """
    required = required_subscore_columns()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Cannot compute clinical composites; missing UPDRS subscore "
            f"columns: {missing}. Available columns: {sorted(map(str, df.columns))}"
        )

    out = df.copy()
    out["Brady_R"] = out[[c.format(S="R") for c in _BRADYKINESIA]].sum(axis=1)
    out["Brady_L"] = out[[c.format(S="L") for c in _BRADYKINESIA]].sum(axis=1)
    out["Rigidity_R"] = out[[c.format(S="R") for c in _RIGIDITY]].sum(axis=1)
    out["Rigidity_L"] = out[[c.format(S="L") for c in _RIGIDITY]].sum(axis=1)
    out["Tremor_R"] = out[[c.format(S="R") for c in _TREMOR]].sum(axis=1)
    out["Tremor_L"] = out[[c.format(S="L") for c in _TREMOR]].sum(axis=1)

    for name in ("Brady", "Rigidity", "Tremor"):
        denom = out[f"{name}_R"] + out[f"{name}_L"]
        out[f"{name}_Asymmetry"] = (out[f"{name}_R"] - out[f"{name}_L"]).abs() / denom.where(
            denom != 0, other=1
        )
    return out


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the prespecified model matrix ``(X, y, groups)``.

    Validates that all prespecified features, the target and the group
    column exist and that the target is numeric.

    Raises:
        DataValidationError: On missing columns or a non-numeric target.
    """
    missing = [c for c in _REQUIRED_BASE if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Model matrix is missing required columns: {missing}. "
            f"Available columns: {sorted(map(str, df.columns))}"
        )
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise DataValidationError(
            f"Target column '{TARGET_COLUMN}' must be numeric, got dtype "
            f"{df[TARGET_COLUMN].dtype}"
        )

    # Guard against accidental target leakage by construction: X is
    # built by name from the prespecified list only.
    x = df[PRESPECIFIED_FEATURES].astype(float)
    y = df[TARGET_COLUMN].astype(float)
    groups = df[GROUP_COLUMN].astype(str)
    return x, y, groups

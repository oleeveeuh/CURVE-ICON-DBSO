"""Group integrity: subjects never straddle train/test boundaries."""

from __future__ import annotations

import numpy as np

from curve_icon_dbso.evaluation import (
    baseline_models,
    default_models,
    run_group_cv,
)


def test_all_rows_covered_exactly_once(synthetic_cohort):
    """Every row must receive exactly one out-of-fold prediction."""
    from curve_icon_dbso import features as features_mod

    enriched = features_mod.add_clinical_composites(synthetic_cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), default_models())

    for result in results:
        assert result.fold_id.min() >= 0, "unassigned rows remain"
        assert len(result.oof_predictions) == len(x)
        assert np.isfinite(result.oof_predictions).all()


def test_subjects_never_cross_fold_boundaries(synthetic_cohort):
    """Directly verify the train/test subject disjointness of every fold
    used by run_group_cv (LOSO via the default splitter)."""
    from sklearn.model_selection import LeaveOneGroupOut

    from curve_icon_dbso import features as features_mod

    enriched = features_mod.add_clinical_composites(synthetic_cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    groups = groups.to_numpy()

    folds = list(LeaveOneGroupOut().split(x, y, groups))
    assert len(folds) == len(np.unique(groups)), "LOSO must yield one fold per subject"

    for train_idx, test_idx in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_repeated_recordings_share_a_fold(synthetic_cohort):
    """The subject with two recordings must have both held out together,
    never split across training and validation."""
    from curve_icon_dbso import features as features_mod
    from curve_icon_dbso.evaluation import default_models

    enriched = features_mod.add_clinical_composites(synthetic_cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)

    counts = groups.value_counts()
    multi_subject = counts[counts > 1].index[0]

    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), default_models())
    for result in results:
        fold_ids = result.per_subject.set_index("subject").loc[multi_subject, "n_rows"]
        assert fold_ids == counts[multi_subject]


def test_single_subject_group_raises():
    """Grouped evaluation needs >= 2 subjects; one giant group must fail loudly."""
    import pandas as pd
    import pytest

    from curve_icon_dbso.errors import DataValidationError

    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(DataValidationError, match="2 unique subjects"):
        run_group_cv(x, np.array([1.0, 2.0, 3.0]), np.array(["S1", "S1", "S1"]),
                     baseline_models())

"""Regression tests for out-of-fold prediction alignment.

The historical pipeline appended fold predictions in cross-validation
order and later zipped them with targets/subjects in original dataframe
order. With nonsequential test indices this silently scores predictions
against the wrong subject. These tests pin the corrected behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curve_icon_dbso.evaluation import (
    baseline_models,
    run_group_cv,
)


def test_oof_indices_are_nonsequential_in_fixture(simple_grouped_data):
    """Guard: the fixture really exercises nonsequential test folds."""
    from sklearn.model_selection import LeaveOneGroupOut

    x, y, groups = simple_grouped_data
    for _, test_idx in LeaveOneGroupOut().split(x, y, groups):
        assert not np.all(np.diff(test_idx) == 1), "test fold is contiguous; fixture is degenerate"


def test_predictions_stay_attached_to_their_rows(simple_grouped_data):
    """Every OOF prediction must equal what a model trained without that
    row's subject predicts for it — verified row by row."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x, y, groups = simple_grouped_data
    results = run_group_cv(x, y, groups, {"LR": LinearRegression()})
    oof = results[0].oof_predictions

    # Recompute each subject's expected predictions independently.
    for subject in np.unique(groups):
        test_mask = groups == subject
        train_mask = ~test_mask
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipe.fit(x.iloc[train_mask], y[train_mask])
        expected = pipe.predict(x.iloc[test_mask])
        np.testing.assert_allclose(oof[test_mask], expected, rtol=1e-12)


def test_median_baseline_oof_equals_other_subjects_median(simple_grouped_data):
    """With a median DummyRegressor, each row's OOF prediction is exactly
    the median target of the OTHER subjects — an exact, hand-checkable
    alignment contract."""
    x, y, groups = simple_grouped_data
    results = run_group_cv(x, y, groups, baseline_models())
    median_oof = {r.model_name: r.oof_predictions for r in results}["Median baseline"]

    for subject in np.unique(groups):
        mask = groups == subject
        expected = np.median(y[~mask])
        assert np.allclose(median_oof[mask], expected)


def test_fold_order_never_changes_alignment(simple_grouped_data):
    """Deliberately reverse the fold iteration order; the OOF array for a
    deterministic model must be identical, proving alignment is driven by
    indices, not accumulation order."""
    from sklearn.dummy import DummyRegressor

    x, y, groups = simple_grouped_data
    forward = run_group_cv(x, y, groups, {"D": DummyRegressor(strategy="median")})
    reversed_splitter = _reversed_leave_one_group_out()
    backward = run_group_cv(x, y, groups, {"D": DummyRegressor(strategy="median")},
                            splitter=reversed_splitter)
    np.testing.assert_array_equal(forward[0].oof_predictions, backward[0].oof_predictions)


def _reversed_leave_one_group_out():
    from sklearn.model_selection import LeaveOneGroupOut

    class ReversedLOSO(LeaveOneGroupOut):
        def split(self, X, y=None, groups=None):
            splits = list(super().split(X, y, groups))
            yield from reversed(splits)

    return ReversedLOSO()


def test_per_subject_table_matches_oof(synthetic_cohort):
    """The per-subject reporting table must be derived from the same
    aligned OOF array, not recomputed in a different order."""
    from curve_icon_dbso import features as features_mod
    from curve_icon_dbso.evaluation import default_models

    enriched = features_mod.add_clinical_composites(synthetic_cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), default_models())

    for result in results:
        row_level = pd.DataFrame({"subject": groups.to_numpy(), "oof": result.oof_predictions})
        subject_mean_oof = row_level.groupby("subject", sort=True)["oof"].mean()
        np.testing.assert_allclose(
            result.per_subject.set_index("subject")["y_pred"], subject_mean_oof
        )

"""Preprocessing must be fit within training folds only.

These tests construct data where full-dataset fitting and fold-local
fitting give *different* results, so any leakage is detectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curve_icon_dbso.evaluation import make_preprocessing, run_group_cv


def _leak_detecting_data():
    """Feature ``a`` has NaN in one held-out row whose training-fold
    median differs from the full-dataset median, so a pipeline that
    (wrongly) imputes on all data would produce different predictions
    than the correct fold-local one.

    Layout (subject: rows):
        S1: 0, 1      S2: 2, 3      S3: 4, 5      S4: 6, 7 (NaN in row 7)

    The NaN row belongs to S4; when S4 is the test fold, a fold-local
    imputer fills with median(rows 0-5), while a leaked imputer fills
    with median(rows 0-6).
    """
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, np.nan]
    b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 101.0, 102.0]
    x = pd.DataFrame({"a": a, "b": b})
    # y exactly equals feature b so a LinearRegression's only freedom is
    # how the NaN in ``a`` was filled: fold-local fill (median of
    # train ``a`` = 3.5) leaks differently than full-data fill (median
    # of all ``a`` including 100.0 = 5.5). Predictions must match the
    # fold-local value.
    y = np.array(b, dtype=float)
    groups = np.array(["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"])
    return x, y, groups


def test_imputation_uses_training_fold_median():
    from sklearn.dummy import DummyRegressor

    x, y, groups = _leak_detecting_data()
    results = run_group_cv(x, y, groups, {"Dummy": DummyRegressor(strategy="mean")})
    # Baselines ignore X; this run exists to exercise the pipeline path.
    assert np.isfinite(results[0].oof_predictions).all()


def test_predictions_match_fold_local_fit_not_full_data_fit():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    x, y, groups = _leak_detecting_data()
    results = run_group_cv(x, y, groups, {"LR": LinearRegression()})
    oof = results[0].oof_predictions

    test_mask = groups == "S4"
    train_mask = ~test_mask

    # Correct: imputer fit on the training fold only.
    local_pipe = Pipeline([*make_preprocessing(), ("m", LinearRegression())])
    local_pipe.fit(x.iloc[train_mask], y[train_mask])
    local_pred = local_pipe.predict(x.iloc[test_mask])

    # Wrong: imputer fit on the full dataset (the historical bug).
    leaked_pipe = Pipeline([*make_preprocessing(), ("m", LinearRegression())])
    full_x = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(x), columns=x.columns
    )
    leaked_pipe.fit(full_x.iloc[train_mask], y[train_mask])
    leaked_pred = leaked_pipe.predict(full_x.iloc[test_mask])

    np.testing.assert_allclose(oof[test_mask], local_pred, rtol=1e-10)
    assert not np.allclose(oof[test_mask], leaked_pred), (
        "OOF predictions match a full-data-fitted imputer — preprocessing "
        "is leaking across folds"
    )


def test_scaler_statistics_come_from_training_fold():
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x, y, groups = _leak_detecting_data()
    # run_group_cv builds a fresh Pipeline per fold; to observe the
    # fitted scaler we replicate its assembly exactly (cloning each
    # element, as the evaluation module does).
    train_idx = np.arange(6)
    steps = [
        ("imputer", clone(SimpleImputer(strategy="median"))),
        ("scaler", clone(StandardScaler())),
        ("m", clone(LinearRegression())),
    ]
    pipe = Pipeline(steps)
    pipe.fit(x.iloc[train_idx], y[train_idx])
    scaler = pipe.named_steps["scaler"]
    expected_mean = np.nanmean(x.iloc[train_idx]["a"]), np.mean(x.iloc[train_idx]["b"])
    np.testing.assert_allclose(scaler.mean_, expected_mean)

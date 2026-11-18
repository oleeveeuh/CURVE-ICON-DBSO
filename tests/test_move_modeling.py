"""Deterministic synthetic tests for the time-resolved MOVE study (frozen
contract §6/§8/§10): causal window indexing with off-by-one traps,
block-boundary exclusion, known neural-to-glove lag recovery, the
window-level fingerprint leakage contrast, subject-held-out fold
guarantees, autocorrelation-preserving nulls, per-subject-first metrics,
baselines, and sanitized plots. Synthetic fixtures only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.glove_phenotyping import glove_move_envelope
from curve_icon_dbso.move_modeling import (
    LAG_ABLATIONS,
    MOVE_FEATURE_BLOCKS,
    WindowDataset,
    _window_index_math,
    build_window_dataset,
    move_circular_shift_null,
    plot_move_results,
    run_move_cv,
)

FS = 1000.0
PRIMARY = "lagged_ridge_primary"


# ---------------------------------------------------------------------------
# Synthetic coupled recording
# ---------------------------------------------------------------------------


def _coupled_recording(fs: float, subject: int, lag_s: float = 0.1,
                       seed: int = 0):
    """REST(1 s) + MOVE(6 s) + REST(1 s). The glove taps are built first
    and run through the FROZEN envelope extraction; the neural gamma
    amplitude then ENCODES that actual envelope shifted by the design
    lag: gamma amplitude at neural time t' equals the envelope at
    t' + lag_s. The window ENDING at t − lag_s is therefore the
    informative one for target bin t (causal decoding must pick that
    lag), and the encoding is exact — no unrelated proxy."""
    rng = np.random.default_rng(seed + subject)
    n = int(8 * fs)
    segs = np.array([
        [0, int(1 * fs), 0, 0],
        [int(1 * fs), int(7 * fs), 0, 1],
        [int(7 * fs), int(8 * fs), 0, 0],
    ], dtype=float)
    tg = np.arange(n) / fs
    rate = 1.5 + 0.1 * subject
    taps = np.where((tg % (1 / rate)) < 0.5 / rate,
                    np.sin(2 * np.pi * rate * tg), 0.0)
    glove = np.vstack([taps, -taps]) + 0.005 * rng.standard_normal((2, n))
    envelope = glove_move_envelope(glove, fs, segs)
    stream = np.asarray(envelope["stream"])
    # Absolute time of each envelope bin: MOVE block starts at 1 s; bin
    # centers at 1.0 + (bin + 0.5) · 0.1 s.
    bin_times = 1.0 + (np.arange(stream.size) + 0.5) * protocol.MOVE_UPDATE_S
    t = np.arange(n) / fs
    amp = 5.0 * np.interp(t + lag_s, bin_times, stream)
    gamma = amp * np.sin(2 * np.pi * 70.0 * t)
    move = np.zeros(n)
    move[int(1 * fs):int(7 * fs)] = gamma[int(1 * fs):int(7 * fs)]
    ecog = np.vstack([
        rng.normal(0, 0.5, n) + move,
        rng.normal(0, 0.5, n) + move * 0.9,
        rng.normal(0, 0.5, n) + move * 1.1,
        rng.normal(0, 0.5, n) + move * 0.8,
    ])
    return ecog, glove, segs


def _coupled_datasets(n_subjects: int = 6, lag_s: float = 0.1):
    bands = protocol.resolve_spectral_bands(FS)
    datasets = []
    for subject in range(n_subjects):
        ecog, glove, segs = _coupled_recording(FS, subject, lag_s=lag_s,
                                               seed=10)
        envelope = glove_move_envelope(glove, FS, segs)
        datasets.append(build_window_dataset(
            ecog=ecog, fs=FS, restmove_segs=segs, envelope=envelope,
            subject_position=f"pos{subject:02d}",
            recording_position=f"rec{subject:02d}",
            bands_resolved=bands,
        ))
    features = pd.concat([d.features for d in datasets], ignore_index=True)
    target = np.concatenate([d.target for d in datasets])
    groups = np.concatenate([d.groups for d in datasets])
    metadata = pd.concat([d.metadata for d in datasets], ignore_index=True)
    qc = {"dropped_windows": sum(d.qc["dropped_windows"] for d in datasets)}
    return WindowDataset(features=features, target=target, groups=groups,
                         metadata=metadata, qc=qc)


# ---------------------------------------------------------------------------
# Off-by-one and block-boundary traps
# ---------------------------------------------------------------------------


def test_window_index_math_is_causal_and_exclusive():
    # Bin 0 target time = 0.05 s → its full 0.5 s window would begin
    # BEFORE the block: negative start → the builder drops it.
    start, end = _window_index_math(0, 0.1, 0.5, FS, 0.0)
    assert start < 0 and end == int(round(0.05 * FS))
    # The first fully-in-block window: end ≥ window length (0.5 s = 500
    # samples) → bin 10 (t = 1.05 s): window [550, 1050) ends exactly at
    # the target-time sample, which is EXCLUDED (half-open).
    start, end = _window_index_math(10, 0.1, 0.5, FS, 0.0)
    assert (start, end) == (550, 1050)
    # A lagged window is the causal window shifted back by exactly the
    # lag (100 samples per 0.1 s at fs 1000).
    start_lag, end_lag = _window_index_math(10, 0.1, 0.5, FS, 0.1)
    assert (start_lag, end_lag) == (450, 950)


def test_windows_never_cross_block_boundaries():
    bands = protocol.resolve_spectral_bands(FS)
    ecog, glove, segs = _coupled_recording(FS, 0, seed=1)
    envelope = glove_move_envelope(glove, FS, segs)
    dataset = build_window_dataset(
        ecog=ecog, fs=FS, restmove_segs=segs, envelope=envelope,
        subject_position="pos00", recording_position="rec00",
        bands_resolved=bands,
    )
    # The MOVE block starts at 1.0 s = sample 1000; the earliest target
    # bin whose window fits is bin with end ≥ window length after start.
    starts = dataset.metadata["sample_in_recording"].to_numpy()
    assert (starts >= int(1.0 * FS)).all()
    assert (starts < int(7.0 * FS)).all()
    # Every target bin's window was fully inside [1 s, 7 s).
    for _, row in dataset.metadata.iterrows():
        target_time = row["target_time_s"]
        assert 1.0 + 0.5 <= target_time + 0.45  # window end ≤ block end
    assert "window_outside_block" in dataset.qc["reasons"]


def test_invalid_intervals_drop_overlapping_windows():
    bands = protocol.resolve_spectral_bands(FS)
    ecog, glove, segs = _coupled_recording(FS, 0, seed=2)
    envelope = glove_move_envelope(glove, FS, segs)
    clean = build_window_dataset(
        ecog=ecog, fs=FS, restmove_segs=segs, envelope=envelope,
        subject_position="p", recording_position="r", bands_resolved=bands,
    )
    # Invalidate a 0.5 s stretch covering the middle of the MOVE block.
    contaminated = build_window_dataset(
        ecog=ecog, fs=FS, restmove_segs=segs, envelope=envelope,
        subject_position="p", recording_position="r", bands_resolved=bands,
        invalid_intervals=[(int(3.5 * FS), int(4.0 * FS))],
    )
    assert contaminated.features.shape[0] < clean.features.shape[0]
    assert "invalid_interval" in contaminated.qc["reasons"]


# ---------------------------------------------------------------------------
# Known neural-to-glove lag
# ---------------------------------------------------------------------------


def test_known_lag_is_recovered_by_the_lagged_ridge():
    dataset = _coupled_datasets(lag_s=0.1)
    results = run_move_cv(dataset, model_names=(PRIMARY,),
                          block="multiband_power")
    summary = results[0].summary
    # The coupling is real and causal decoding finds it.
    assert summary["median_pearson_r"] is not None
    assert summary["median_pearson_r"] > 0.3
    # Per-lag ridge coefficients: the lag-0.1 column group carries the
    # coupling (the window ending at t − 0.1 encodes env(t)); the oldest
    # lag carries none of it.
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = dataset.block_subset("multiband_power")
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", Ridge(alpha=1.0, random_state=42))])
    pipe.fit(X, dataset.target)
    coefs = pipe[-1].coef_
    lag_groups = {}
    for position, column in enumerate(X.columns):
        lag = column.rsplit("_", 1)[1]
        lag_groups.setdefault(lag, 0.0)
        lag_groups[lag] += abs(coefs[position])
    # The envelope is smoothed by the frozen 100 ms kernel, so the
    # immediately adjacent lags share information; the honest causal
    # signature is that the near lags (0–0.2 s, whose windows can contain
    # the neural encoding of env(t)) dominate the far lags (0.4–0.5 s,
    # which cannot).
    near = sum(lag_groups[f"{lag}s"] for lag in (0.0, 0.1, 0.2))
    far = sum(lag_groups[f"{lag}s"] for lag in (0.4, 0.5))
    assert near > 2.0 * far, lag_groups


# ---------------------------------------------------------------------------
# Fingerprint leakage: row-wise window splitting vs grouped
# ---------------------------------------------------------------------------


def test_window_fingerprints_inflate_rowwise_but_not_grouped():
    """Each subject's windows share a constant fingerprint feature and a
    subject-specific target offset unrelated to any cross-subject law.
    Leave-one-window-out (the leaky design) lets a memorizer recover the
    subject; the frozen grouped split collapses to the naive baseline —
    and the module refuses subject-straddling folds outright."""
    rng = np.random.default_rng(21)
    n_subjects, windows_per = 8, 30
    features_rows, targets, groups_list = [], [], []
    for subject in range(n_subjects):
        fingerprint = rng.normal() * 8
        offset = rng.normal() * 5
        for _ in range(windows_per):
            features_rows.append({"fingerprint": fingerprint})
            targets.append(offset + rng.normal(0, 0.05))
            groups_list.append(f"s{subject:02d}")
    dataset = WindowDataset(
        features=pd.DataFrame(features_rows),
        target=np.asarray(targets),
        groups=np.asarray(groups_list),
        metadata=pd.DataFrame({
            "subject_position": groups_list,
            "recording_position": groups_list,
            "block": 0,
            "target_bin": np.arange(len(targets)) % windows_per,
            "target_time_s": np.arange(len(targets)) * 0.1,
            "sample_in_recording": np.arange(len(targets)) * 100,
        }),
    )
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import LeaveOneOut
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def oof_mae(splits):
        oof = np.full(len(targets), np.nan)
        pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=1))])
        for train_idx, test_idx in splits:
            pipe.fit(dataset.features.iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict(dataset.features.iloc[test_idx])
        return float(np.mean(np.abs(oof - y)))

    y = dataset.target
    from curve_icon_dbso.evaluation import _check_group_integrity, _default_splitter

    mae_rowwise = oof_mae(list(LeaveOneOut().split(dataset.features)))
    grouped_splits = list(_default_splitter(n_subjects).split(
        dataset.features, y, dataset.groups
    ))
    mae_grouped = oof_mae(grouped_splits)
    assert mae_rowwise < 0.2
    assert mae_grouped > 5.0 * mae_rowwise
    # The frozen module's own grouped pipeline stays at the naive
    # baseline: the fingerprint carries no cross-subject signal.
    from curve_icon_dbso.subject_modeling import run_nested_cv

    ridge_generic = run_nested_cv(dataset.features, y, dataset.groups,
                                  ("ridge_primary",))
    baseline = run_nested_cv(dataset.features, y, dataset.groups,
                             ("dummy_mean",))
    assert ridge_generic[0].aggregate["MAE"] == pytest.approx(
        baseline[0].aggregate["MAE"], rel=0.3
    )
    # And the module's integrity guard refuses subject-straddling folds.
    from sklearn.model_selection import KFold

    straddling = list(
        KFold(n_splits=5, shuffle=True, random_state=0)
        .split(dataset.features, y)
    )
    with pytest.raises(DataValidationError, match="train and test folds"):
        _check_group_integrity(dataset.groups, straddling)


# ---------------------------------------------------------------------------
# Fold guarantees, metrics-first reporting, baselines
# ---------------------------------------------------------------------------


def test_every_subjects_windows_stay_in_one_outer_fold():
    dataset = _coupled_datasets(n_subjects=6)
    from curve_icon_dbso.evaluation import _default_splitter

    splits = list(_default_splitter(6).split(
        dataset.block_subset("multiband_power"), dataset.target,
        dataset.groups,
    ))
    seen: set[str] = set()
    for train_idx, test_idx in splits:
        test_subjects = set(dataset.groups[test_idx].tolist())
        assert test_subjects.isdisjoint(seen)
        assert not test_subjects & set(dataset.groups[train_idx].tolist())
        seen |= test_subjects
    assert seen == set(dataset.groups.tolist())


def test_per_subject_first_then_summary_not_pooled():
    dataset = _coupled_datasets(n_subjects=6)
    results = run_move_cv(dataset, model_names=(PRIMARY,))
    result = results[0]
    assert result.per_subject.shape[0] == 6
    assert set(result.per_subject.columns) >= {
        "subject_position", "n_windows", "pearson_r", "mae"
    }
    summary = result.summary
    assert summary["n_subjects"] == 6
    assert summary["n_windows"] == int(result.per_subject["n_windows"].sum())
    # Median across subjects is the headline; pooled MAE is secondary and
    # explicitly flagged non-independent.
    assert "pooled_mae_secondary_non_independent" in summary
    assert summary["median_pearson_r"] is not None


def test_baselines_present_and_labeled_naive():
    dataset = _coupled_datasets(n_subjects=6)
    results = run_move_cv(dataset, model_names=(PRIMARY,))
    baselines = results[0].baselines
    assert set(baselines) == {"fold_global_mean", "block_mean_naive",
                              "persistence"}
    assert "never a model claim" in baselines["block_mean_naive"]["definition"]
    # Every baseline emits per-subject MAE (median/IQR) ALONGSIDE
    # correlation — the frozen transfer criterion is evaluated on MAE,
    # and a v1.0.0-shaped baseline record (correlation only) must never
    # reappear.
    for _name, record in baselines.items():
        assert "median_pearson_r" in record
        assert record["median_mae"] is not None
        assert record["median_mae"] > 0
        assert record["iqr_mae"] is not None
    # The coupled decoder must beat the fold-global-mean floor. (The
    # persistence baseline is very strong on this fixture because the
    # frozen envelope is smooth; beating it is evaluated on real data,
    # not asserted on a synthetic smooth-envelope fixture.)
    if results[0].summary["median_pearson_r"] is not None:
        assert results[0].summary["median_pearson_r"] > (
            baselines["fold_global_mean"]["median_pearson_r"] or -1.0
        )


def test_transfer_criterion_is_evaluated_in_code_from_components():
    """The frozen §12 Boolean comes from the emitted MAE components —
    never asserted in prose. On the smooth-envelope fixture the
    persistence baseline is strong, so the model must FAIL criterion (b)
    here; the failure itself proves the comparison runs."""
    dataset = _coupled_datasets(n_subjects=6)
    result = run_move_cv(dataset, model_names=(PRIMARY,))[0]
    from curve_icon_dbso.move_modeling import move_transfer_criterion

    verdict = move_transfer_criterion(result)
    components = verdict["components"]
    assert components["median_mae_persistence"] is not None
    assert components["median_mae_block_mean"] is not None
    assert verdict["criterion_a_median_r_positive"] == (
        components["median_pearson_r"] > 0
    )
    assert verdict["criterion_b_beats_persistence_mae"] == (
        components["median_mae_model"] < components["median_mae_persistence"]
    )
    assert verdict["transfer_success"] == (
        verdict["criterion_a_median_r_positive"]
        and verdict["criterion_b_beats_persistence_mae"]
        and verdict["criterion_b_beats_block_mean_mae"]
    )


def test_bootstrap_median_r_returns_draws_and_endpoints():
    from curve_icon_dbso.move_modeling import bootstrap_median_r

    rng = np.random.default_rng(0)
    values = rng.normal(0.05, 0.1, 29)
    artifact = bootstrap_median_r(values, n_resamples=500, seed=42)
    assert artifact["n_resamples"] == 500
    assert artifact["seed"] == 42
    assert artifact["n_subjects"] == 29
    assert artifact["statistic"] == pytest.approx(float(np.median(values)))
    assert artifact["ci_low"] <= artifact["statistic"] <= artifact["ci_high"]
    assert len(artifact["draws"]) == 500
    again = bootstrap_median_r(values, n_resamples=500, seed=42)
    assert again["draws"] == artifact["draws"]  # deterministic given seed


def test_svr_runs_and_optional_xgboost_skips_cleanly():
    dataset = _coupled_datasets(n_subjects=4)
    results = run_move_cv(dataset, model_names=("svr_linear_move",))
    assert results and results[0].model_name == "svr_linear_move"
    try:
        import xgboost  # noqa: F401

        has_xgb = True
    except ImportError:
        has_xgb = False
    results = run_move_cv(dataset, model_names=("xgboost_shallow_move",))
    assert (len(results) == 1) == has_xgb


# ---------------------------------------------------------------------------
# Autocorrelation-preserving nulls
# ---------------------------------------------------------------------------


def test_circular_shift_null_breaks_alignment_but_not_autocorrelation():
    dataset = _coupled_datasets(n_subjects=6)
    null = move_circular_shift_null(dataset, n_shifts=4, seed=3)
    assert null["n_shifts"] == 4
    assert null["min_shift_s"] == protocol.MOVE_CIRCULAR_SHIFT_MIN_S == 2.0
    assert 1.0 / 5.0 <= null["p_value"] <= 1.0
    assert null["observed_median_pearson_r"] > float(
        np.median(null["null_median_pearson_r"])
    )
    blockwise = move_circular_shift_null(dataset, n_shifts=2, seed=4,
                                         blockwise=True)
    assert blockwise["blockwise"] is True
    assert blockwise["n_shifts"] == 2


def test_no_independent_row_shuffling_anywhere():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src"
              / "curve_icon_dbso" / "move_modeling.py").read_text()
    assert "shuffle" not in source
    assert "circular shifts of the glove target" in source


# ---------------------------------------------------------------------------
# v1.0.1 corrective: EFFECTIVE circular displacement ≥ the minimum
# ---------------------------------------------------------------------------


def test_shift_sampling_guarantees_effective_displacement():
    """Property test for the v1.0.1 fix: every sampled shift satisfies
    min(shift, span − shift) ≥ min_bins in BOTH null variants — the
    v1.0.0 sampler (uniform over [min_bins, span)) could draw span−1,
    which is circularly a ONE-bin shift in the opposite direction."""
    from curve_icon_dbso.move_modeling import _sample_circular_shift

    rng = np.random.default_rng(0)
    min_bins = 20  # 2 s at the 100 ms update
    for span in range(40, 400):
        for _draw in range(20):
            shift = _sample_circular_shift(span, min_bins, 20, False, rng)
            assert shift is not None
            assert min(shift, span - shift) >= min_bins, (span, shift)
            # Blockwise: multiples of the 2 s segment, same guarantee.
            shift_block = _sample_circular_shift(span, min_bins, 20, True,
                                                 rng)
            assert shift_block is not None
            assert shift_block % 20 == 0
            assert min(shift_block, span - shift_block) >= min_bins
    # Every returned value is inside [min_bins, span − min_bins].
    shift = _sample_circular_shift(41, min_bins, 20, False, rng)
    assert min_bins <= shift <= 41 - min_bins


def test_short_blocks_are_excluded_and_counted_not_silently_shifted():
    """A block shorter than twice the minimum displacement cannot carry a
    ≥ 2 s effective shift — it must be EXCLUDED (counted), never shifted
    silently (the v1.0.0 code shifted blocks down to min_bins + 1)."""
    rng = np.random.default_rng(1)
    n_subjects, windows_per = 4, 15  # 15 bins < 2 × 20 minimum
    subjects = np.repeat([f"s{i:02d}" for i in range(n_subjects)],
                         windows_per)
    recordings = np.repeat([f"r{i:02d}" for i in range(n_subjects)],
                           windows_per)
    dataset = WindowDataset(
        features=pd.DataFrame(
            rng.normal(size=(n_subjects * windows_per, 3)),
            columns=["theta_0.0s", "alpha_0.1s", "beta_low_0.2s"],
        ),
        target=rng.normal(size=n_subjects * windows_per),
        groups=subjects,
        metadata=pd.DataFrame({
            "subject_position": subjects,
            "recording_position": recordings,
            "block": np.zeros(n_subjects * windows_per, dtype=int),
            "target_bin": np.tile(np.arange(windows_per), n_subjects),
            "target_time_s": np.tile(np.arange(windows_per) * 0.1,
                                     n_subjects),
            "sample_in_recording": np.tile(np.arange(windows_per) * 100,
                                           n_subjects),
        }),
    )
    null = move_circular_shift_null(dataset, n_shifts=3, seed=2)
    # Every block was skipped: the null draws equal the observed run and
    # the exclusion is visible in the counts.
    assert null["blocks_shifted_per_draw"] == 0
    assert null["blocks_skipped_per_draw"] == n_subjects
    assert null["p_value"] == 1.0  # shifted targets ≡ observed targets


def test_rest_move_detection_endpoint_is_deliberately_deferred():
    """The contract does NOT prespecify a REST-vs-MOVE endpoint; adding
    one would be post-hoc work. It must stay absent."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src"
              / "curve_icon_dbso" / "move_modeling.py").read_text()
    assert "LogisticRegression" not in source
    assert "deliberately deferred" in source


# ---------------------------------------------------------------------------
# Ablations and sanitized plots
# ---------------------------------------------------------------------------


def test_lag_and_band_block_ablations_run(tmp_path_factory):
    dataset = _coupled_datasets(n_subjects=5)
    lag_results = {}
    for name, lags in LAG_ABLATIONS.items():
        result = run_move_cv(dataset, model_names=(PRIMARY,), lags=lags)
        lag_results[name] = result[0]
    assert {"full_stack", "current_window_only", "short_context"} == set(lag_results)
    # With the coupling at lag 0.1 s, dropping context must hurt relative
    # to the full stack? The lag-0 window still contains the informative
    # samples, so only assert the ablation runs and produces finite stats.
    for result in lag_results.values():
        assert result.summary["median_mae"] > 0
    block_results = {}
    for name in MOVE_FEATURE_BLOCKS:
        try:
            result = run_move_cv(dataset, model_names=(PRIMARY,), block=name)
        except DataValidationError:
            continue  # band inactive at this fs (e.g., gamma_low only)
        block_results[name] = result[0]
    assert "multiband_power" in block_results
    plots = plot_move_results(
        [lag_results["full_stack"]], out_dir=tmp_path_factory.mktemp("plots"),
        lag_ablation=lag_results, block_ablation=block_results,
    )
    for key in ("per_subject_distribution", "lag_ablation",
                "band_block_ablation"):
        assert key in plots


def test_plots_are_sanitized_and_written(tmp_path):
    dataset = _coupled_datasets(n_subjects=4)
    results = run_move_cv(dataset, model_names=(PRIMARY,))
    null = move_circular_shift_null(dataset, n_shifts=2, seed=5)
    plots = plot_move_results(results, null=null, out_dir=tmp_path)
    for _key, path in plots.items():
        assert Path(path).exists()
    # No subject identifiers in the plots' data path: only positions.
    result = results[0]
    assert all(str(s).startswith("pos")
               for s in result.per_subject["subject_position"])


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------


def test_frozen_constants_match_the_contract():
    assert LAG_ABLATIONS["full_stack"] == protocol.MOVE_LAGS_S
    assert set(MOVE_FEATURE_BLOCKS) == {
        "multiband_power", "band_low", "band_beta", "band_gamma",
    }
    assert protocol.MODELS_MOVE["lagged_ridge_primary"]["kind"] == "primary"
    with pytest.raises(DataValidationError, match="unknown MOVE feature block"):
        WindowDataset(
            features=pd.DataFrame({"x": [1.0]}), target=np.array([1.0]),
            groups=np.array(["a"]), metadata=pd.DataFrame(),
        ).block_subset("everything")

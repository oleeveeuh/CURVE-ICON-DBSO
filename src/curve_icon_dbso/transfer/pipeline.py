"""The v2 adaptive-transfer study orchestrator (Tracks A–D + ablations).

Runs the complete frozen matrix over a list of loaded recordings
(``cohort.Recording``): Track A detection LOSO + block-permutation null;
Track B/C zero-shot and few-shot conditions with the AR-baseline
residual design and the corrected shift nulls; the calibration curve;
Track D event-level association + prediction; the prespecified
ablations; and the Holm-corrected primary family with explicit
verdicts. All fitting respects subject and time boundaries; nothing
here can see an outer-test subject's labels unless the condition's
contract allows it (few-shot calibration segments only).

Outputs aggregate artifacts only; per-subject tables stay in the
caller-designated gitignored run directory.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import contract
from .cohort import CohortFlow, Recording, evaluate_cohort
from .detection import detection_null_pvalues, holm_adjust, loso_detection_cv
from .events import (
    association_analysis,
    association_null,
    build_event_table,
    event_neural_features,
    event_prediction_loso,
)
from .features import FeatureTable, build_bipolar_features, build_grid_features
from .regression import (
    SubjectConditionResult,
    ar_predict,
    blockwise_shift_null_delta_mae,
    circular_shift_null_delta_mae,
    circular_shift_null_median_r,
    circular_shift_nulls_fewshot_secondary,
    evaluate_fewshot_subject,
    evaluate_zero_shot_subject,
    fit_ar_on_streams,
    make_ridge_pipeline,
    select_alpha_inner,
)
from .splits import build_calibration_plan
from .targets import event_outcome_z, move_envelope_stream

# ---------------------------------------------------------------------------
# Optional fold-level parallelism (pure performance; identical output)
# ---------------------------------------------------------------------------

#: Module-level payloads consumed by worker processes (set in the parent
#: before dispatch; injected into workers through pickled initializer
#: args). All seeds are fixed module constants, so worker scheduling
#: cannot change any result.
_PARALLEL_RECORDINGS: list[Recording] = []
_PARALLEL_TABLES: list[FeatureTable] = []
_PARALLEL_ENVELOPES: list[dict] = []
_PARALLEL_WITH_XGB = True
_PARALLEL_BUILD_SPEC: dict = {}
_PARALLEL_DET_MODELS: dict[str, str] = {}


def _payload_snapshot() -> dict:
    """The dispatch payload as one picklable snapshot (v2.1.0: workers
    are SPAWNED, never forked — fork-after-OpenMP deadlocks on macOS
    (XGBoost's libomp barrier) hang the workers forever; spawn gives
    every worker a clean interpreter)."""
    return {
        "recordings": _PARALLEL_RECORDINGS,
        "tables": _PARALLEL_TABLES,
        "envelopes": _PARALLEL_ENVELOPES,
        "with_xgb": _PARALLEL_WITH_XGB,
        "build_spec": _PARALLEL_BUILD_SPEC,
        "det_models": _PARALLEL_DET_MODELS,
    }


def _init_worker(payload: dict) -> None:
    """Worker-side payload injection (runs once per spawned worker)."""
    global _PARALLEL_RECORDINGS, _PARALLEL_TABLES, _PARALLEL_ENVELOPES
    global _PARALLEL_WITH_XGB, _PARALLEL_BUILD_SPEC, _PARALLEL_DET_MODELS
    _PARALLEL_RECORDINGS = payload["recordings"]
    _PARALLEL_TABLES = payload["tables"]
    _PARALLEL_ENVELOPES = payload["envelopes"]
    _PARALLEL_WITH_XGB = payload["with_xgb"]
    _PARALLEL_BUILD_SPEC = payload["build_spec"]
    _PARALLEL_DET_MODELS = payload["det_models"]


def _map_folds(fn, n: int, workers: int) -> list:
    """Map ``fn`` over ``range(n)`` (fold/subject indices), optionally in
    spawned worker processes (clean interpreters — immune to the
    fork-after-OpenMP deadlock; each worker receives the payload once
    via its initializer, so memory grows by one payload copy per
    worker). Serial only when ``workers <= 1``. An unpicklable payload
    raises — a silent serial fallback could contradict recorded
    provenance (``workers`` > 1 while running serial). Output order —
    and therefore every artifact — is identical to the serial path."""
    if workers <= 1 or n < 2:
        return [fn(i) for i in range(n)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    payload = _payload_snapshot()
    pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)  # fail loudly
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, n), mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(payload,)) as ex:
        return list(ex.map(fn, range(n)))


def _build_tables_worker(index: int) -> FeatureTable:
    spec = _PARALLEL_BUILD_SPEC
    recording = _PARALLEL_RECORDINGS[index]
    return build_recording_tables(
        recording,
        window_s=spec.get("window_s", contract.WINDOW_S_PRIMARY),
        with_lags=spec.get("with_lags", False),
        bipolar=spec.get("bipolar", False),
    )


def _detection_fold_worker(held_out: int) -> tuple[dict, dict]:
    """One LOSO detection fold: inner alpha selection + fit + scoring on
    the held-out subject. Reads the injected tables and model spec."""
    from .detection import (
        _balanced_metrics,
        _make_pipeline,
        _select_alpha,
        _xgboost_classifier,
        event_detection_rate,
    )

    tables = _PARALLEL_TABLES
    train_tables = [t for i, t in enumerate(tables) if i != held_out]
    X_train = pd.concat([t.features for t in train_tables], ignore_index=True)
    y_train = np.concatenate([np.asarray(t.label) for t in train_tables])
    g_train = np.concatenate([
        np.full(len(t.features), i) for i, t in enumerate(train_tables)])
    test_table = tables[held_out]
    X_test = test_table.features
    y_test = np.asarray(test_table.label)
    block_keys = pd.factorize(test_table.meta["block"].values)[0]

    rows: dict[str, dict] = {}
    models: dict[str, object] = {}
    for model_name, kind in _PARALLEL_DET_MODELS.items():
        if kind == "skip":
            continue
        if kind == "ridge":
            alpha = _select_alpha(X_train, y_train, g_train)
            model = _make_pipeline(alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            chosen = {"alpha": alpha}
        else:
            model = _xgboost_classifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            chosen = dict(contract.XGB_FIXED_PARAMS)
        metrics = _balanced_metrics(y_test, y_pred, y_score)
        metrics["event_detection_rate"] = event_detection_rate(
            y_test, y_pred, block_keys)
        rows[model_name] = {
            "held_out_position": held_out,
            "n_windows": int(len(y_test)),
            "chosen": chosen,
            **metrics,
        }
        models[model_name] = model
    return rows, models


def _bc_fold_worker(held_out: int) -> tuple:
    """One Tracks-B/C fold: training fits + zero-shot + few-shot
    conditions for held-out subject ``held_out``. Reads the injected
    recordings/tables/envelopes; returns the same per-subject results as
    the serial loop body.

    The held-out subject's calibration and test tables are built
    SPAN-LOCALLY (v2.1.0): each span's ECoG preprocessing, features, and
    glove targets read only that span's raw samples, so mutating any
    future test sample cannot change a calibration artifact."""
    recordings = _PARALLEL_RECORDINGS
    tables = _PARALLEL_TABLES
    envelopes = _PARALLEL_ENVELOPES
    with_xgboost = _PARALLEL_WITH_XGB
    n = len(recordings)
    recording = recordings[held_out]
    if envelopes[held_out]["stream"] is None:
        return (False, None, None, {})

    train_idx = [i for i in range(n)
                 if i != held_out and envelopes[i]["stream"] is not None]
    train_streams = []
    train_keys = []
    for i in train_idx:
        stream = np.asarray(envelopes[i]["stream"], dtype=float)
        mu, sd = _move_constants(stream)
        train_streams.append((stream - mu) / sd)
        train_keys.append(np.asarray(envelopes[i]["block_keys"], dtype=int))
    ar_model = fit_ar_on_streams(train_streams, block_ids=train_keys)

    X_train_parts, y_train_parts, g_train_parts = [], [], []
    X_resid_parts, resid_parts, g_resid_parts = [], [], []
    for i in train_idx:
        t = tables[i]
        rows = np.isfinite(t.target)
        y_i = np.asarray(t.target)[rows]
        mu, sd = _move_constants(y_i)
        y_z = (y_i - mu) / sd
        stream_z = train_streams[train_idx.index(i)]
        keys_z = train_keys[train_idx.index(i)]
        ar_full = ar_predict(ar_model, stream_z, block_ids=keys_z)
        stream_pos = np.asarray(t.meta["stream_index"])[rows]
        resid_i = y_z - ar_full[stream_pos.astype(int)]
        resid_finite = np.isfinite(resid_i)  # NaN at AR block warm-ups
        flat = np.flatnonzero(rows)
        X_train_parts.append(t.features.iloc[flat])
        y_train_parts.append(y_z)
        g_train_parts.append(np.full(int(rows.sum()), i))
        X_resid_parts.append(t.features.iloc[flat[resid_finite]])
        resid_parts.append(resid_i[resid_finite])
        g_resid_parts.append(np.full(int(resid_finite.sum()), i))
    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = np.concatenate(y_train_parts)
    g_train = np.concatenate(g_train_parts)
    X_resid_train = pd.concat(X_resid_parts, ignore_index=True)
    resid_train = np.concatenate(resid_parts)
    g_resid_train = np.concatenate(g_resid_parts)
    alpha_fixed = select_alpha_inner(X_train, y_train, g_train)
    neural_pipe = make_ridge_pipeline(alpha_fixed)
    neural_pipe.fit(X_train, y_train)
    alpha_resid = select_alpha_inner(X_resid_train, resid_train, g_resid_train)
    residual_pipe = make_ridge_pipeline(alpha_resid)
    residual_pipe.fit(X_resid_train, resid_train)

    # Zero-shot full-recording endpoint: the held-out subject's full
    # table (its targets are the span-local stitched envelope, so the
    # purge/edge bins are already excluded).
    table = tables[held_out]
    full_rows = np.isfinite(table.target)
    zs = evaluate_zero_shot_subject(
        held_out, table.features.iloc[np.flatnonzero(full_rows)].reset_index(drop=True),
        np.asarray(table.target)[full_rows],
        table.meta["block"].values[full_rows], neural_pipe)

    # Held-out span tables — built independently per span (v2.1.0).
    union_columns = list(dict.fromkeys(c for t in tables for c in t.features.columns))
    plan_primary = build_calibration_plan(recording.blocks, recording.n_samples,
                                          recording.fs)
    test_table = build_recording_tables(
        recording, with_lags=True,
        span=(plan_primary.test_start_sample, recording.n_samples))
    test_table = _reindexed_table(
        test_table, union_columns)
    test_rows = np.isfinite(test_table.target)
    X_test = test_table.features.loc[test_rows].reset_index(drop=True)
    y_test_raw = np.asarray(test_table.target)[test_rows]
    test_blocks = test_table.meta["block"].values[test_rows]

    per_fraction: dict = {}
    for fraction in contract.CALIBRATION_FRACTIONS:
        plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                      recording.fs, fraction)
        calib_table = build_recording_tables(
            recording, with_lags=True, span=(0, plan.calib_end_sample))
        calib_table = _reindexed_table(
            calib_table,
            union_columns)
        calib_rows = np.isfinite(calib_table.target)
        if calib_rows.sum() < 10 or test_rows.sum() < 10:
            continue
        y_calib_raw = np.asarray(calib_table.target)[calib_rows]
        mu, sd = _move_constants(y_calib_raw)
        per_fraction[fraction] = evaluate_fewshot_subject(
            held_out, f"fewshot_{int(fraction*100)}",
            calib_table.features.loc[calib_rows].reset_index(drop=True),
            (y_calib_raw - mu) / sd,
            X_test, (y_test_raw - mu) / sd, test_blocks, alpha_fixed, ar_model,
            zero_shot_residual_pipe=residual_pipe,
            zero_shot_neural_pipe=neural_pipe,
            with_xgboost=with_xgboost,
            calib_block_ids=calib_table.meta["block"].values[calib_rows])

    zs_test = None
    if test_rows.sum() >= 10:
        zs_test = evaluate_zero_shot_subject(
            held_out, X_test, y_test_raw, test_blocks, neural_pipe)
    return (True, zs, zs_test, per_fraction)


def _detection_conditions_worker(held_out: int) -> dict:
    """One Track-C detection fold (zero-shot + few-shot refit scoring).

    The few-shot refits use SPAN-LOCAL calibration tables (v2.1.0): the
    calibration span's features are built from that span's raw ECoG
    only, so future test samples cannot alter the refit."""
    from .detection import _balanced_metrics, _make_pipeline, _select_alpha

    recordings = _PARALLEL_RECORDINGS
    tables = _PARALLEL_TABLES
    n = len(tables)
    recording = recordings[held_out]
    table = tables[held_out]
    y_all = np.asarray(table.label)
    X_all = table.features
    train_idx = [i for i in range(n) if i != held_out]
    X_train = pd.concat([tables[i].features for i in train_idx], ignore_index=True)
    y_train = np.concatenate([np.asarray(tables[i].label) for i in train_idx])
    g_train = np.concatenate([np.full(len(tables[i].features), i)
                              for i in train_idx])
    alpha = _select_alpha(X_train, y_train, g_train)
    pipe = _make_pipeline(alpha)
    pipe.fit(X_train, y_train)
    zs_full = _balanced_metrics(y_all, pipe.predict(X_all),
                                pipe.predict_proba(X_all)[:, 1])
    row: dict = {"held_out_position": held_out,
                 "bac_zero_shot_full": zs_full["balanced_accuracy"]}
    union_columns = list(dict.fromkeys(c for t in tables for c in t.features.columns))
    plan_primary = build_calibration_plan(recording.blocks, recording.n_samples,
                                          recording.fs)
    test_table = build_recording_tables(
        recording, with_lags=False,
        span=(plan_primary.test_start_sample, recording.n_samples))
    test_table = _reindexed_table(
        test_table, union_columns)
    if len(test_table.features) < 10:
        return row
    X_test = test_table.features.reset_index(drop=True)
    y_test = np.asarray(test_table.label)
    zs_test = _balanced_metrics(y_test, pipe.predict(X_test),
                                pipe.predict_proba(X_test)[:, 1])
    row["bac_zero_shot_test_interval"] = zs_test["balanced_accuracy"]
    for fraction in contract.CALIBRATION_FRACTIONS:
        plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                      recording.fs, fraction)
        calib_table = build_recording_tables(
            recording, with_lags=False, span=(0, plan.calib_end_sample))
        calib_table = _reindexed_table(
            calib_table,
            union_columns)
        y_calib = np.asarray(calib_table.label)
        if np.unique(y_calib).size < 2:
            # Single-class calibration segment (§2 skip discipline, v2.1.0
            # machinery): e.g. a recording whose chronological prefix
            # begins inside one MOVE block holds no REST rows, so the
            # binary few-shot refit is not computable for this
            # subject-fraction. Skipped with a machine-readable reason
            # and counted in the calibration curve — never silently.
            # An EMPTY span (all rows boundary-guarded out) is a row
            # deficit, not a class statement — labeled accordingly.
            row[f"bac_fewshot_{int(fraction*100)}_not_computable"] = (
                "calibration_segment_single_class" if y_calib.size
                else "calibration_too_few_windows")
            continue
        if len(calib_table.features) < 10:
            row[f"bac_fewshot_{int(fraction*100)}_not_computable"] = \
                "calibration_too_few_windows"
            continue
        refit = _make_pipeline(alpha)  # selected C fixed — frozen rule
        refit.fit(calib_table.features, y_calib)
        few = _balanced_metrics(y_test, refit.predict(X_test),
                                refit.predict_proba(X_test)[:, 1])
        row[f"bac_fewshot_{int(fraction*100)}"] = few["balanced_accuracy"]
    return row


def _subset_condition_worker(held_out: int) -> SubjectConditionResult | None:
    """One fold of an ablation variant condition (30% few-shot). The
    held-out subject's span tables are built with the variant spec
    (window/bipolar) and SPAN-LOCAL rules (v2.1.0), then reindexed to
    the training variant's canonical column set."""
    recordings = _PARALLEL_RECORDINGS
    tables = _PARALLEL_TABLES
    envelopes = _PARALLEL_ENVELOPES
    spec = _PARALLEL_BUILD_SPEC
    fraction = spec.get("fraction", contract.FEWSHOT_PRIMARY_FRACTION)
    n = len(recordings)
    recording = recordings[held_out]
    if envelopes[held_out]["stream"] is None:
        return None
    train_idx = [i for i in range(n) if i != held_out]
    train_streams = []
    train_keys = []
    for i in train_idx:
        stream = np.asarray(envelopes[i]["stream"], dtype=float)
        mu, sd = _move_constants(stream)
        train_streams.append((stream - mu) / sd)
        train_keys.append(np.asarray(envelopes[i]["block_keys"], dtype=int))
    ar_model = fit_ar_on_streams(train_streams, block_ids=train_keys)
    X_train_parts, y_train_parts, g_parts = [], [], []
    X_resid_parts, resid_parts, g_resid_parts = [], [], []
    for i in train_idx:
        t = tables[i]
        rows = np.isfinite(t.target)
        y_i = np.asarray(t.target)[rows]
        mu, sd = _move_constants(y_i)
        y_z = (y_i - mu) / sd
        stream_z = train_streams[train_idx.index(i)]
        keys_z = train_keys[train_idx.index(i)]
        ar_full = ar_predict(ar_model, stream_z, block_ids=keys_z)
        stream_pos = np.asarray(t.meta["stream_index"])[rows]
        resid_i = y_z - ar_full[stream_pos.astype(int)]
        resid_finite = np.isfinite(resid_i)  # NaN at AR block warm-ups
        flat = np.flatnonzero(rows)
        X_train_parts.append(t.features.iloc[flat])
        y_train_parts.append(y_z)
        g_parts.append(np.full(int(rows.sum()), i))
        X_resid_parts.append(t.features.iloc[flat[resid_finite]])
        resid_parts.append(resid_i[resid_finite])
        g_resid_parts.append(np.full(int(resid_finite.sum()), i))
    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = np.concatenate(y_train_parts)
    g_train = np.concatenate(g_parts)
    X_resid_train = pd.concat(X_resid_parts, ignore_index=True)
    resid_train = np.concatenate(resid_parts)
    g_resid_train = np.concatenate(g_resid_parts)
    alpha_fixed = select_alpha_inner(X_train, y_train, g_train)
    alpha_resid = select_alpha_inner(X_resid_train, resid_train, g_resid_train)
    residual_pipe = make_ridge_pipeline(alpha_resid)
    residual_pipe.fit(X_resid_train, resid_train)
    neural_pipe = make_ridge_pipeline(alpha_fixed)
    neural_pipe.fit(X_train, y_train)
    union_columns = list(dict.fromkeys(c for t in tables for c in t.features.columns))
    plan_primary = build_calibration_plan(recording.blocks, recording.n_samples,
                                          recording.fs)
    test_table = build_recording_tables(
        recording,
        window_s=spec.get("window_s", contract.WINDOW_S_PRIMARY),
        with_lags=True, bipolar=spec.get("bipolar", False),
        span=(plan_primary.test_start_sample, recording.n_samples))
    test_table = _reindexed_table(
        test_table, union_columns)
    test_rows = np.isfinite(test_table.target)
    plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                  recording.fs, fraction)
    calib_table = build_recording_tables(
        recording,
        window_s=spec.get("window_s", contract.WINDOW_S_PRIMARY),
        with_lags=True, bipolar=spec.get("bipolar", False),
        span=(0, plan.calib_end_sample))
    calib_table = _reindexed_table(
        calib_table,
        union_columns)
    calib_rows = np.isfinite(calib_table.target)
    if calib_rows.sum() < 10 or test_rows.sum() < 10:
        return None
    y_calib_raw = np.asarray(calib_table.target)[calib_rows]
    y_test_raw = np.asarray(test_table.target)[test_rows]
    mu, sd = _move_constants(y_calib_raw)
    return evaluate_fewshot_subject(
        held_out, "fewshot_30_condition",
        calib_table.features.loc[calib_rows].reset_index(drop=True),
        (y_calib_raw - mu) / sd,
        test_table.features.loc[test_rows].reset_index(drop=True),
        (y_test_raw - mu) / sd,
        test_table.meta["block"].values[test_rows], alpha_fixed, ar_model,
        zero_shot_residual_pipe=residual_pipe,
        zero_shot_neural_pipe=neural_pipe,
        with_xgboost=False,
        calib_block_ids=calib_table.meta["block"].values[calib_rows])


# ---------------------------------------------------------------------------
# Per-recording table construction
# ---------------------------------------------------------------------------


def _stitched_envelope(recording: Recording) -> dict:
    """Span-local stitched target envelope for ONE recording (v2.1.0):
    the calibration prefix (first 30 % of MOVE + its edge exclusions)
    and the untouched test interval are filtered INDEPENDENTLY from
    their own clipped raw glove spans and concatenated. Bins in the
    purge gap and inside either span's edge-exclusion margin carry no
    value — they are simply absent. ``block_keys`` are unique across the
    two spans so AR/persistence histories can never bridge the junction."""
    plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                  recording.fs)
    cal = move_envelope_stream(recording.glove, recording.fs,
                               recording.blocks,
                               clip=(0, plan.calib_end_sample))
    test = move_envelope_stream(recording.glove, recording.fs,
                                recording.blocks,
                                clip=(plan.test_start_sample,
                                      recording.n_samples))
    parts = [e for e in (cal, test) if e["stream"] is not None
             and np.asarray(e["stream"]).size]
    if not parts:
        return {"stream": None, "block_keys": None, "bin_starts": None,
                "qc_status": "failed", "qc_reason": "; ".join(
                    e["qc_reason"] for e in (cal, test) if e["qc_reason"]),
                "edge_excluded_bins": 0}
    offset = 0
    streams, keys, starts = [], [], []
    for envelope in parts:
        ids = np.asarray(envelope["block_ids"], dtype=int) + offset
        offset = int(ids.max()) + 1
        streams.append(np.asarray(envelope["stream"], dtype=float))
        keys.append(ids)
        starts.append(np.asarray(envelope["bin_starts"], dtype=int))
    stream = np.concatenate(streams)
    block_keys = np.concatenate(keys)
    bin_starts = np.concatenate(starts)
    return {
        "stream": stream,
        "block_keys": block_keys,
        "bin_starts": bin_starts,
        "qc_status": "ok",
        "qc_reason": "",
        "edge_excluded_bins": int(sum(e["edge_excluded_bins"] for e in parts)),
    }


def build_recording_tables(
    recording: Recording,
    window_s: float = contract.WINDOW_S_PRIMARY,
    with_lags: bool = False,
    bipolar: bool = False,
    span: tuple[int, int] | None = None,
) -> FeatureTable:
    """Feature table for one recording — either the FULL recording
    (``span=None``; ECoG preprocessing is whole-recording, the explicit
    offline-retrospective allowance used only where no span boundary
    exists, e.g. Track A's zero-shot evaluation and training subjects)
    or ONE clipped span ``span=(lo, hi)`` (v2.1.0: rereference, line
    treatment, features, and targets are built from that span's raw
    samples only).

    Targets come from the span-local envelope construction: full builds
    stitch the calibration-prefix and test-interval envelopes (bins in
    the purge gap or edge-exclusion margins carry NaN); span builds use
    that span's clipped envelope. Detection labels attach to every row."""
    if span is None:
        ecog = recording.ecog
        blocks = recording.blocks
        sample_shift = 0
    else:
        lo, hi = int(span[0]), int(span[1])
        ecog = recording.ecog[:, lo:hi]
        blocks = [(max(s, lo) - lo, min(e, hi) - lo, k)
                  for s, e, k in recording.blocks if min(e, hi) > max(s, lo)]
        sample_shift = lo
    if bipolar:
        table = build_bipolar_features(ecog, recording.fs, blocks,
                                       window_s=window_s, with_lags=with_lags)
    else:
        table = build_grid_features(ecog, recording.fs,
                                    recording.anchor_position, blocks,
                                    window_s=window_s, with_lags=with_lags)
    labels = (table.meta["block_kind"].values == "move").astype(int) \
        if len(table.meta) else np.zeros(0, dtype=int)
    table.label = labels
    if len(table.meta) == 0:
        # Empty span (every clipped block died to the boundary guard):
        # no rows, no merge, no keys — the caller's row-count guards
        # handle it (counted skip, never a crash).
        table.target = np.zeros(0, dtype=float)
        table.meta["stream_index"] = np.zeros(0, dtype=float)
        table.meta["stream_block_key"] = np.zeros(0, dtype=float)
        table.qc["envelope"] = "empty_span"
        return table
    if sample_shift and len(table.meta):
        for column in ("bin_start_sample", "window_earliest_sample",
                       "window_latest_sample"):
            if column in table.meta.columns:
                table.meta[column] = table.meta[column] + sample_shift
    if span is None:
        envelope = _stitched_envelope(recording)
    else:
        envelope = move_envelope_stream(recording.glove, recording.fs,
                                        recording.blocks, clip=span)
        envelope["block_keys"] = envelope["block_ids"]
    table.qc["envelope"] = envelope["qc_status"]
    table.qc["envelope_edge_excluded_bins"] = envelope.get(
        "edge_excluded_bins", 0)
    table.qc["span"] = None if span is None else tuple(int(v) for v in span)
    if envelope["stream"] is not None:
        merged = pd.merge(
            table.meta.assign(row_order=np.arange(len(table.meta))),
            pd.DataFrame({
                "bin_start_sample": envelope["bin_starts"],
                "target_raw": envelope["stream"],
                "stream_index": np.arange(len(envelope["stream"])),
                "stream_block_key": envelope["block_keys"],
            }),
            on="bin_start_sample", how="left",
        ).sort_values("row_order")
        table.target = merged["target_raw"].to_numpy(dtype=float)
        table.meta["stream_index"] = merged["stream_index"].to_numpy(dtype=float)
        table.meta["stream_block_key"] = merged["stream_block_key"].to_numpy(dtype=float)
    else:
        table.target = np.full(len(table.meta), np.nan)
        table.meta["stream_index"] = np.full(len(table.meta), np.nan)
        table.meta["stream_block_key"] = np.full(len(table.meta), np.nan)
    return table


def _move_constants(envelope_values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(envelope_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("not enough calibration MOVE bins for scale constants")
    return float(np.mean(values)), float(np.std(values))


def canonicalize_columns(tables: list[FeatureTable]) -> list[FeatureTable]:
    """Cohort-uniform feature space (contract §5: feature blocks are
    cohort-uniform). A recording whose grid lacks an offset gets that
    column filled 0.0 — the paired missingness indicator already marks
    the absence explicitly (never a silent median across channels)."""
    union: list[str] = []
    seen: set[str] = set()
    for table in tables:
        for column in table.features.columns:
            if column not in seen:
                seen.add(column)
                union.append(column)
    out: list[FeatureTable] = []
    for table in tables:
        frame = table.features.reindex(columns=union, fill_value=0.0)
        out.append(FeatureTable(features=frame, meta=table.meta,
                                target=table.target, label=table.label,
                                qc=table.qc))
    return out


# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------


def run_track_a(tables: list[FeatureTable], reduction_draws: int | None = None,
                label: str = "detection", workers: int = 1,
                with_xgboost: bool = True) -> dict:
    """LOSO detection + frozen null. The model-family Holm over
    {Ridge, XGBoost} is DESCRIPTIVE only (v2.1.0): the primary family
    consumes the Ridge endpoint's RAW permutation p exactly once, and
    the secondary detection family (AUROC / sensitivity / specificity /
    event-detection rate) is Holm-adjusted within itself from the same
    permutation draws."""
    results, fold_models = loso_detection_cv(tables, with_xgboost=with_xgboost,
                                             workers=workers)
    observed = {name: r.median_bac for name, r in results.items()
                if not name.endswith("_not_executed")}
    draws = reduction_draws or contract.DETECTION_NULL_DRAWS
    observed_secondary = {
        name: {metric: (float(np.nanmedian(frame[metric].to_numpy(dtype=float)))
                        if metric in frame.columns
                        and np.isfinite(frame[metric].to_numpy(dtype=float)).any()
                        else np.nan)
               for metric in contract.SECONDARY_FAMILY_DETECTION}
        for name, frame in ((n, r.per_subject) for n, r in results.items())
    }
    nulls = detection_null_pvalues(
        tables, {name: fold_models[name] for name in observed}, observed,
        n_draws=draws, observed_secondary=observed_secondary)
    family_p = {name: rec["p_raw"] for name, rec in nulls.items()
                if isinstance(rec, dict) and "p_raw" in rec}
    holm = holm_adjust(family_p) if family_p else {}
    for name, result in results.items():
        if name in nulls and isinstance(nulls[name], dict):
            result.null = {**nulls[name], "p_holm": holm.get(name)}
    per_subject = {name: frame.to_dict(orient="records")
                   for name, frame in
                   ((n, r.per_subject) for n, r in results.items())}
    secondary_summary = {}
    ridge_null = nulls.get("ridge_logistic", {})
    if isinstance(ridge_null, dict) and "secondary_family" in ridge_null:
        secondary_summary["ridge_logistic"] = ridge_null["secondary_family"]
    return {
        "label": label,
        "results": {
            name: {
                "median_bac": r.median_bac,
                "iqr_bac": list(r.iqr_bac),
                "bootstrap": r.bootstrap,
                "null": r.null,
                "verdict": r.verdict,
            } for name, r in results.items()
        },
        "per_subject_secondary_local_only": per_subject,
        "secondary_family": secondary_summary,
        "majority_baseline_bac": 0.5,
    }


def run_detection_conditions(
    recordings: list[Recording], tables: list[FeatureTable], workers: int = 1,
) -> dict:
    """Track C's detection side: zero-shot vs few-shot refit conditions
    on the identical test interval, plus the whole-recording zero-shot
    endpoint. The few-shot refit uses only the calibration span's
    labeled windows (logistic refit at the outer-fold-selected C)."""
    n = len(recordings)
    per_subject_rows: list[dict] = []
    calibration_curve: list[dict] = []
    calib_minutes: dict[float, list[float]] = {}

    global _PARALLEL_RECORDINGS, _PARALLEL_TABLES
    _PARALLEL_RECORDINGS = recordings
    _PARALLEL_TABLES = tables
    for row in _map_folds(_detection_conditions_worker, n, workers):
        if "bac_zero_shot_test_interval" not in row:
            continue
        per_subject_rows.append(row)
        recording = recordings[row["held_out_position"]]
        for fraction in contract.CALIBRATION_FRACTIONS:
            plan = build_calibration_plan(recording.blocks,
                                          recording.n_samples,
                                          recording.fs, fraction)
            minutes = plan.calibration_move_samples / recording.fs / 60.0
            calib_minutes.setdefault(fraction, []).append(minutes)

    frame = pd.DataFrame(per_subject_rows)
    zs_curve = {
        r["held_out_position"]: r["bac_zero_shot_test_interval"]
        for r in per_subject_rows
    }
    for fraction in contract.CALIBRATION_FRACTIONS:
        key = f"bac_fewshot_{int(fraction*100)}"
        skip_key = f"{key}_not_computable"
        n_single_class = int(
            frame[skip_key].notna().sum()) if skip_key in frame.columns else 0
        if key not in frame.columns:
            calibration_curve.append({
                "track": "detection",
                "fraction": fraction,
                "n_subjects": 0,
                "n_single_class_calibration_skips": n_single_class,
                "median_bac_fewshot": _median_or_nan([]),
                "median_bac_zero_shot_test_interval": _median_or_nan([]),
                "gain_over_zero_shot_bac": _paired_gain_bootstrap([], []),
                "median_usable_calibration_minutes": (
                    float(np.median(calib_minutes[fraction]))
                    if calib_minutes.get(fraction) else np.nan),
            })
            continue
        paired_model, paired_ref = [], []
        for _, r in frame.iterrows():
            model_value = r.get(key)
            ref_value = zs_curve.get(r["held_out_position"])
            if model_value is not None and ref_value is not None \
                    and np.isfinite(model_value) and np.isfinite(ref_value):
                paired_model.append(float(model_value))
                paired_ref.append(float(ref_value))
        calibration_curve.append({
            "track": "detection",
            "fraction": fraction,
            "n_subjects": len(paired_model),
            "n_single_class_calibration_skips": n_single_class,
            "median_bac_fewshot": _median_or_nan(paired_model),
            "median_bac_zero_shot_test_interval": _median_or_nan(paired_ref),
            "gain_over_zero_shot_bac": _paired_gain_bootstrap(
                paired_model, paired_ref),
            "median_usable_calibration_minutes": (
                float(np.median(calib_minutes[fraction]))
                if calib_minutes.get(fraction) else np.nan),
        })
    return {
        "per_subject_local_only": per_subject_rows,
        "median_bac_zero_shot_full": _median_or_nan(
            [r["bac_zero_shot_full"] for r in per_subject_rows]),
        "median_bac_zero_shot_test_interval": _median_or_nan(
            list(zs_curve.values())),
        "calibration_curve": calibration_curve,
    }


# ---------------------------------------------------------------------------
# Tracks B + C (conditions share one loop)
# ---------------------------------------------------------------------------


@dataclass
class RegressionArtifacts:
    zero_shot: dict = field(default_factory=dict)
    fewshot: dict = field(default_factory=dict)
    calibration_curve: list = field(default_factory=list)
    per_subject_local_only: dict = field(default_factory=dict)


def _aggregate_conditions(
    per_subject: list[SubjectConditionResult], condition: str,
) -> dict:
    rows = []
    for r in per_subject:
        row: dict = {
            "held_out_position": r.held_out_position,
            **{f"mae_{k}": v for k, v in r.mae.items()},
            **{f"r_{k}": v for k, v in r.r.items()},
            "incremental_r2": r.incremental_r2,
            "innovation_r": r.innovation_r,
            "n_test_windows": r.n_test_windows,
        }
        if "combined" in r.mae and "ar_baseline" in r.mae:
            row["mae_delta"] = r.mae["combined"] - r.mae["ar_baseline"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    aggregate: dict = {"condition": condition, "n_subjects": len(per_subject)}
    for column in frame.columns:
        if column in ("held_out_position", "n_test_windows"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        aggregate[f"median_{column}"] = float(np.median(finite))
        aggregate[f"iqr_{column}"] = [
            float(np.percentile(finite, 25)), float(np.percentile(finite, 75))]
    return aggregate


def _median_bootstrap(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if clean.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(contract.BOOTSTRAP_SEED)
    stats = np.empty(contract.BOOTSTRAP_N_RESAMPLES)
    for i in range(contract.BOOTSTRAP_N_RESAMPLES):
        idx = rng.integers(0, clean.size, size=clean.size)
        stats[i] = np.median(clean[idx])
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def _paired_gain_bootstrap(model: list[float], reference: list[float]) -> dict:
    """Paired per-subject gain with subject-cluster bootstrap CI and a
    sign-flip paired null (the frozen criterion for the gain endpoint)."""
    pairs = [(m, r) for m, r in zip(model, reference, strict=False)
             if m is not None and r is not None
             and np.isfinite(m) and np.isfinite(r)]
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "status": "insufficient_pairs"}
    gains = np.asarray([m - r for m, r in pairs])
    rng = np.random.default_rng(contract.BOOTSTRAP_SEED)
    stats = np.empty(contract.BOOTSTRAP_N_RESAMPLES)
    for i in range(contract.BOOTSTRAP_N_RESAMPLES):
        idx = rng.integers(0, gains.size, size=gains.size)
        stats[i] = float(np.median(gains[idx]))
    ci = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
    flips = 2000
    flip_stats = np.empty(flips)
    for i in range(flips):
        signs = rng.choice([-1.0, 1.0], size=gains.size)
        flip_stats[i] = float(np.median(signs * gains))
    count = int(np.count_nonzero(flip_stats >= float(np.median(gains))))
    return {
        "n_pairs": int(gains.size),
        "median_gain": float(np.median(gains)),
        "bootstrap_ci": list(ci),
        "sign_flip_p_raw": float((1 + count) / (1 + flips)),
        "status": "ok",
    }


def run_tracks_bc(
    recordings: list[Recording], tables: list[FeatureTable],
    null_draws: int | None = None, with_xgboost: bool = True,
    workers: int = 1,
) -> RegressionArtifacts:
    """Zero-shot + three few-shot conditions + calibration curve.

    ``tables`` are the WITH-LAGS full-recording tables (one per
    recording, same order). Zero-shot scoring is correlation-only;
    few-shot MAE endpoints use calibration-fitted scale constants.
    ``workers > 1`` dispatches the independent LOSO folds to spawned
    workers (identical, seed-deterministic output).
    """
    draws = null_draws or contract.REGRESSION_NULL_DRAWS
    artifacts = RegressionArtifacts()
    n = len(recordings)
    zero_shot_all: list[SubjectConditionResult] = []
    zero_shot_test_interval: list[SubjectConditionResult] = []
    per_condition: dict[float, list[SubjectConditionResult]] = {
        f: [] for f in contract.CALIBRATION_FRACTIONS
    }
    machinery_failures = 0

    # Training-side streams for the AR baseline: per-recording SPAN-LOCAL
    # stitched envelope streams (calibration prefix + test interval built
    # independently; standardization is training-side for TRAINING
    # subjects — legitimate; the held-out subject's constants never exist
    # zero-shot and come from the calibration segment few-shot).
    envelopes = [_stitched_envelope(r) for r in recordings]

    global _PARALLEL_RECORDINGS, _PARALLEL_TABLES, _PARALLEL_ENVELOPES, _PARALLEL_WITH_XGB
    _PARALLEL_RECORDINGS = recordings
    _PARALLEL_TABLES = tables
    _PARALLEL_ENVELOPES = envelopes
    _PARALLEL_WITH_XGB = with_xgboost
    folds = _map_folds(_bc_fold_worker, n, workers)
    for machinery_ok, zs, zs_test, per_fraction in folds:
        if not machinery_ok:
            machinery_failures += 1
            continue
        zero_shot_all.append(zs)
        for fraction, result in per_fraction.items():
            per_condition[fraction].append(result)
        if zs_test is not None:
            zero_shot_test_interval.append(zs_test)

    # --- Aggregates and nulls ---
    artifacts.zero_shot = _aggregate_conditions(zero_shot_all, "zero_shot")
    # Subject-cluster bootstrap CIs for the headline medians (the frozen
    # criteria are CI-based, not IQR-based).
    from .detection import _median_bootstrap_ci

    r_values = [r.r.get("neural_only") for r in zero_shot_all
                if r.r.get("neural_only") is not None]
    artifacts.zero_shot["bootstrap_ci_median_r_neural_only"] = list(
        _median_bootstrap_ci(np.asarray(r_values, dtype=float)))
    artifacts.zero_shot["null_median_r_neural_only"] = circular_shift_null_median_r(
        zero_shot_all, n_draws=draws)
    artifacts.zero_shot["null_median_r_persistence"] = circular_shift_null_median_r(
        zero_shot_all, n_draws=draws, model_key="persistence")
    fisher_pairs = _fisher_z_pairs(zero_shot_all)
    if fisher_pairs:
        artifacts.zero_shot["fisher_z_delta_vs_persistence"] = _fisher_z_delta(
            fisher_pairs)
    artifacts.zero_shot["median_usable_move_minutes_full_recording"] = float(
        np.median([
            np.isfinite(envelopes[i]["stream"]).sum() * contract.UPDATE_S / 60.0
            for i in range(n) if envelopes[i]["stream"] is not None
        ]))

    for fraction, per_subject in per_condition.items():
        key = f"fewshot_{int(fraction*100)}"
        aggregate = _aggregate_conditions(per_subject, key)
        if fraction == contract.FEWSHOT_PRIMARY_FRACTION:
            deltas = []
            for r in per_subject:
                if "combined" in r.mae and "ar_baseline" in r.mae:
                    deltas.append(r.mae["combined"] - r.mae["ar_baseline"])
            aggregate["bootstrap_ci_median_delta_mae"] = list(
                _median_bootstrap_ci(np.asarray(deltas, dtype=float)))
            aggregate["null_delta_mae_circular"] = circular_shift_null_delta_mae(
                per_subject, n_draws=draws)
            aggregate["null_delta_mae_blockwise"] = blockwise_shift_null_delta_mae(
                per_subject, n_draws=draws)
            aggregate["verdict_delta_mae"] = _delta_mae_verdict(aggregate)
            # Secondary regression family (contract §13): Holm WITHIN the
            # family, from the same circular-shift machinery (v2.1.0).
            aggregate["secondary_family_nulls"] = \
                circular_shift_nulls_fewshot_secondary(per_subject, n_draws=draws)
        artifacts.fewshot[key] = aggregate
        artifacts.per_subject_local_only[key] = [
            {k: v for k, v in r.__dict__.items()
             if k not in ("observed_stream", "combined_pred", "ar_pred", "block_ids")}
            for r in per_subject
        ]

    # --- Calibration curve (paired: same test interval everywhere) ---
    zs_test_r = {
        r.held_out_position: r.r.get("neural_only") for r in zero_shot_test_interval
    }
    calib_minutes = {}
    for recording in recordings:
        for fraction in contract.CALIBRATION_FRACTIONS:
            plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                          recording.fs, fraction)
            calib_minutes.setdefault(fraction, []).append(
                plan.calibration_move_samples / recording.fs / 60.0)
    for fraction in contract.CALIBRATION_FRACTIONS:
        key = f"fewshot_{int(fraction*100)}"
        per_subject = per_condition.get(fraction, [])
        model_r = {
            r.held_out_position: r.r.get("combined") for r in per_subject
        }
        paired_model, paired_ref = [], []
        for position, model_r_value in model_r.items():
            reference = zs_test_r.get(position)
            if model_r_value is not None and reference is not None:
                paired_model.append(model_r_value)
                paired_ref.append(reference)
        gain = _paired_gain_bootstrap(paired_model, paired_ref)
        artifacts.calibration_curve.append({
            "fraction": fraction,
            "n_subjects": len(per_subject),
            "median_usable_calibration_minutes": float(
                np.median(calib_minutes[fraction])) if calib_minutes[fraction] else np.nan,
            "median_r_combined_test_interval": _median_or_nan(
                [v for v in model_r.values() if v is not None]),
            "median_r_zero_shot_test_interval": _median_or_nan(
                [v for v in zs_test_r.values() if v is not None]),
            "gain_over_zero_shot_r": gain,
        })
    return artifacts


def _fisher_z_pairs(per_subject: list[SubjectConditionResult]) -> list[tuple[float, float]]:
    pairs = []
    for r in per_subject:
        m = r.r.get("neural_only")
        p = r.r.get("persistence")
        if m is not None and p is not None and np.isfinite(m) and np.isfinite(p):
            pairs.append((m, p))
    return pairs


def _fisher_z_delta(pairs: list[tuple[float, float]]) -> dict:
    zs_model = np.arctanh(np.clip([m for m, _ in pairs], -0.999999, 0.999999))
    zs_persist = np.arctanh(np.clip([p for _, p in pairs], -0.999999, 0.999999))
    delta = zs_model - zs_persist
    rng = np.random.default_rng(contract.BOOTSTRAP_SEED)
    stats = np.empty(contract.BOOTSTRAP_N_RESAMPLES)
    for i in range(contract.BOOTSTRAP_N_RESAMPLES):
        idx = rng.integers(0, delta.size, size=delta.size)
        stats[i] = float(np.tanh(np.mean(delta[idx])))
    ci = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
    flips = 2000
    flip_stats = np.empty(flips)
    for i in range(flips):
        signs = rng.choice([-1.0, 1.0], size=delta.size)
        flip_stats[i] = float(np.tanh(np.mean(signs * delta)))
    count = int(np.count_nonzero(flip_stats >= float(np.tanh(np.mean(delta)))))
    return {
        "n_pairs": int(delta.size),
        "median_delta_fisher_z_r": float(np.tanh(np.mean(delta))),
        "bootstrap_ci_delta_r": list(ci),
        "sign_flip_p_raw": float((1 + count) / (1 + flips)),
    }


def _delta_mae_verdict(aggregate: dict) -> dict:
    median_delta = aggregate.get("median_mae_delta")
    null = aggregate.get("null_delta_mae_circular", {})
    return {
        "median_delta_mae": median_delta,
        "note": "delta = MAE(combined) − MAE(ar_baseline); negative = improvement",
        "circular_null_p_raw": null.get("p_raw"),
        "criterion_note": (
            "success requires median < 0 with subject-cluster bootstrap CI "
            "excluding 0 and Holm-adjusted shift-null p < 0.05 (§11)"
        ),
    }


def _median_or_nan(values: list) -> float:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else np.nan


# ---------------------------------------------------------------------------
# Track D
# ---------------------------------------------------------------------------


def run_track_d(recordings: list[Recording], null_draws: int | None = None) -> dict:
    """Event-level analysis (contract §10, v2.1.0 corrections).

    Association claims run for the three PER-TAP outcomes —
    ``log_inter_tap_interval`` (primary), ``tap_amplitude_z``,
    ``peak_velocity_z`` (within-recording z) — each with the Fisher-z
    bootstrap CI AND the within-recording shift null. The block-level
    outcomes (``iti_cv``, ``within_block_amplitude_decrement``) are
    reported descriptively (they have no prespecified per-tap
    association test, hence no p-value and no silent family
    membership). The mean of within-recording z-scores is NEVER
    reported as an endpoint (approximately zero by construction —
    removed in v2.1.0). All event p-values enter ONE Holm family
    (``contract.EVENT_P_VALUE_FAMILY``) exactly once."""
    draws = null_draws or contract.REGRESSION_NULL_DRAWS
    outcome_columns = {
        "log_inter_tap_interval": "log_iti_s",
        "tap_amplitude_z": "amplitude_z",
        "peak_velocity_z": "peak_velocity_z",
    }
    association_records: dict[str, list[dict]] = {name: [] for name in outcome_columns}
    association_streams: dict[str, list[dict]] = {name: [] for name in outcome_columns}
    event_frames: list[pd.DataFrame] = []
    secondary: dict[str, list[float]] = {
        "iti_cv": [], "within_block_amplitude_decrement": []}
    eligible = 0
    for recording in recordings:
        events = build_event_table(recording.glove, recording.fs, recording.blocks)
        if not events:
            continue
        frame = pd.DataFrame(events)
        onsets = frame["onset_sample"].to_numpy(dtype=int)
        hfa = event_neural_features(recording.ecog, recording.fs,
                                    recording.anchor_position, onsets)
        frame["anchor_hfa_log_power"] = hfa
        frame["amplitude_z"] = event_outcome_z(
            frame["amplitude_raw"].to_numpy(dtype=float))
        frame["peak_velocity_z"] = event_outcome_z(
            frame["peak_velocity_raw"].to_numpy(dtype=float))
        valid = frame.dropna(subset=["log_iti_s", "anchor_hfa_log_power"])
        eligible += 1
        event_frames.append(valid)
        if len(valid) >= 10:
            feature = valid["anchor_hfa_log_power"].to_numpy(dtype=float)
            blocks = valid["block"].to_numpy(dtype=int)
            for outcome_name, column in outcome_columns.items():
                outcome_values = valid[column].to_numpy(dtype=float)
                finite = np.isfinite(outcome_values)
                if finite.sum() < 10:
                    continue
                rho = float(spearmanr_scipy(outcome_values[finite], feature[finite]))
                association_records[outcome_name].append({
                    "spearman_r": rho, "n_taps": int(finite.sum()),
                })
                association_streams[outcome_name].append({
                    "outcome": outcome_values[finite],
                    "feature": feature[finite],
                    "block_ids": blocks[finite],
                    "observed_spearman": rho,
                })
        # Block-level secondary outcomes (descriptive; within-person).
        for _block_number, block_frame in valid.groupby("block"):
            iti = block_frame["iti_s"].to_numpy(dtype=float)
            if iti.size >= 5:
                secondary["iti_cv"].append(float(np.std(iti) / np.mean(iti)))
                amplitude = block_frame["amplitude_raw"].to_numpy(dtype=float)
                if np.mean(amplitude) != 0 and amplitude.size >= 2:
                    slope = float(np.polyfit(np.arange(amplitude.size),
                                             amplitude, 1)[0])
                    secondary["within_block_amplitude_decrement"].append(
                        slope / float(np.mean(amplitude)))

    result: dict = {
        "n_recordings_with_events": eligible,
        "primary_association": association_analysis(
            association_records["log_inter_tap_interval"]),
    }
    if association_streams["log_inter_tap_interval"]:
        result["primary_association_null"] = association_null(
            association_streams["log_inter_tap_interval"], n_draws=draws)
    # Secondary per-tap outcome associations (same machinery, same nulls).
    result["secondary_tap_associations"] = {}
    for outcome_name in ("tap_amplitude_z", "peak_velocity_z"):
        entry: dict = {"association": association_analysis(
            association_records[outcome_name], outcome=outcome_name)}
        if association_streams[outcome_name]:
            entry["null"] = association_null(
                association_streams[outcome_name], n_draws=draws)
        result["secondary_tap_associations"][outcome_name] = entry
    result["secondary_family_per_recording_median"] = {
        name: (float(np.median(values)) if values else np.nan)
        for name, values in secondary.items()
    }
    result["secondary_descriptive_note"] = (
        "iti_cv and within_block_amplitude_decrement are block-level "
        "outcomes with no prespecified per-tap association test: they "
        "carry no p-value and are not members of the event p-value "
        "family; per-tap amplitude/velocity z are analyzed through "
        "their association claims instead of a mean-of-z endpoint "
        "(approximately zero by construction)")
    result["event_prediction"] = event_prediction_loso(event_frames) \
        if event_frames else {"status": "not_executable_no_events"}

    # --- ONE event p-value family, adjusted once (v2.1.0) ---
    family_p: dict[str, float] = {}
    if result["primary_association_null"] is not None:
        family_p["log_inter_tap_interval_association"] = float(
            result["primary_association_null"]["p_raw"])
    for outcome_name, entry in result["secondary_tap_associations"].items():
        null = entry.get("null")
        if null is not None:
            family_p[f"{outcome_name}_association"] = float(null["p_raw"])
    event_pred = result["event_prediction"]
    if event_pred.get("status") == "ok":
        family_p["event_prediction_shift_null"] = float(
            event_pred["null"]["p_raw"])
    result["event_family"] = {
        "family": list(contract.EVENT_P_VALUE_FAMILY),
        "p_raw": family_p,
        "p_holm_within_family": holm_adjust(family_p) if family_p else {},
        "members_without_p_descriptive_only": [
            name for name in contract.EVENT_SECONDARY_FAMILY
            if name not in ("tap_amplitude_z", "peak_velocity_z")
        ],
    }
    return result


def spearmanr_scipy(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


# ---------------------------------------------------------------------------
# Full study
# ---------------------------------------------------------------------------


@dataclass
class StudyArtifacts:
    cohort_flow: dict
    track_a: dict
    detection_conditions: dict = field(default_factory=dict)
    tracks_bc: RegressionArtifacts = field(default_factory=RegressionArtifacts)
    track_d: dict = field(default_factory=dict)
    ablations: dict = field(default_factory=dict)
    primary_family: dict = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    #: LOCAL-ONLY cohort flow (ordinal-position lists; gitignored run
    #: directory) — never merged into the committable summary.
    cohort_flow_local_only: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def _machinery_status(flow: CohortFlow, track_d_n: int) -> dict:
    """Per-track executability under the §14 stopping rule: a track with
    fewer than 50 % of the loaded recordings passing its data gates is
    reported NOT EXECUTABLE — no substitution, and its endpoints are
    excluded from the primary Holm family rather than silently
    failing."""
    def _fraction(n_passing: int) -> dict:
        denominator = max(flow.n_converted, 1)
        fraction = n_passing / denominator
        return {
            "n_passing": int(n_passing),
            "n_loaded": int(flow.n_converted),
            "pass_fraction": float(fraction),
            "executable": bool(n_passing > 0
                               and fraction >= (1.0 - contract.MAX_INELIGIBLE_FRACTION)),
        }

    n_eligible = len(flow.eligible_positions)
    return {
        "detection": _fraction(n_eligible),
        "regression": _fraction(n_eligible),
        "events": _fraction(track_d_n),
    }


def run_transfer_study(
    recordings: list[Recording],
    with_ablations: bool = True,
    null_draws: int | None = None,
    with_xgboost: bool = True,
    workers: int = 1,
) -> StudyArtifacts:
    """The complete frozen matrix (§16 prohibitions apply to any caller
    change made after seeing real results). ``workers > 1`` parallelizes
    independent feature builds and LOSO folds across spawned worker
    processes — a pure performance change with identical, seed-deterministic
    output."""
    flow = evaluate_cohort(recordings)
    eligible = [r for r in recordings if r.position in set(flow.eligible_positions)]

    global _PARALLEL_RECORDINGS, _PARALLEL_BUILD_SPEC
    _PARALLEL_RECORDINGS = eligible

    def _build(spec: dict) -> list[FeatureTable]:
        global _PARALLEL_BUILD_SPEC
        _PARALLEL_BUILD_SPEC = spec
        built = _map_folds(_build_tables_worker, len(eligible), workers)
        return canonicalize_columns(list(built))

    detection_tables = _build({"with_lags": False})
    regression_tables = _build({"with_lags": True})

    track_a = run_track_a(detection_tables, reduction_draws=null_draws,
                          workers=workers, with_xgboost=with_xgboost)
    detection_conditions = run_detection_conditions(eligible, detection_tables,
                                                    workers=workers)
    tracks_bc = run_tracks_bc(eligible, regression_tables,
                              null_draws=null_draws, with_xgboost=with_xgboost,
                              workers=workers)
    track_d_eligible = [r for r in eligible
                        if r.position not in set(flow.track_d_ineligible_positions)]
    track_d = run_track_d(track_d_eligible, null_draws=null_draws)

    # Per-subject references for the ablation contrast family (§12).
    primary_detection_bac = {
        int(row["held_out_position"]): float(row["balanced_accuracy"])
        for row in track_a["per_subject_secondary_local_only"].get(
            "ridge_logistic", [])
    }
    primary_fewshot_deltas: dict[int, float] = {}
    for record in tracks_bc.per_subject_local_only.get(
            f"fewshot_{int(contract.FEWSHOT_PRIMARY_FRACTION*100)}", []):
        mae = record.get("mae", {})
        if "combined" in mae and "ar_baseline" in mae:
            primary_fewshot_deltas[int(record["held_out_position"])] = float(
                mae["combined"] - mae["ar_baseline"])

    ablations: dict = {}
    if with_ablations:
        ablations = run_ablations(eligible, detection_tables, regression_tables,
                                  null_draws=null_draws, workers=workers,
                                  primary_detection_bac=primary_detection_bac,
                                  primary_fewshot_deltas=primary_fewshot_deltas)

    machinery = _machinery_status(flow, track_d.get("n_recordings_with_events", 0))
    primary_family = _primary_family(track_a, tracks_bc, machinery)
    verdicts = _verdicts(track_a, tracks_bc, track_d, primary_family, machinery)
    return StudyArtifacts(
        cohort_flow=flow.summary(),
        track_a=track_a,
        detection_conditions=detection_conditions,
        tracks_bc=tracks_bc,
        track_d=track_d,
        ablations=ablations,
        primary_family=primary_family,
        verdicts=verdicts,
        cohort_flow_local_only=flow.local_summary(),
    )


def _sign_flip_contrast(deltas: np.ndarray,
                        seed: int = contract.BOOTSTRAP_SEED) -> dict:
    """Two-sided sign-flip null on a paired per-subject contrast median
    (the ablation-contrast convention; direction noted per member)."""
    if deltas.size < 3:
        return {"n_pairs": int(deltas.size), "status": "insufficient_pairs"}
    observed = float(np.median(deltas))
    rng = np.random.default_rng(seed)
    flips = contract.BOOTSTRAP_N_RESAMPLES
    stats = np.empty(flips)
    for i in range(flips):
        signs = rng.choice([-1.0, 1.0], size=deltas.size)
        stats[i] = float(np.median(signs * deltas))
    count = int(np.count_nonzero(np.abs(stats) >= abs(observed)))
    return {
        "n_pairs": int(deltas.size),
        "median_contrast": observed,
        "sign_flip_p_raw_two_sided": float((1 + count) / (1 + flips)),
        "status": "ok",
    }


def run_ablations(
    eligible: list[Recording], detection_tables: list[FeatureTable],
    regression_tables: list[FeatureTable], null_draws: int | None = None,
    workers: int = 1,
    primary_detection_bac: dict[int, float] | None = None,
    primary_fewshot_deltas: dict[int, float] | None = None,
) -> dict:
    """Prespecified ablations (§12): all reported, none selected. Each
    ablation also yields a PAIRED per-subject contrast against the
    corresponding primary condition, with a two-sided sign-flip p and a
    Holm correction ACROSS the contrast family (the documented
    secondary-family correction, v2.1.0)."""
    draws = null_draws or contract.DETECTION_NULL_DRAWS
    out: dict = {}

    global _PARALLEL_RECORDINGS, _PARALLEL_BUILD_SPEC
    _PARALLEL_RECORDINGS = eligible

    def _build(spec: dict) -> list[FeatureTable]:
        global _PARALLEL_BUILD_SPEC
        _PARALLEL_BUILD_SPEC = spec
        built = _map_folds(_build_tables_worker, len(eligible), workers)
        return canonicalize_columns(list(built))

    # Window ablation — detection (1.0 s build).
    tables_1s = _build({"window_s": contract.WINDOW_S_ABLATION,
                        "with_lags": False})
    det_1s = run_track_a(tables_1s, reduction_draws=draws,
                         label="detection_window_1s", workers=workers,
                         with_xgboost=False)
    out["detection_window_1s"] = {
        k: v for k, v in det_1s["results"].items() if k == "ridge_logistic"
    }
    # Bipolar ablation — detection.
    bipolar_tables = _build({"with_lags": False, "bipolar": True})
    det_bipolar = run_track_a(bipolar_tables, reduction_draws=draws,
                              label="detection_bipolar", workers=workers,
                              with_xgboost=False)
    out["detection_bipolar"] = {
        k: v for k, v in det_bipolar["results"].items() if k == "ridge_logistic"
    }
    # Window ablation — regression few-shot 30% (1.0 s build).
    tables_1s_lag = _build({"window_s": contract.WINDOW_S_ABLATION,
                            "with_lags": True})
    out["regression_window_1s"] = _condition_on_tables(
        eligible, tables_1s_lag, keep=None, workers=workers)
    # Lag + band-group ablations — regression few-shot 30% column subsets.
    out["regression_lag_subset_00_02s"] = _condition_on_tables(
        eligible, regression_tables,
        lambda c: c.endswith(("_0.0s", "_0.1s", "_0.2s")), workers=workers)
    low_bands = tuple(contract.BAND_GROUP_LOW)
    high_bands = tuple(contract.BAND_GROUP_HIGH)
    out["regression_band_group_low"] = _condition_on_tables(
        eligible, regression_tables,
        lambda c: any(f"_{band}_" in c for band in low_bands), workers=workers)
    out["regression_band_group_high"] = _condition_on_tables(
        eligible, regression_tables,
        lambda c: any(f"_{band}_" in c for band in high_bands), workers=workers)

    # --- Contrast family (paired per-subject; §13 secondary family) ---
    primary_detection_bac = primary_detection_bac or {}
    primary_fewshot_deltas = primary_fewshot_deltas or {}
    members: dict[str, dict] = {}

    def _paired(variant: dict[int, float], primary: dict[int, float]) -> np.ndarray:
        keys = sorted(set(variant) & set(primary))
        return np.asarray([variant[k] - primary[k] for k in keys], dtype=float)

    det_1s_bac = {
        int(row["held_out_position"]): float(row["balanced_accuracy"])
        for row in det_1s["per_subject_secondary_local_only"].get(
            "ridge_logistic", [])}
    det_bipolar_bac = {
        int(row["held_out_position"]): float(row["balanced_accuracy"])
        for row in det_bipolar["per_subject_secondary_local_only"].get(
            "ridge_logistic", [])}
    detection_variants = {
        "detection_window_1s_vs_500ms_bac": det_1s_bac,
        "detection_bipolar_vs_referential_bac": det_bipolar_bac,
    }
    for name, variant in detection_variants.items():
        deltas = _paired(variant, primary_detection_bac)
        members[name] = {
            **_sign_flip_contrast(deltas),
            "direction_note": "balanced-accuracy contrast; positive favors the variant",
        }
    regression_variants = {
        "regression_window_1s_vs_500ms_delta_mae": "regression_window_1s",
        "regression_lag_subset_00_02s_vs_full_delta_mae":
            "regression_lag_subset_00_02s",
        "regression_band_group_low_vs_full_delta_mae":
            "regression_band_group_low",
        "regression_band_group_high_vs_full_delta_mae":
            "regression_band_group_high",
    }
    for name, key in regression_variants.items():
        variant_deltas = (out.get(key, {}) or {}).get("per_subject_delta_mae", {})
        deltas = _paired(
            {int(k): float(v) for k, v in variant_deltas.items()},
            {int(k): float(v) for k, v in primary_fewshot_deltas.items()})
        members[name] = {
            **_sign_flip_contrast(deltas),
            "direction_note": "delta-MAE contrast; negative favors the variant",
        }
    family_p = {name: member.get("sign_flip_p_raw_two_sided", 1.0)
                for name, member in members.items()}
    out["contrast_family"] = {
        "p_raw": family_p,
        "p_holm_within_family": holm_adjust(family_p) if family_p else {},
        "members": members,
        "rule": "two-sided sign-flip on paired per-subject contrasts; "
                "Holm within the ablation-contrast family",
    }
    return out


def _condition_on_tables(
    eligible: list[Recording], tables: list[FeatureTable],
    keep=None,
    fraction: float = contract.FEWSHOT_PRIMARY_FRACTION,
    workers: int = 1,
) -> dict:
    """The few-shot 30% condition on the given (canonicalized) tables,
    optionally restricted to a frozen column subset. Returns the
    per-subject paired delta-MAEs too (the ablation contrast family
    needs them; v2.1.0)."""
    if keep is not None:
        keep_columns = sorted({
            c for t in tables for c in t.features.columns
            if keep(c) or c == "coverage_frac"
        })
        tables = canonicalize_columns([
            _reindexed_table(t, [c for c in keep_columns
                                 if c in t.features.columns]) for t in tables])
    per_subject: list[SubjectConditionResult] = []
    n = len(eligible)
    envelopes = [_stitched_envelope(r) for r in eligible]

    global _PARALLEL_RECORDINGS, _PARALLEL_TABLES, _PARALLEL_ENVELOPES, _PARALLEL_BUILD_SPEC
    _PARALLEL_RECORDINGS = eligible
    _PARALLEL_TABLES = tables
    _PARALLEL_ENVELOPES = envelopes
    _PARALLEL_BUILD_SPEC = {"fraction": fraction}
    for result in _map_folds(_subset_condition_worker, n, workers):
        if result is not None:
            per_subject.append(result)
    aggregate = _aggregate_conditions(per_subject, "fewshot_30_condition")
    per_subject_deltas = {}
    for record in per_subject:
        if "combined" in record.mae and "ar_baseline" in record.mae:
            per_subject_deltas[record.held_out_position] = float(
                record.mae["combined"] - record.mae["ar_baseline"])
    return {"median_delta_mae": aggregate.get("median_mae_delta"),
            "median_innovation_r": aggregate.get("median_innovation_r"),
            "median_incremental_r2": aggregate.get("median_incremental_r2"),
            "per_subject_delta_mae": per_subject_deltas,
            "n_subjects": aggregate.get("n_subjects")}


def _serial_condition_on_tables(
    eligible: list[Recording], tables: list[FeatureTable],
    keep=None,
    fraction: float = contract.FEWSHOT_PRIMARY_FRACTION,
) -> dict:
    """Serial reference implementation of the ablation condition
    (kept for the determinism test): held-out span tables are built
    exactly as the parallel worker builds them (v2.1.0)."""
    from .regression import evaluate_fewshot_subject

    per_subject: list[SubjectConditionResult] = []
    n = len(eligible)
    envelopes = [_stitched_envelope(r) for r in eligible]
    for held_out in range(n):
        recording = eligible[held_out]
        envelope = envelopes[held_out]
        if envelope["stream"] is None:
            continue
        train_idx = [i for i in range(n) if i != held_out]
        train_streams = []
        train_keys = []
        for i in train_idx:
            stream = np.asarray(envelopes[i]["stream"], dtype=float)
            mu, sd = _move_constants(stream)
            train_streams.append((stream - mu) / sd)
            train_keys.append(np.asarray(envelopes[i]["block_keys"], dtype=int))
        ar_model = fit_ar_on_streams(train_streams, block_ids=train_keys)
        X_train_parts, y_train_parts, g_parts = [], [], []
        X_resid_parts, resid_parts, g_resid_parts = [], [], []
        for i in train_idx:
            t = tables[i]
            rows = np.isfinite(t.target)
            y_i = np.asarray(t.target)[rows]
            mu, sd = _move_constants(y_i)
            y_z = (y_i - mu) / sd
            stream_z = train_streams[train_idx.index(i)]
            keys_z = train_keys[train_idx.index(i)]
            ar_full = ar_predict(ar_model, stream_z, block_ids=keys_z)
            stream_pos = np.asarray(t.meta["stream_index"])[rows]
            resid_i = y_z - ar_full[stream_pos.astype(int)]
            resid_finite = np.isfinite(resid_i)  # NaN at AR block warm-ups
            flat = np.flatnonzero(rows)
            X_train_parts.append(t.features.iloc[flat])
            y_train_parts.append(y_z)
            g_parts.append(np.full(int(rows.sum()), i))
            X_resid_parts.append(t.features.iloc[flat[resid_finite]])
            resid_parts.append(resid_i[resid_finite])
            g_resid_parts.append(np.full(int(resid_finite.sum()), i))
        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = np.concatenate(y_train_parts)
        g_train = np.concatenate(g_parts)
        X_resid_train = pd.concat(X_resid_parts, ignore_index=True)
        resid_train = np.concatenate(resid_parts)
        g_resid_train = np.concatenate(g_resid_parts)
        alpha_fixed = select_alpha_inner(X_train, y_train, g_train)
        alpha_resid = select_alpha_inner(X_resid_train, resid_train, g_resid_train)
        residual_pipe = make_ridge_pipeline(alpha_resid)
        residual_pipe.fit(X_resid_train, resid_train)
        neural_pipe = make_ridge_pipeline(alpha_fixed)
        neural_pipe.fit(X_train, y_train)
        union_columns = list(dict.fromkeys(c for t in tables for c in t.features.columns))
        plan_primary = build_calibration_plan(recording.blocks,
                                              recording.n_samples,
                                              recording.fs)
        test_table = build_recording_tables(
            recording, window_s=contract.WINDOW_S_PRIMARY, with_lags=True,
            span=(plan_primary.test_start_sample, recording.n_samples))
        test_table = _reindexed_table(
            test_table, union_columns)
        test_rows = np.isfinite(test_table.target)
        plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                      recording.fs, fraction)
        calib_table = build_recording_tables(
            recording, window_s=contract.WINDOW_S_PRIMARY, with_lags=True,
            span=(0, plan.calib_end_sample))
        calib_table = _reindexed_table(
            calib_table,
            union_columns)
        calib_rows = np.isfinite(calib_table.target)
        if calib_rows.sum() < 10 or test_rows.sum() < 10:
            continue
        y_calib_raw = np.asarray(calib_table.target)[calib_rows]
        y_test_raw = np.asarray(test_table.target)[test_rows]
        mu, sd = _move_constants(y_calib_raw)
        per_subject.append(evaluate_fewshot_subject(
            held_out, "fewshot_30_condition",
            calib_table.features.loc[calib_rows].reset_index(drop=True),
            (y_calib_raw - mu) / sd,
            test_table.features.loc[test_rows].reset_index(drop=True),
            (y_test_raw - mu) / sd,
            test_table.meta["block"].values[test_rows], alpha_fixed, ar_model,
            zero_shot_residual_pipe=residual_pipe,
            zero_shot_neural_pipe=neural_pipe,
            with_xgboost=False,
            calib_block_ids=calib_table.meta["block"].values[calib_rows]))
    aggregate = _aggregate_conditions(per_subject, "fewshot_30_condition")
    return {"median_delta_mae": aggregate.get("median_mae_delta"),
            "median_innovation_r": aggregate.get("median_innovation_r"),
            "median_incremental_r2": aggregate.get("median_incremental_r2"),
            "n_subjects": aggregate.get("n_subjects")}


def _reindexed_table(table: FeatureTable, columns: list[str]) -> FeatureTable:
    """Reindex a table's features to an exact column set (cohort-uniform
    order). Columns the table lacks are filled 0.0 — the canonicalize
    convention: the paired missingness indicator already marks the
    absence explicitly (never a silent cross-channel median)."""
    return FeatureTable(
        features=table.features.reindex(columns=columns, fill_value=0.0).reset_index(drop=True),
        meta=table.meta.copy(), target=table.target,
        label=table.label, qc=table.qc)


# ---------------------------------------------------------------------------
# Primary family + verdicts
# ---------------------------------------------------------------------------


def _primary_family(track_a: dict, tracks_bc: RegressionArtifacts,
                    machinery: dict) -> dict:
    """The ONE primary Holm family (contract §13, v2.1.0): every
    endpoint contributes its RAW null p exactly once — the detection
    endpoint's model-family Holm p is never an input. Endpoints from
    tracks reported not-executable under the §14 machinery rule are
    excluded from the family (and counted), never silently failed."""
    detection = track_a["results"].get("ridge_logistic", {})
    zero_shot = tracks_bc.zero_shot
    fewshot30 = tracks_bc.fewshot.get(
        f"fewshot_{int(contract.FEWSHOT_PRIMARY_FRACTION*100)}", {})
    detection_p = (detection.get("null", {}) or {}).get("p_raw")
    zero_r_null = zero_shot.get("null_median_r_neural_only", {})
    delta_null = fewshot30.get("null_delta_mae_circular", {})
    # Gain endpoint: paired sign-flip p (the CI criterion is applied at
    # verdict time; the sign-flip p enters the family).
    gain_stats = []
    for curve_entry in tracks_bc.calibration_curve:
        if curve_entry["fraction"] == contract.FEWSHOT_PRIMARY_FRACTION:
            gain = curve_entry.get("gain_over_zero_shot_r", {})
            gain_stats.append(gain.get("sign_flip_p_raw"))
    gain_p = next((p for p in gain_stats if p is not None), 1.0)
    pvalues_all = {
        "detection_zero_shot_median_bac": detection_p if detection_p is not None else 1.0,
        "regression_zero_shot_median_r": zero_r_null.get("p_raw", 1.0),
        "fewshot30_delta_mae": delta_null.get("p_raw", 1.0),
        "fewshot30_gain_over_zero_shot": gain_p if gain_p is not None else 1.0,
    }
    excluded: list[str] = []
    if not machinery["detection"]["executable"]:
        excluded.append("detection_zero_shot_median_bac")
    if not machinery["regression"]["executable"]:
        excluded.extend(["regression_zero_shot_median_r", "fewshot30_delta_mae",
                         "fewshot30_gain_over_zero_shot"])
    pvalues = {name: p for name, p in pvalues_all.items() if name not in excluded}
    holm = holm_adjust(pvalues)
    return {"p_raw": pvalues_all, "p_raw_in_family": pvalues,
            "p_holm": holm,
            "excluded_not_executable": excluded,
            "rule": contract.MULTIPLICITY_RULE,
            "single_adjustment": contract.PRIMARY_FAMILY_SINGLE_ADJUSTMENT}


def _verdicts(track_a: dict, tracks_bc: RegressionArtifacts, track_d: dict,
              primary_family: dict, machinery: dict) -> dict:
    """Every displayed verdict derives from the FINAL adjusted result
    (v2.1.0): a primary verdict requires its effect criterion, its
    subject-cluster bootstrap CI, AND its primary-family Holm p < 0.05 —
    so per-track verdicts can never contradict ``global_null``. Event
    verdicts use the event p-value family's Holm adjustment."""
    holm = primary_family["p_holm"]

    def _holm_ok(name: str) -> bool:
        p = holm.get(name)
        return p is not None and p < 0.05

    det_exec = machinery["detection"]["executable"]
    reg_exec = machinery["regression"]["executable"]
    events_exec = machinery["events"]["executable"]

    detection = track_a["results"].get("ridge_logistic", {})
    detection_verdict = detection.get("verdict", {})
    detection_ok = bool(det_exec and detection_verdict.get("criterion_met")
                        and _holm_ok("detection_zero_shot_median_bac"))

    zero_shot = tracks_bc.zero_shot
    r_ci = zero_shot.get("bootstrap_ci_median_r_neural_only", (np.nan, np.nan))
    median_r = zero_shot.get("median_r_neural_only", np.nan)
    zero_r_ok = bool(
        reg_exec and median_r is not None and np.isfinite(median_r)
        and median_r > 0
        and r_ci and np.isfinite(r_ci[0]) and r_ci[0] > 0
        and _holm_ok("regression_zero_shot_median_r")
    )
    fewshot30 = tracks_bc.fewshot.get(
        f"fewshot_{int(contract.FEWSHOT_PRIMARY_FRACTION*100)}", {})
    delta = fewshot30.get("median_mae_delta")
    delta_ci = fewshot30.get("bootstrap_ci_median_delta_mae", (np.nan, np.nan))
    delta_ok = bool(
        reg_exec and delta is not None and np.isfinite(delta) and delta < 0
        and delta_ci and np.isfinite(delta_ci[1]) and delta_ci[1] < 0
        and _holm_ok("fewshot30_delta_mae")
    )
    gain_curve = next((c for c in tracks_bc.calibration_curve
                       if c["fraction"] == contract.FEWSHOT_PRIMARY_FRACTION), {})
    gain = gain_curve.get("gain_over_zero_shot_r", {})
    gain_ok = bool(
        reg_exec and gain.get("status") == "ok"
        and gain.get("bootstrap_ci", [np.nan, np.nan])[0] > 0
        and _holm_ok("fewshot30_gain_over_zero_shot")
    )
    # Track D: ONE event p-value family, adjusted once (v2.1.0).
    event_holm = (track_d.get("event_family", {}) or {}).get(
        "p_holm_within_family", {})

    def _event_holm_ok(name: str) -> bool:
        p = event_holm.get(name)
        return p is not None and p < 0.05

    assoc = track_d.get("primary_association", {})
    assoc_ok = bool(
        events_exec and assoc.get("status") == "ok"
        and assoc.get("bootstrap_ci_fisher_z_r", [np.nan, np.nan])[0] > 0
        and _event_holm_ok("log_inter_tap_interval_association")
    )
    event_pred = track_d.get("event_prediction", {})
    pred_ci = event_pred.get("bootstrap_ci_median_delta_mae",
                             [np.nan, np.nan])
    event_pred_ok = bool(
        events_exec and event_pred.get("status") == "ok"
        and event_pred.get("observed_median_delta_mae", 0) < 0
        and pred_ci and np.isfinite(pred_ci[1]) and pred_ci[1] < 0
        and _event_holm_ok("event_prediction_shift_null")
    )
    any_primary = bool(detection_ok or zero_r_ok or delta_ok or gain_ok)
    return {
        "detection_transfer": {
            "verdict": detection_ok,
            "criterion_met": bool(detection_verdict.get("criterion_met")),
            "median_bac": detection_verdict.get("median_bac"),
            "ci": detection_verdict.get("ci"),
            "holm_p": holm.get("detection_zero_shot_median_bac"),
            "not_executable_machinery": not det_exec,
        },
        "regression_zero_shot": {
            "verdict": zero_r_ok,
            "median_r": median_r,
            "bootstrap_ci": list(r_ci) if r_ci else None,
            "holm_p": holm.get("regression_zero_shot_median_r"),
            "not_executable_machinery": not reg_exec,
        },
        "fewshot_delta_mae": {
            "verdict": delta_ok,
            "median_delta_mae": delta,
            "bootstrap_ci": list(delta_ci) if delta_ci else None,
            "holm_p": holm.get("fewshot30_delta_mae"),
            "not_executable_machinery": not reg_exec,
        },
        "fewshot_gain": {
            "verdict": gain_ok,
            "median_gain_r": gain.get("median_gain"),
            "bootstrap_ci": gain.get("bootstrap_ci"),
            "holm_p": holm.get("fewshot30_gain_over_zero_shot"),
            "not_executable_machinery": not reg_exec,
        },
        "event_association": {"verdict": assoc_ok,
                              "holm_p": event_holm.get(
                                  "log_inter_tap_interval_association"),
                              "not_executable_machinery": not events_exec,
                              "detail": {k: v for k, v in assoc.items()
                                         if k != "n_resamples"}},
        "event_prediction": {"verdict": event_pred_ok,
                             "holm_p": event_holm.get(
                                 "event_prediction_shift_null"),
                             "not_executable_machinery": not events_exec,
                             "detail": {"observed_median_delta_mae":
                                        event_pred.get("observed_median_delta_mae"),
                                        "bootstrap_ci_median_delta_mae":
                                        event_pred.get(
                                            "bootstrap_ci_median_delta_mae"),
                                        "null_p_raw":
                                        (event_pred.get("null", {}) or {}).get("p_raw")}},
        "machinery_executable": {name: status["executable"]
                                 for name, status in machinery.items()},
        "machinery_pass_fractions": {name: status["pass_fraction"]
                                     for name, status in machinery.items()},
        "stopping_rule": contract.STOPPING_RULES["machinery_failure"],
        "global_null": bool(not any_primary),
        "stage": contract.STAGE,
        "verdict_rule": (
            "every verdict = criterion + bootstrap CI + final Holm-adjusted "
            "p (primary family for the four primary endpoints; the single "
            "event family for the event claims); per-track verdicts are "
            "therefore consistent with global_null by construction"),
        "caveat": (
            "all verdicts are exploratory at this cohort size; statistical "
            "evidence is not practical usefulness; subject heterogeneity is "
            "reported regardless of aggregate outcome"
        ),
    }

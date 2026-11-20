"""Parallel-runner determinism: ``workers > 1`` must produce output
IDENTICAL to the serial path — not merely on selected objects but on
the COMPLETE scientific output set (full study artifacts and the
committable summary JSON). All seeds are fixed module constants, so
worker scheduling cannot change any result.

The ONLY nondeterministic fields in a real run live in the DRIVER layer
(``generated_utc`` in the summary and the ``provenance`` block: wall
clock, duration, worker count) — verified here by removing exactly
those fields and demanding byte-identical JSON of everything else.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.transfer.detection import loso_detection_cv
from curve_icon_dbso.transfer.pipeline import (
    build_recording_tables,
    canonicalize_columns,
    run_tracks_bc,
    run_transfer_study,
)
from curve_icon_dbso.transfer.summary import build_run_summary
from curve_icon_dbso.transfer.synthetic import make_fast_cohort


def _canonical(obj):
    """Normalize numpy containers/scalars and NaNs so two independently
    built artifact trees can be compared byte-for-byte."""
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(
            obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(dataclasses.asdict(obj))
    return obj


def _json_bytes(obj) -> str:
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"))


@pytest.mark.slow
def test_parallel_full_study_output_set_is_identical() -> None:
    """The complete StudyArtifacts tree — every track, condition,
    family, verdict, and the cohort flow — is byte-identical between the
    serial and the spawned-parallel path."""
    cohort = make_fast_cohort(n=3, seed0=9950, plant="innovation")
    kwargs = dict(with_ablations=True, null_draws=8, with_xgboost=False)
    serial = run_transfer_study(cohort, workers=1, **kwargs)
    parallel = run_transfer_study(cohort, workers=2, **kwargs)
    assert _json_bytes(serial) == _json_bytes(parallel)

    # The committable summary (everything except the driver's two
    # documented nondeterministic fields) is byte-identical too.
    serial_summary = build_run_summary(serial)
    parallel_summary = build_run_summary(parallel)
    assert "generated_utc" in serial_summary
    serial_summary.pop("generated_utc")
    parallel_summary.pop("generated_utc")
    assert _json_bytes(serial_summary) == _json_bytes(parallel_summary)


@pytest.mark.slow
def test_parallel_tracks_bc_and_detection_folds_match_serial() -> None:
    cohort = make_fast_cohort(n=4, seed0=9900, plant="innovation")
    tables = canonicalize_columns(
        [build_recording_tables(r, with_lags=True) for r in cohort])

    serial_bc = run_tracks_bc(cohort, tables, null_draws=10, with_xgboost=False,
                              workers=1)
    parallel_bc = run_tracks_bc(cohort, tables, null_draws=10,
                                with_xgboost=False, workers=2)
    assert _json_bytes(serial_bc) == _json_bytes(parallel_bc)

    det_tables = canonicalize_columns(
        [build_recording_tables(r, with_lags=False) for r in cohort])
    serial_det, serial_models = loso_detection_cv(det_tables, workers=1)
    parallel_det, parallel_models = loso_detection_cv(det_tables, workers=2)
    for name in serial_det:
        a = serial_det[name].per_subject.sort_values("held_out_position")
        b = parallel_det[name].per_subject.sort_values("held_out_position")
        pd.testing.assert_frame_equal(a, b)
        assert serial_det[name].median_bac == parallel_det[name].median_bac
        assert serial_det[name].null == parallel_det[name].null
    # fold models score identically on a probe matrix
    probe = det_tables[0].features.iloc[:10]
    for name, models in serial_models.items():
        for m_serial, m_parallel in zip(models, parallel_models[name],
                                        strict=True):
            np.testing.assert_array_equal(m_serial.predict(probe),
                                          m_parallel.predict(probe))

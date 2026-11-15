"""Kinematic extraction on SYNTHETIC fixtures only (no real data involved).

Fixtures plant a known tap frequency in glove MOVE blocks and a known HFO
tone in ECoG REST blocks, then verify the prespecified extraction recovers
them, applies the prominence floor, and fails QC explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso.errors import DataValidationError
from curve_icon_dbso.kinematics import (
    KINEMATIC_FEATURES,
    KINEMATIC_TARGET,
    build_kinematic_model_matrix,
    extract_kinematic_recording,
    parse_restmove_blocks,
    resolve_output_dir,
    tap_rate_for_recording,
)

FS_GLOVE = 100.0
FS_ECOG = 1200.0
TAP_HZ = 2.0
HFO_HZ = 250.0


def make_recording(n_glove=5, flat_glove_channels=1, n_ecog=4, duration_s=8.0, seed=0):
    """REST/MOVE/REST layout in the shared sample clock (glove rate); MOVE
    blocks carry a planted tap sine on all but `flat_glove_channels` glove
    channels; the ECoG REST spans carry a planted 250 Hz tone on every
    channel. ECoG is sampled at FS_ECOG over the same wall-clock duration
    and sliced with the clamped shared-clock bounds (as the real pipeline
    does for preprocessed per-recording files)."""
    rng = np.random.default_rng(seed)
    n_glove_samples = int(duration_s * FS_GLOVE)
    quarter = n_glove_samples // 4
    # Real layout: >=4 columns, kind flag in column 4 (index 3).
    segs = np.array([
        [0, quarter, 0, 0],
        [quarter, 3 * quarter, 0, 1],
        [3 * quarter, n_glove_samples, 0, 0],
    ])
    t_glove = np.arange(n_glove_samples) / FS_GLOVE
    glove = rng.normal(0.0, 0.05, (n_glove, n_glove_samples))
    move = slice(quarter, 3 * quarter)
    for ch in range(n_glove - flat_glove_channels):
        glove[ch, move] += np.sin(2 * np.pi * TAP_HZ * t_glove[move])
    # "Flat"/non-tapping fingers: diffuse off-bin jitter (four equal sines at
    # 0.9/1.9/2.9/3.9 Hz). With the prespecified 1 s Welch windows (1 Hz
    # resolution) each bin carries ~12% of in-band power, below the 0.20
    # prominence floor — unlike white noise, whose ~4 in-band bins make even
    # a featureless channel peak above the floor.
    for ch in range(n_glove - flat_glove_channels, n_glove):
        for jitter_hz in (0.9, 1.9, 2.9, 3.9):
            glove[ch, move] += 0.4 * np.sin(2 * np.pi * jitter_hz * t_glove[move])

    n_ecog_samples = int(duration_s * FS_ECOG)
    t_ecog = np.arange(n_ecog_samples) / FS_ECOG
    ecog = rng.normal(0.0, 5e-6, (n_ecog, n_ecog_samples))
    for start, end, _, kind in segs:
        if kind == 0:  # plant the HFO tone in every REST span, above the noise
            for ch in range(n_ecog):
                ecog[ch, start:end] += 1e-5 * np.sin(2 * np.pi * HFO_HZ * t_ecog[start:end])
    return ecog, glove, segs


# --- parsing -----------------------------------------------------------------

def test_parse_restmove_blocks_kinds_and_bounds():
    segs = np.array([[0, 10, 0, 0], [10, 20, 0, 1]])
    assert parse_restmove_blocks(segs) == [(0, 10, "rest"), (10, 20, "move")]
    with pytest.raises(DataValidationError, match="block-kind flag"):
        parse_restmove_blocks(np.array([[0, 10, 0, 2]]))
    with pytest.raises(DataValidationError, match="bounds invalid"):
        parse_restmove_blocks(np.array([[10, 10, 0, 1]]))
    with pytest.raises(DataValidationError, match="n_blocks"):
        parse_restmove_blocks(np.array([[0, 10]]))


# --- tap rate ----------------------------------------------------------------

def test_tap_rate_recovers_planted_frequency():
    _, glove, segs = make_recording(flat_glove_channels=0)
    result = tap_rate_for_recording(glove, FS_GLOVE, segs)
    assert result["tap_qc_status"] == "ok"
    # 5 oscillating channels x 1 MOVE block contribute.
    assert result["n_tap_contributing"] == 5
    # 1 s Welch windows give 1 Hz resolution; the planted 2 Hz peak is exact.
    assert result["tap_rate_mean"] == pytest.approx(TAP_HZ, abs=0.2)


def test_prominence_floor_branch_excludes_below_floor_channels(monkeypatch):
    """The floor's exclusion branch, verified with a raised (test-only)
    threshold: the prespecified production floor is 0.20 and must not be
    retuned by tests or code. See the methodology caveat on binning."""
    import curve_icon_dbso.kinematics as kin

    _, glove, segs = make_recording(flat_glove_channels=0)
    results_at_default = kin.tap_rate_for_recording(glove, FS_GLOVE, segs)
    assert results_at_default["n_tap_contributing"] == 5  # on-bin taps pass 0.20

    monkeypatch.setattr(kin, "TAP_PROMINENCE_FLOOR", 0.9)
    results_strict = kin.tap_rate_for_recording(glove, FS_GLOVE, segs)
    assert results_strict["tap_qc_status"] == "failed"
    assert "prominence floor" in results_strict["tap_qc_reason"]
    assert results_strict["tap_rate_mean"] is None


def test_degenerate_channel_is_never_counted_as_a_tap():
    from curve_icon_dbso.kinematics import _channel_tap_rate

    # A zero channel has no in-band power at all: no tap, no fake zero.
    assert _channel_tap_rate(np.zeros(400), FS_GLOVE) is None


def test_qc_fails_when_fewer_than_two_channel_blocks_contribute():
    """A single contributing channel-block cannot satisfy the >=2 QC rule."""
    _, glove, segs = make_recording(n_glove=1, flat_glove_channels=0)
    result = tap_rate_for_recording(glove, FS_GLOVE, segs)
    assert result["tap_qc_status"] == "failed"
    assert result["tap_rate_mean"] is None
    assert result["n_tap_contributing"] == 1
    assert "2 required" in result["tap_qc_reason"]


def test_qc_fails_explicitly_without_move_blocks_or_with_bad_glove():
    _, glove, _ = make_recording()
    rest_only = np.array([[0, 100, 0, 0]])
    result = tap_rate_for_recording(glove, FS_GLOVE, rest_only)
    assert result["tap_qc_status"] == "failed"
    assert "no MOVE blocks" in result["tap_qc_reason"]

    short_segs = np.array([[0, 50, 0, 0], [50, 100, 0, 1]])
    nan_glove = glove.copy()
    nan_glove[0, 5] = np.nan
    result = tap_rate_for_recording(nan_glove, FS_GLOVE, short_segs)
    assert result["tap_qc_status"] == "failed"
    assert "non-finite" in result["tap_qc_reason"]


def test_sampling_rate_violation_raises():
    _, glove, segs = make_recording()
    with pytest.raises(DataValidationError, match="Nyquist"):
        tap_rate_for_recording(glove, fs_glove=6.0, restmove_segs=segs)  # Nyquist 3 Hz < 4 Hz
    with pytest.raises(DataValidationError, match="positive finite"):
        tap_rate_for_recording(glove, fs_glove=float("nan"), restmove_segs=segs)


# --- full recording extraction ----------------------------------------------

def test_extract_kinematic_recording_recovers_rest_hfo_state():
    ecog, glove, segs = make_recording()
    result = extract_kinematic_recording(ecog, glove, FS_ECOG, FS_GLOVE, segs)
    assert result["tap_qc_status"] == "ok"
    assert result["n_rest_ok_records"] == 8  # 2 REST blocks x 4 channels
    assert result["n_rest_failed_records"] == 0
    assert result["hfo_power_rest_mean"] is not None and result["hfo_power_rest_mean"] > 0
    # Planted 250 Hz tone dominates every REST in-band PSD (200-sample Welch
    # segments at 1200 Hz give 6 Hz resolution).
    assert result["hfo_peak_freq_rest_mean"] == pytest.approx(HFO_HZ, abs=8.0)


def test_extract_kinematic_recording_is_deterministic():
    ecog, glove, segs = make_recording(seed=3)
    first = extract_kinematic_recording(ecog, glove, FS_ECOG, FS_GLOVE, segs)
    second = extract_kinematic_recording(ecog, glove, FS_ECOG, FS_GLOVE, segs)
    assert first == second


# --- model matrix ------------------------------------------------------------

def test_kinematic_model_matrix_is_exactly_the_five_prespecified_features():
    frame = pd.DataFrame({
        **{name: [1.0, 2.0, 3.0] for name in KINEMATIC_FEATURES},
        KINEMATIC_TARGET: [1.5, 2.0, 2.5],
        "subject": ["S1", "S2", "S3"],
        "extra_column_that_must_not_leak": [9.0, 9.0, 9.0],
    })
    x, y, groups = build_kinematic_model_matrix(frame)
    assert list(x.columns) == KINEMATIC_FEATURES
    assert len(KINEMATIC_FEATURES) == 5
    assert "extra_column_that_must_not_leak" not in x.columns
    assert y.tolist() == [1.5, 2.0, 2.5]
    assert groups.tolist() == ["S1", "S2", "S3"]


def test_kinematic_model_matrix_rejects_missing_columns_and_bad_target():
    frame = pd.DataFrame({name: [1.0] for name in KINEMATIC_FEATURES[:4]})
    with pytest.raises(DataValidationError, match="missing required columns"):
        build_kinematic_model_matrix(frame)
    bad = pd.DataFrame({
        **{name: [1.0] for name in KINEMATIC_FEATURES},
        KINEMATIC_TARGET: ["not-numeric"],
        "subject": ["S1"],
    })
    with pytest.raises(DataValidationError, match="must be numeric"):
        build_kinematic_model_matrix(bad)


# --- output-location guard (real-data acknowledgement rule) -------------------

def test_resolve_output_dir_requires_explicit_outside_path(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(DataValidationError, match="explicit output directory"):
        resolve_output_dir(None, repo_root)
    with pytest.raises(DataValidationError, match="inside the repository"):
        resolve_output_dir(repo_root / "outputs" / "real", repo_root)
    with pytest.raises(DataValidationError, match="inside the repository"):
        resolve_output_dir(repo_root, repo_root)
    # The deliberate override is allowed but must be explicit.
    inside = resolve_output_dir(repo_root / "outputs", repo_root, allow_in_repo=True)
    assert inside == (repo_root / "outputs").resolve()
    outside = resolve_output_dir(tmp_path / "outside" / "results", repo_root)
    assert outside == (tmp_path / "outside" / "results").resolve()

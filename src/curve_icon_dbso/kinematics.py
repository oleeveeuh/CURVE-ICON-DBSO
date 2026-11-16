"""Kinematic feature extraction: sensor-glove tap rate and REST-block HFO state.

Implements the prespecified kinematic analysis in ``docs/methodology.md``
("Kinematic analysis (active; prespecified)"): the finger-tapping-rate
target and the five prespecified predictors. This is **not** DBS-outcome
prediction — glove kinematics measure motor state during the recording
session, not treatment response.

Prespecification contract (frozen before any real-data model fit; see the
methodology's November 2026 status note):

* Target ``tap_rate_mean`` — median dominant tapping frequency (Hz) over
  contributing glove channel-blocks during MOVE blocks.
* A channel-block contributes only if the dominant peak carries
  ``>= TAP_PROMINENCE_FLOOR`` (0.20) of the in-band (0.5–4.0 Hz) PSD — a
  prespecified prominence floor that keeps flat, non-oscillating fingers
  from counting as taps.
* A recording passes QC only if ``>= TAP_MIN_CHANNEL_BLOCKS`` (2)
  channel-blocks contribute; otherwise it is excluded from modeling with
  an explicit QC record — never imputed silently, never zero-filled.
* Predictors ``hfo_power_rest_mean`` / ``hfo_peak_freq_rest_mean`` come
  from REST blocks only (cross-state design), averaged over blocks and
  QC-OK ECoG channels, via :mod:`curve_icon_dbso.hfo`.

Every function here is pure computation on arrays; verification uses
synthetic fixtures only. The external-data entry point
(``scripts/evaluate_real_kinematics.py``) is compliance-gated and must
not be run until a dataset is recorded ``VERIFIED`` in the
``docs/data_access.md`` compliance register.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

from .config import HFOConfig
from .errors import DataValidationError

KINEMATIC_FEATURES: list[str] = [
    "hfo_power_rest_mean",
    "hfo_peak_freq_rest_mean",
    "age",
    "dbs_volt",
    "dbs_freq",
]
KINEMATIC_TARGET = "tap_rate_mean"
KINEMATIC_GROUP = "subject"

TAP_RATE_LOW_HZ = 0.5
TAP_RATE_HIGH_HZ = 4.0
TAP_PROMINENCE_FLOOR = 0.20
TAP_MIN_CHANNEL_BLOCKS = 2
TAP_WELCH_SEGMENT_SECONDS = 1.0

# restmove_segs layout (methodology): columns 1-2 are start/end samples
# (0-based half-open), column 4 is the block kind (0 = REST, 1 = MOVE).
_SEG_START, _SEG_END, _SEG_KIND = 0, 1, 3


def parse_restmove_blocks(segs: np.ndarray) -> list[tuple[int, int, str]]:
    """Parse the ``restmove_segs`` table into ``(start, end, kind)`` tuples.

    Bounds are 0-based half-open. Kinds other than 0 (REST) / 1 (MOVE) are
    rejected loudly; end <= start is rejected as malformed.
    """
    segs = np.asarray(segs)
    if segs.ndim != 2 or segs.shape[1] < 4:
        raise DataValidationError(
            f"restmove_segs must be (n_blocks, >=4); got {segs.shape}"
        )
    blocks: list[tuple[int, int, str]] = []
    for row in segs:
        start, end = int(row[_SEG_START]), int(row[_SEG_END])
        kind_flag = int(row[_SEG_KIND])
        if kind_flag == 1:
            kind = "move"
        elif kind_flag == 0:
            kind = "rest"
        else:
            raise DataValidationError(
                f"restmove_segs block-kind flag must be 0 (REST) or 1 (MOVE), got "
                f"{kind_flag}"
            )
        if end <= start or start < 0:
            raise DataValidationError(
                f"restmove_segs block bounds invalid: start={start}, end={end} "
                "(expected 0 <= start < end)"
            )
        blocks.append((start, end, kind))
    return blocks


def _channel_tap_rate(channel: np.ndarray, fs: float) -> tuple[float, float] | None:
    """Single-channel tap rate + prominence share, or ``None`` below the floor."""
    detrended = detrend(channel)
    nperseg = min(int(TAP_WELCH_SEGMENT_SECONDS * fs), detrended.size)
    freqs, psd = welch(detrended, fs=fs, window="hamming", nperseg=nperseg,
                       noverlap=nperseg // 2)
    in_band = (freqs >= TAP_RATE_LOW_HZ) & (freqs <= TAP_RATE_HIGH_HZ)
    if not in_band.any():
        return None
    band_psd = psd[in_band]
    total = float(band_psd.sum())
    if total <= 0.0:
        return None
    peak_idx = int(np.argmax(band_psd))
    share = float(band_psd[peak_idx]) / total
    if share < TAP_PROMINENCE_FLOOR:
        return None
    return float(freqs[in_band][peak_idx]), share


def tap_rate_for_recording(
    glove: np.ndarray, fs_glove: float, restmove_segs: np.ndarray
) -> dict:
    """Compute ``tap_rate_mean`` and its QC record for one recording.

    Glove is ``(n_channels, n_samples)``. Contributions are collected over
    all MOVE blocks x channels; the target is their median. QC fails
    (target ``None``, explicit reason) when fewer than
    ``TAP_MIN_CHANNEL_BLOCKS`` channel-blocks contribute, when there are no
    MOVE blocks, or when the glove data are non-finite/empty.
    """
    glove = np.asarray(glove, dtype=float)
    blocks = parse_restmove_blocks(restmove_segs)
    move_blocks = [(s, e) for s, e, kind in blocks if kind == "move"]

    if glove.ndim != 2 or glove.shape[0] == 0:
        return {"tap_rate_mean": None, "n_tap_contributing": 0,
                "tap_qc_status": "failed", "tap_qc_reason": "glove data missing or not 2-D"}
    if not np.isfinite(glove).all():
        return {"tap_rate_mean": None, "n_tap_contributing": 0,
                "tap_qc_status": "failed", "tap_qc_reason": "non-finite glove samples"}
    if not move_blocks:
        return {"tap_rate_mean": None, "n_tap_contributing": 0,
                "tap_qc_status": "failed", "tap_qc_reason": "no MOVE blocks in restmove_segs"}
    if not np.isfinite(fs_glove) or fs_glove <= 0:
        raise DataValidationError(
            f"Glove sampling rate must be a positive finite number, got {fs_glove!r}"
        )
    if TAP_RATE_HIGH_HZ >= fs_glove / 2.0:
        raise DataValidationError(
            f"Tap-rate band edge {TAP_RATE_HIGH_HZ} Hz is not below the glove "
            f"Nyquist frequency ({fs_glove / 2.0} Hz)"
        )

    contributions: list[float] = []
    for start, end in move_blocks:
        end = min(end, glove.shape[1])
        segment = glove[:, start:end]
        for channel in segment:
            result = _channel_tap_rate(channel, fs_glove)
            if result is not None:
                contributions.append(result[0])

    n = len(contributions)
    if n < TAP_MIN_CHANNEL_BLOCKS:
        return {
            "tap_rate_mean": None,
            "n_tap_contributing": n,
            "tap_qc_status": "failed",
            "tap_qc_reason": (
                f"only {n} channel-block(s) reached the "
                f"{TAP_PROMINENCE_FLOOR:.2f} prominence floor; "
                f"{TAP_MIN_CHANNEL_BLOCKS} required"
            ),
        }
    return {
        "tap_rate_mean": float(np.median(contributions)),
        "n_tap_contributing": n,
        "tap_qc_status": "ok",
        "tap_qc_reason": "",
    }


def extract_kinematic_recording(
    ecog: np.ndarray,
    glove: np.ndarray,
    fs_ecog: float,
    fs_glove: float,
    restmove_segs: np.ndarray,
    band: HFOConfig | None = None,
) -> dict:
    """All per-recording kinematic-design quantities for one recording.

    REST-block HFO features average the per-channel
    :func:`~curve_icon_dbso.hfo.extract_channel_features` results over all
    REST blocks and QC-OK channels (cross-state design: REST HFO state as
    predictor, MOVE tapping as target). Failed channel-blocks are counted,
    never zero-filled.

    Returns a dict with ``tap_rate_mean``, ``n_tap_contributing``,
    ``tap_qc_status``, ``tap_qc_reason``, ``hfo_power_rest_mean``,
    ``hfo_peak_freq_rest_mean``, ``n_rest_ok_records``,
    ``n_rest_failed_records``.
    """
    from .hfo import extract_channel_features

    band = band or HFOConfig()
    result = tap_rate_for_recording(glove, fs_glove, restmove_segs)

    ecog = np.asarray(ecog, dtype=float)
    blocks = parse_restmove_blocks(restmove_segs)
    rest_blocks = [(s, e) for s, e, kind in blocks if kind == "rest"]

    powers: list[float] = []
    peaks: list[float] = []
    n_failed = 0
    if ecog.ndim == 2 and np.isfinite(ecog).all():
        for start, end in rest_blocks:
            end = min(end, ecog.shape[1])
            for channel in ecog[:, start:end]:
                record = extract_channel_features(channel, fs_ecog, band)
                if record["qc_status"] == "ok":
                    powers.append(float(record["hfo_power"]))
                    peaks.append(float(record["hfo_peak_freq"]))
                else:
                    n_failed += 1
    else:
        n_failed = -1  # recording-level problem; features stay None

    result.update({
        "hfo_power_rest_mean": float(np.mean(powers)) if powers else None,
        "hfo_peak_freq_rest_mean": float(np.mean(peaks)) if peaks else None,
        "n_rest_ok_records": len(powers),
        "n_rest_failed_records": n_failed,
    })
    return result


def build_kinematic_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the prespecified kinematic design ``(X, y, groups)``.

    Exactly the five prespecified predictors, by name — no extra columns,
    no identifiers, no target leakage by construction. QC-failed recordings
    (``tap_qc_status != "ok"``) must be excluded by the caller, which keeps
    every exclusion explicit; this function validates instead that the
    columns exist and the target is numeric.
    """
    required = KINEMATIC_FEATURES + [KINEMATIC_TARGET, KINEMATIC_GROUP]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Kinematic model matrix is missing required columns: {missing}. "
            f"Available columns: {sorted(map(str, df.columns))}"
        )
    if not pd.api.types.is_numeric_dtype(df[KINEMATIC_TARGET]):
        raise DataValidationError(
            f"Target column '{KINEMATIC_TARGET}' must be numeric, got "
            f"{df[KINEMATIC_TARGET].dtype}"
        )
    x = df[KINEMATIC_FEATURES].astype(float)
    y = df[KINEMATIC_TARGET].astype(float)
    groups = df[KINEMATIC_GROUP].astype(str)
    return x, y, groups


def resolve_output_dir(
    output_dir: str | Path | None,
    repo_root: Path | None,
    allow_in_repo: bool = False,
) -> Path:
    """Enforce the real-data output-location rule (docs/data_access.md).

    Real-data scripts must default outputs **outside** the repository:
    ``output_dir`` is required (no implicit default), and a path inside the
    repository checkout is rejected unless ``allow_in_repo=True`` is passed
    as the deliberate override.
    """
    if output_dir is None:
        raise DataValidationError(
            "An explicit output directory is required for real-data runs, and it "
            "must default outside the repository checkout (docs/data_access.md)."
        )
    path = Path(output_dir).expanduser().resolve()
    if repo_root is not None and not allow_in_repo:
        root = Path(repo_root).resolve()
        if path == root or root in path.parents:
            raise DataValidationError(
                f"Refusing to write real-data outputs inside the repository "
                f"({path}). Choose a directory outside the checkout, or pass the "
                "explicit in-repository override if you truly intend this."
            )
    return path

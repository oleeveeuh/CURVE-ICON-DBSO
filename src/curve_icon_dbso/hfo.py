"""HFO feature extraction from intracranial ECoG recordings.

Scientific scope — what this code actually computes, per channel:

* ``hfo_power``: integral of the Welch power spectral density over the
  HFO band (units: signal^2).
* ``hfo_peak_freq``: frequency of the PSD maximum within the band (Hz).
* QC fields: per-channel extraction status and failure reason.

That is the complete list. Phase-amplitude coupling, artifact rejection,
signal-to-noise ratios, temporal-stability metrics and spatial
distribution summaries are **not implemented** and must not be claimed.

Integrity rules (corrective to the historical script):

* Missing input data raises :class:`MissingDataError`. There is no
  random or zero-filled fallback anywhere in this module.
* A failed channel extraction is recorded with ``qc_status="failed"``
  and ``None`` feature values — never as a true zero measurement.
* The sampling rate must be a positive finite number and the upper band
  edge must be strictly below Nyquist; violations raise
  :class:`SamplingRateError`.
* All processing is parameterized; there are no hardcoded paths.

File-selection behavior: every ``*.mat`` file directly inside each
subject subdirectory is processed, in lexicographic order. Channel-level
results keep the originating file so selection is auditable.
Subject-level aggregation is a separate, explicit step
(:func:`aggregate_to_subject`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch

from .config import HFOConfig
from .errors import DataValidationError, MissingDataError, SamplingRateError

FILTER_ORDER = 4
WELCH_SEGMENT_SECONDS = 1.0


def validate_sampling_rate(fs: float, band: HFOConfig) -> None:
    """Validate a sampling rate against the analysis band.

    Raises:
        SamplingRateError: If ``fs`` is not a positive finite number, or
            if the upper band edge is not strictly below Nyquist.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise SamplingRateError(f"Sampling rate must be a positive finite number, got {fs!r}")
    nyquist = fs / 2.0
    if band.high_hz >= nyquist:
        raise SamplingRateError(
            f"Upper band edge {band.high_hz} Hz is not strictly below Nyquist "
            f"({nyquist} Hz for fs={fs} Hz). Lower the band edge or increase fs."
        )
    if band.low_hz >= nyquist:
        raise SamplingRateError(
            f"Lower band edge {band.low_hz} Hz is not below Nyquist ({nyquist} Hz for fs={fs} Hz)."
        )


def extract_channel_features(
    signal: np.ndarray, fs: float, band: HFOConfig | None = None
) -> dict:
    """Extract HFO features from a single channel of a recording.

    Args:
        signal: 1-D array of samples. NaN/inf-contaminated or empty
            signals produce a ``failed`` QC status, not zeros.
        fs: Sampling rate in Hz (validated).
        band: HFO band definition (default 80-500 Hz).

    Returns:
        Dict with keys ``hfo_power`` (float or None), ``hfo_peak_freq``
        (float or None), ``qc_status`` (``"ok"`` or ``"failed"``) and
        ``qc_reason`` (empty string when ok).
    """
    band = band or HFOConfig()
    validate_sampling_rate(fs, band)

    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size == 0:
        return _failed("empty signal")
    if not np.isfinite(signal).all():
        return _failed("non-finite samples (NaN/inf)")

    nyquist = fs / 2.0
    b, a = butter(FILTER_ORDER, [band.low_hz / nyquist, band.high_hz / nyquist], btype="band")
    filtered = filtfilt(b, a, signal)

    nperseg = min(int(WELCH_SEGMENT_SECONDS * fs), signal.size)
    frequencies, psd = welch(filtered, fs=fs, window="hamming", nperseg=nperseg,
                             noverlap=nperseg // 2)
    in_band = (frequencies >= band.low_hz) & (frequencies <= band.high_hz)
    if not in_band.any():
        return _failed("no PSD bins inside the analysis band")

    power = float(np.trapezoid(psd[in_band], frequencies[in_band]))
    peak_freq = float(frequencies[in_band][int(np.argmax(psd[in_band]))])
    return {
        "hfo_power": power,
        "hfo_peak_freq": peak_freq,
        "qc_status": "ok",
        "qc_reason": "",
    }


def _failed(reason: str) -> dict:
    """QC record for a channel whose features could not be extracted."""
    return {"hfo_power": None, "hfo_peak_freq": None, "qc_status": "failed",
            "qc_reason": reason}


def load_mat_file(path: str | Path, fs_override: float | None = None) -> tuple[np.ndarray, float]:
    """Load a MATLAB ``.mat`` file into ``(trial, fs)``.

    Supported structures, in priority order:

    1. FieldSync-style struct: ``data['trial']`` (cell or array) with
       ``data['fsample']``.
    2. Flat fields: ``trial`` and ``fsample`` at top level.
    3. A single numeric array named ``data`` (channels x samples) —
       requires ``fs_override``.

    Args:
        path: Path to the ``.mat`` file.
        fs_override: Use this sampling rate instead of the one stored in
            the file (also required for structure 3).

    Returns:
        ``(trial, fs)`` where ``trial`` is a 2-D float array shaped
        ``(n_channels, n_samples)`` and ``fs`` is a float.

    Raises:
        MissingDataError: If the file does not exist.
        DataValidationError: If no supported structure is found or the
            sampling rate is missing where it is required.
    """
    path = Path(path)
    if not path.exists():
        raise MissingDataError(f"ECoG data file not found: {path}")

    mat = loadmat(path)
    mat = {k: v for k, v in mat.items() if not k.startswith("__")}

    trial: np.ndarray | None = None
    fs: float | None = None

    for key in ("data", "trial"):
        if key in mat and isinstance(mat[key], np.ndarray) and mat[key].dtype == object:
            struct = mat[key]
            try:
                inner = struct[0, 0]
            except (IndexError, ValueError, TypeError):
                continue
            names = inner.dtype.names if hasattr(inner, "dtype") else None
            if names and "trial" in names:
                trial = np.asarray(inner["trial"], dtype=float)
                if "fsample" in names and fs_override is None:
                    fs = float(np.asarray(inner["fsample"]).ravel()[0])
                break

    if trial is None:
        for key in ("trial", "data"):
            if key in mat and isinstance(mat[key], np.ndarray) and mat[key].dtype != object:
                trial = np.asarray(mat[key], dtype=float)
                break

    if trial is None:
        raise DataValidationError(
            f"{path.name}: no supported data structure found. Supported: a struct with "
            f"'trial'/'fsample' fields, or a numeric 'trial'/'data' array. Top-level "
            f"fields seen: {sorted(mat)}"
        )

    if trial.ndim == 1:
        trial = trial.reshape(1, -1)
    elif trial.ndim > 2:
        trial = trial.reshape(trial.shape[0], -1)

    if fs_override is not None:
        fs = float(fs_override)
    elif fs is None:
        flat_fs = mat.get("fsample", mat.get("fs"))
        if flat_fs is None:
            raise DataValidationError(
                f"{path.name}: no sampling rate in file and no --fs-override given. "
                "Pass fs_override explicitly."
            )
        fs = float(np.asarray(flat_fs).ravel()[0])

    return trial, fs


def extract_recording_features(
    trial: np.ndarray,
    fs: float,
    band: HFOConfig | None = None,
    channel_labels: list[str] | None = None,
    source_file: str | None = None,
) -> pd.DataFrame:
    """Extract channel-level HFO features for one recording.

    Channels are processed independently; a channel that fails QC keeps
    its row with ``qc_status="failed"`` and missing feature values
    (NaN in the returned frame — never zero). This function never
    raises on per-channel extraction problems — the QC columns are the
    structured record — but an invalid sampling rate still raises
    :class:`SamplingRateError` because it invalidates every channel at
    once.
    """
    band = band or HFOConfig()
    validate_sampling_rate(fs, band)

    n_channels = trial.shape[0]
    labels = list(channel_labels) if channel_labels else [f"ch_{i}" for i in range(n_channels)]
    if len(labels) < n_channels:
        labels += [f"ch_{i}" for i in range(len(labels), n_channels)]

    records = []
    for idx in range(n_channels):
        record = extract_channel_features(trial[idx], fs, band)
        records.append({
            "channel": labels[idx],
            "source_file": source_file,
            **record,
        })
    return pd.DataFrame(records)


def extract_dataset(
    input_dir: str | Path,
    band: HFOConfig | None = None,
    fs_override: float | None = None,
) -> pd.DataFrame:
    """Extract channel-level HFO features for every recording in a dataset.

    Discovers subject subdirectories under ``input_dir`` and processes
    every ``*.mat`` file in each, in lexicographic order. There is no
    per-subject subsampling and no preference heuristic: the selection
    rule is "all files, sorted", and each output row records its source
    file so it can be audited.

    Raises:
        MissingDataError: If ``input_dir`` does not exist or contains no
            subject folders with ``.mat`` files. (Never silently
            substitutes placeholder data.)
    """
    input_dir = Path(input_dir)
    band = band or HFOConfig()
    if not input_dir.is_dir():
        raise MissingDataError(
            f"ECoG input directory not found: {input_dir}. Point --input-dir at the "
            "folder of per-subject .mat files; this package does not distribute data."
        )

    subject_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    mat_files: list[tuple[str, Path]] = []
    for subject_dir in subject_dirs:
        for mat_path in sorted(subject_dir.glob("*.mat")):
            mat_files.append((subject_dir.name, mat_path))

    if not mat_files:
        raise MissingDataError(
            f"No .mat files found under subject folders in {input_dir}. Expected one "
            "subdirectory per subject containing .mat recordings."
        )

    frames = []
    for subject, mat_path in mat_files:
        trial, fs = load_mat_file(mat_path, fs_override=fs_override)
        frame = extract_recording_features(
            trial, fs, band=band, source_file=mat_path.name
        )
        frame.insert(0, "subject", subject)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def aggregate_to_subject(channel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate channel-level features to one row per subject.

    Only channels with ``qc_status == "ok"`` contribute to the means;
    failed channels are counted in ``n_failed_channels`` so that data
    loss remains visible. A subject whose channels all failed gets NaN
    features (never zeros) and ``n_ok_channels == 0``.
    """
    required = {"subject", "hfo_power", "hfo_peak_freq", "qc_status"}
    missing = required - set(channel_df.columns)
    if missing:
        raise DataValidationError(
            f"aggregate_to_subject expects channel-level rows with columns {sorted(required)}; "
            f"missing {sorted(missing)}"
        )

    ok = channel_df[channel_df["qc_status"] == "ok"]
    subjects = channel_df["subject"].unique()
    counts = channel_df.groupby("subject").size().reindex(subjects).rename("n_channels")
    ok_counts = (
        ok.groupby("subject").size().reindex(subjects, fill_value=0).rename("n_ok_channels")
    )
    failed_counts = (
        channel_df[channel_df["qc_status"] != "ok"]
        .groupby("subject")
        .size()
        .reindex(subjects, fill_value=0)
        .rename("n_failed_channels")
    )

    means = ok.groupby("subject")[["hfo_power", "hfo_peak_freq"]].mean().reindex(subjects)
    out = means.join(counts).join(ok_counts).join(failed_counts).reset_index()
    out = out.rename(columns={"hfo_power": "hfo_power_mean", "hfo_peak_freq": "hfo_peak_freq_mean"})
    return out

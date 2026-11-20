"""Deterministic synthetic cohort with planted signals (v2 verification).

Generates recordings with the same structure as the converted contract
(strip channels, glove, fs, segment table, anchor position) plus
OPTIONAL planted ground truth, used by the leakage and recovery tests:

- ``plant="movement"`` — a movement-coupled HFA-band oscillation whose
  envelope follows the movement drive, at ONE localized grid offset
  (spatially localized signal): detection and zero-shot correlation
  must recover it.
- ``plant="innovation"`` — the movement envelope carries AR(1)
  dynamics PLUS a lagged contribution from the planted neural channel's
  own slow drive: the regression track must beat the AR baseline only
  when neural features are available.
- ``plant="autocorr_only"`` — the envelope has AR(1) dynamics but NO
  neural coupling: the regression track must stay null.
- ``plant="identity"`` — a large subject-specific baseline power offset
  with no movement coupling: detection must NOT benefit (subject
  identity is not a shortcut).
- ``plant="missing_anchor"`` — the anchor-offset channel is dead
  (flatline) and the movement signal lives at the adjacent offset:
  missingness must be represented, never median-filled away.

Everything is seeded and byte-deterministic; no real data anywhere.
"""

from __future__ import annotations

import numpy as np

from .cohort import Recording

FS = 1200.0


def _colored_noise(rng: np.random.Generator, n_samples: int,
                   channel_scale: float = 1.0) -> np.ndarray:
    """1/f-ish neural background: cumulative filtered white noise."""
    white = rng.standard_normal(n_samples + 400)
    smoothed = np.convolve(white, np.ones(96) / 96, mode="same")[400:]
    smoothed -= smoothed.mean()
    std = float(np.std(smoothed)) or 1.0
    return channel_scale * smoothed / std


def _slow_drive(rng: np.random.Generator, n_bins: int,
                ar: float = 0.92) -> np.ndarray:
    """Slow AR(1) movement-vigor drive on the update grid, ~N(0, 1)."""
    drive = np.empty(n_bins)
    drive[0] = rng.standard_normal()
    innovations = rng.standard_normal(n_bins) * 0.4
    for t in range(1, n_bins):
        drive[t] = ar * drive[t - 1] + innovations[t]
    drive -= drive.mean()
    std = float(np.std(drive)) or 1.0
    return drive / std


def _blocks(fs: float, rest_s: float, move_s: float, n_pairs: int) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    sample = 0
    for _ in range(n_pairs):
        for kind, seconds in (("rest", rest_s), ("move", move_s)):
            length = int(round(seconds * fs))
            blocks.append((sample, sample + length, kind))
            sample += length
    return blocks


def make_synthetic_recording(
    position: int,
    seed: int,
    plant: str = "movement",
    anchor_position: int = 5,
    n_strips: int = 1,
    fs: float = FS,
    rest_s: float = 12.0,
    move_s: float = 16.0,
    n_block_pairs: int = 2,
    innovation_lag_bins: int = 3,
    innovation_beta: float = 0.6,
    plant_offset_delta: int = 0,
) -> Recording:
    """One deterministic synthetic recording with the requested plant.

    The planted movement channel is the grid offset
    ``contract.ANCHOR_OFFSET + plant_offset_delta`` (default: the anchor
    itself). Dual-strip recordings (``n_strips=2``) add 8 channels that
    carry nothing (the frozen second-strip policy treats them as
    unanchored context).
    """
    rng = np.random.default_rng(seed)
    blocks = _blocks(fs, rest_s, move_s, n_block_pairs)
    n_samples = blocks[-1][1]
    n_channels = 8 * n_strips
    channel_scales = np.linspace(0.6, 1.6, n_channels)

    # Movement drive on the update grid, expanded to samples per block.
    # The innovation plant gives the neural drive SHORTER memory (more
    # novel variation the envelope history cannot predict) and a stronger
    # coupling — otherwise the innovation's novelty drowns in the drive's
    # own AR-predictable structure.
    update = int(round(0.1 * fs))
    move_bins_per_block = int(round(move_s * fs)) // update
    drives: list[np.ndarray] = [
        _slow_drive(rng, move_bins_per_block) for _ in range(n_block_pairs)
    ]
    neural_ar = 0.6 if plant == "innovation" else 0.92
    neural_slow: list[np.ndarray] = [
        _slow_drive(rng, move_bins_per_block, ar=neural_ar)
        for _ in range(n_block_pairs)
    ]
    if plant == "innovation":
        innovation_beta = max(innovation_beta, 3.0)

    # Planted channel index = strip-1 channel of the requested grid
    # offset (offset 0 ↔ channel anchor−1; offset +delta shifts by delta).
    planted_channel = (anchor_position - 1) + plant_offset_delta
    if not (0 <= planted_channel <= 7):
        planted_channel = min(max(planted_channel, 0), 7)

    ecog = np.vstack([
        _colored_noise(rng, n_samples, channel_scales[ch])
        for ch in range(n_channels)
    ])
    glove = np.zeros((5, n_samples))

    # Subject-identity plant: a big baseline offset on every channel.
    identity_offset = 40.0 if plant == "identity" else 0.0
    ecog = ecog + identity_offset

    move_block_index = 0
    for start, end, kind in blocks:
        length = end - start
        if kind == "move":
            drive = drives[move_block_index]
            neural = neural_slow[move_block_index]
            # Expand the drive to samples (piecewise-constant).
            expanded = np.repeat(drive, update)[:length]
            expanded_neural = np.repeat(neural, update)[:length]
            t = np.arange(length) / fs
            # Glove: taps whose instantaneous rate follows the vigor drive
            # (so inter-tap intervals vary — Track D ground truth) and
            # whose amplitude follows the drive.
            rate = 1.4 + 0.25 * expanded                      # Hz
            phase = 2 * np.pi * np.cumsum(rate) / fs
            for sensor in range(5):
                glove[sensor, start:end] = (
                    (2.0 + 0.8 * expanded)
                    * np.sin(phase + 0.7 * sensor)
                    + 0.05 * rng.standard_normal(length)
                )
            # Planted neural channel: HFA oscillation whose amplitude is
            # RECTIFIED-monotone in the movement drive (power ∝ drive, so
            # band power carries the drive's sign information — an
            # amplitude-squared plant would decorrelate from the linear
            # envelope target and is deliberately avoided). The carrier is
            # 132 Hz — deliberately OFF the line-noise harmonic ladder
            # (60/120/180…), which the frozen line-noise rule would notch.
            if plant in ("movement", "innovation", "missing_anchor"):
                combined = 0.5 * expanded + 0.5 * expanded_neural
                amplitude = np.clip(0.9 + 0.9 * combined, 0.05, None)
                carrier = np.sin(2 * np.pi * 132.0 * t)
                ecog[planted_channel, start:end] += amplitude * carrier
            move_block_index += 1
        else:
            for sensor in range(5):
                glove[sensor, start:end] = 0.03 * rng.standard_normal(length)

    if plant == "innovation":
        # Envelope innovation: the glove amplitude (ALL sensors — vigor
        # modulates the whole hand) gets an extra component driven by the
        # LAGGED planted-neural slow drive, on top of the amplitude's own
        # AR dynamics — the neural feature adds beyond-history information
        # by construction.
        for i, (start, end, kind) in enumerate(blocks):
            if kind != "move":
                continue
            neural = neural_slow[_move_block_index(blocks, i)]
            lagged = np.concatenate([
                np.zeros(innovation_lag_bins),
                neural[:-innovation_lag_bins] if innovation_lag_bins else neural,
            ])
            length = end - start
            update = max(int(round(0.1 * fs)), 1)
            bins = min(lagged.size, length // update)
            amp_extra = np.repeat(innovation_beta * lagged[:bins], update)[:length]
            rate = 1.4 + 0.25 * np.repeat(
                drives[_move_block_index(blocks, i)], update)[:length]
            phase = 2 * np.pi * np.cumsum(rate) / fs
            for sensor in range(5):
                glove[sensor, start:end] += amp_extra * np.sin(
                    phase + 0.7 * sensor)

    if plant == "missing_anchor":
        # Kill the anchor-offset channel itself (grid offset 0 → channel
        # anchor−1, flatline); the planted signal lives one grid offset
        # away (channel ``anchor_position``, i.e. offset +1). Missingness
        # must be represented, never median-filled away.
        ecog[anchor_position - 1, :] = 0.0

    return Recording(
        position=position,
        ecog=ecog,
        glove=glove,
        fs=fs,
        blocks=blocks,
        anchor_position=anchor_position,
        covariates={"age": 60.0 + position, "dbs_volt": 1.0 + 0.1 * position,
                    "dbs_freq": 130.0},
        signal_quality_flag=1.0,
    )


def _move_block_index(blocks: list[tuple[int, int, str]], block_position: int) -> int:
    """Move-block ordinal of the block at ``block_position``."""
    index = 0
    for i, (_s, _e, kind) in enumerate(blocks):
        if i == block_position:
            return index
        if kind == "move":
            index += 1
    return index


def make_synthetic_cohort(
    n: int = 6, seed0: int = 4200, plant: str = "movement", **kwargs
) -> list[Recording]:
    """A deterministic cohort of ``n`` synthetic recordings."""
    anchors = [5, 4, 6, 5, 3, 7, 5, 4]
    plants = {
        "movement": ["movement"] * n,
        "innovation": ["innovation"] * n,
        "autocorr_only": ["autocorr_only"] * n,
        "identity": ["identity"] * n,
        "missing_anchor": ["missing_anchor"] * n,
        "mixed": (["movement", "autocorr_only"] * ((n + 1) // 2))[:n],
    }
    per_recording_plant = plants.get(plant, [plant] * n)
    recordings = []
    for i in range(n):
        kwargs_per = dict(kwargs)
        if plant == "missing_anchor":
            kwargs_per.setdefault("plant_offset_delta", 1)
        recordings.append(make_synthetic_recording(
            position=i + 1, seed=seed0 + i,
            plant=per_recording_plant[i],
            anchor_position=anchors[i % len(anchors)],
            **kwargs_per,
        ))
    return recordings


def make_fast_cohort(
    n: int = 6, seed0: int = 4200, plant: str = "movement", n_strips: int = 1,
) -> list[Recording]:
    """A cohort that satisfies the FROZEN cohort gates (usable MOVE
    ≥ 30 s, untouched 30%-split test ≥ 55 s, calibration ≥ 8 s) at a
    reduced fs for test speed. Blocks: 3 × (12 s REST + 28 s MOVE);
    30 % of 84 s MOVE = 25.2 s calibration, 58.8 s test."""
    return make_synthetic_cohort(
        n=n, seed0=seed0, plant=plant, n_strips=n_strips, fs=600.0,
        rest_s=12.0, move_s=28.0, n_block_pairs=3,
    )

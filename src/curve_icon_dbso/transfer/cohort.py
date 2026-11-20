"""Cohort flow and eligibility (v2 contract §2).

A recording is a plain dict with the loaded npz arrays plus its
workbook anchor position. Eligibility gates are evaluated per track;
exclusions carry machine-readable reasons and are counted, never
silent. Aggregate outputs only — subject codes never enter committed
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import DataValidationError
from ..glove_phenotyping import channel_block_kinematics
from ..kinematics import parse_restmove_blocks
from . import contract
from .spatial import build_grid_map, ok_channel_mask
from .splits import build_calibration_plan, shared_test_start
from .targets import move_envelope_stream


@dataclass
class Recording:
    """One loaded recording (local-memory only; never serialized with
    identifiers)."""

    position: int                      # 1-based ordinal, stable across runs
    ecog: np.ndarray                   # strip channels × samples (P-label order)
    glove: np.ndarray                  # 5 sensors × samples
    fs: float
    blocks: list[tuple[int, int, str]]
    anchor_position: int
    covariates: dict = field(default_factory=dict)
    signal_quality_flag: float | None = None

    @property
    def n_samples(self) -> int:
        return int(self.ecog.shape[1])


def load_recording_npz(
    path: Path, position: int, anchor_position: int,
    signal_quality_flag: float | None = None,
) -> Recording:
    npz = np.load(path, allow_pickle=False)
    required = {"ecog", "glove", "fs_ecog", "restmove_segs"}
    missing = required - set(npz.files)
    if missing:
        raise DataValidationError(f"{path.name}: missing arrays {sorted(missing)}")
    ecog = np.asarray(npz["ecog"], dtype=float)
    glove = np.asarray(npz["glove"], dtype=float)
    fs = float(np.asarray(npz["fs_ecog"]).squeeze())
    fs_glove = float(np.asarray(npz["fs_glove"]).squeeze())
    if fs != fs_glove:
        raise DataValidationError(
            f"recording #{position:03d}: ECoG fs {fs} != glove fs {fs_glove}"
        )
    if ecog.shape[1] != glove.shape[1]:
        raise DataValidationError(
            f"recording #{position:03d}: ECoG and glove sample grids differ"
        )
    blocks = parse_restmove_blocks(np.asarray(npz["restmove_segs"], dtype=float))
    return Recording(
        position=position,
        ecog=ecog,
        glove=glove,
        fs=fs,
        blocks=blocks,
        anchor_position=int(anchor_position),
        covariates={
            key: float(npz[key]) for key in ("age", "dbs_volt", "dbs_freq")
            if key in npz.files
        },
        signal_quality_flag=signal_quality_flag,
    )


def usable_seconds(ecog: np.ndarray, fs: float,
                   blocks: list[tuple[int, int, str]]) -> dict[str, float]:
    """Usable REST/MOVE seconds after whole-recording channel QC
    (failed channels removed, never imputed)."""
    ok = ok_channel_mask(ecog)
    per_kind = {"rest": 0.0, "move": 0.0}
    for start, end, kind in blocks:
        segment = ecog[ok, start:min(end, ecog.shape[1])]
        if segment.size == 0:
            continue
        # A second counts when at least half the OK channels are finite.
        finite_rows = np.isfinite(segment).all(axis=0)
        per_kind[kind] += float(finite_rows.sum()) / fs
    return per_kind


def event_eligibility(glove: np.ndarray, fs: float,
                      blocks: list[tuple[int, int, str]]) -> dict:
    """Track D gate: ≥ 2 contributing channel-blocks under the frozen
    tap floors (reuses the frozen prominence + cycle rules)."""
    contributing = 0
    cycles = 0
    for start, end, kind in blocks:
        if kind != "move":
            continue
        end = min(end, glove.shape[1])
        for channel in range(glove.shape[0]):
            record = channel_block_kinematics(glove[channel, start:end], fs)
            if record["contributing"] and record["n_cycles"] and \
                    record["n_cycles"] >= 5:
                contributing += 1
                cycles += record["n_cycles"]
    return {"contributing_channel_blocks": contributing,
            "n_cycles": cycles,
            "passed": contributing >= 2}


@dataclass
class CohortFlow:
    """Aggregate cohort-flow record (commit-safe: no identifiers)."""

    n_converted: int
    eligible_positions: list[int] = field(default_factory=list)
    exclusion_reasons: dict[str, list[int]] = field(default_factory=dict)
    signal_quality_flagged_positions: list[int] = field(default_factory=list)
    track_d_ineligible_positions: list[int] = field(default_factory=list)
    calibration_gate_positions: dict[str, list[int]] = field(default_factory=dict)

    def summary(self) -> dict:
        """PUBLIC aggregate cohort flow (commit-safe): COUNTS only —
        recording ordinal-position lists never enter committable output
        (v2.1.0 privacy rule); see ``local_summary`` for the local-only
        artifact."""
        return {
            "n_recordings_loaded": self.n_converted,
            "n_eligible": len(self.eligible_positions),
            "n_excluded_by_reason": {
                reason: len(positions)
                for reason, positions in self.exclusion_reasons.items()
            },
            "n_signal_quality_flagged_retained": len(
                self.signal_quality_flagged_positions),
            "n_track_d_ineligible": len(self.track_d_ineligible_positions),
            "n_calibration_gate_failures_by_fraction": {
                fraction: len(positions)
                for fraction, positions in self.calibration_gate_positions.items()
            },
            "eligibility_rule": (
                "gates: usable MOVE ≥ 30 s, usable REST ≥ 5 s, anchor present "
                "[1,8], 30%-split test ≥ 55 s and calibration ≥ 8 s; "
                "signal-quality flag retained + reported (no defined rule "
                "failed); Track D additionally needs ≥ 2 contributing "
                "channel-blocks; ordinal-position lists are local-only "
                "(counts only in committable output)"
            ),
        }

    def local_summary(self) -> dict:
        """LOCAL-ONLY cohort flow (gitignored run directory): carries
        the recording ordinal-position lists the public summary counts."""
        return {
            **self.summary(),
            "eligible_ordinal_positions": sorted(self.eligible_positions),
            "exclusions_by_reason": {
                reason: sorted(positions)
                for reason, positions in self.exclusion_reasons.items()
            },
            "signal_quality_flagged_ordinal_positions": sorted(
                self.signal_quality_flagged_positions),
            "track_d_ineligible_ordinal_positions": sorted(
                self.track_d_ineligible_positions),
            "calibration_gate_failures_by_fraction": {
                fraction: sorted(positions)
                for fraction, positions in self.calibration_gate_positions.items()
            },
        }


def evaluate_cohort(recordings: list[Recording]) -> CohortFlow:
    """Evaluate the frozen gates for every recording."""
    flow = CohortFlow(n_converted=len(recordings))
    for recording in recordings:
        reasons: list[str] = []
        try:
            build_grid_map(recording.ecog.shape[0], recording.anchor_position)
        except DataValidationError as exc:
            reasons.append(f"anchor_gate: {exc}")
        usable = usable_seconds(recording.ecog, recording.fs, recording.blocks)
        if usable["move"] < contract.MIN_MOVE_S:
            reasons.append(
                f"min_move_s: {usable['move']:.1f} < {contract.MIN_MOVE_S}")
        if usable["rest"] < contract.MIN_REST_S:
            reasons.append(
                f"min_rest_s: {usable['rest']:.1f} < {contract.MIN_REST_S}")
        envelope = move_envelope_stream(recording.glove, recording.fs,
                                        recording.blocks)
        if envelope["stream"] is None:
            reasons.append(f"envelope: {envelope['qc_reason']}")
        # Calibration/test gates at every frozen fraction (identical test
        # interval from the primary fraction).
        for fraction in contract.CALIBRATION_FRACTIONS:
            plan = build_calibration_plan(recording.blocks, recording.n_samples,
                                          recording.fs, fraction)
            for gate, record in plan.gates.items():
                if not record["passed"]:
                    flow.calibration_gate_positions.setdefault(
                        f"{fraction}:{gate}", []).append(recording.position)
        test_start = shared_test_start(recording.blocks, recording.n_samples,
                                       recording.fs)
        test_move_s = sum(
            (min(e, recording.n_samples) - max(s, test_start)) / recording.fs
            for s, e, kind in recording.blocks
            if kind == "move" and e > test_start
        )
        if test_move_s < contract.MIN_TEST_S:
            reasons.append(
                f"min_test_s: {test_move_s:.1f} < {contract.MIN_TEST_S}")
        if reasons:
            for reason in reasons:
                flow.exclusion_reasons.setdefault(reason, []).append(
                    recording.position)
            continue
        flow.eligible_positions.append(recording.position)
        if recording.signal_quality_flag is not None and \
                recording.signal_quality_flag == 0:
            flow.signal_quality_flagged_positions.append(recording.position)
        events = event_eligibility(recording.glove, recording.fs,
                                   recording.blocks)
        if not events["passed"]:
            flow.track_d_ineligible_positions.append(recording.position)
    return flow

"""The frozen, versioned analysis contract for the follow-on ECoG/kinematic
study (three arms: REST, MOVE, DELTA).

This module IS the machine-readable protocol (analogous to
:mod:`curve_icon_dbso.schema` for the serving contract):

* every prespecified quantity — targets, bands, QC floors, windows/lags,
  model grids, fold rules, metrics, baselines, promotion/transfer
  criteria, uncertainty and null procedures, ablations, and the rules
  separating confirmatory / exploratory / post-hoc work — is defined
  here exactly once;
* :func:`protocol_document` renders the full contract as a
  JSON-serializable dict; its canonical encoding is fingerprinted by
  :func:`protocol_hash`;
* :func:`validate_protocol` enforces structural invariants at import by
  consumers (band ordering, grid bounds, floor levels, semver shape).

Freeze discipline: the contract was frozen **before any follow-on
real-data result was inspected** (owner instruction,
``docs/neural_decoding_implementation_prompts.md`` Prompt 0,
2026-11-16). Any change to a frozen quantity requires a new
``PROTOCOL_VERSION`` and a changelog entry in
``docs/followon_protocol.md``; runs record ``protocol_version`` and
``protocol_hash``. A pinned-hash test
(``tests/test_protocol_contract.py``) fails on any silent edit, so the
version bump can never happen unnoticed.

Relationship to the frozen first study: the recorded first authorized
real-data result (the prespecified five-feature kinematic analysis) is
immutable history; this contract governs **follow-on analyses only**
and never re-fits, re-scores, or overwrites that result.

Dependency discipline: this module is deliberately **standard-library
only** (no numpy/scipy/sklearn imports) so the contract can be hashed,
validated, and recorded on any machine. Where a frozen constant mirrors
an implemented constant elsewhere in the package (tap-rate QC floors,
permutation/bootstrap floors, the fold-count rule), the equivalence is
enforced by tests against those modules, not by importing them here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Final

#: Version of THIS contract document. Semver. Bump (with a changelog
#: entry in docs/followon_protocol.md) on any change to a frozen quantity.
PROTOCOL_VERSION: Final[str] = "1.0.1"

#: UTC date the contract was frozen, before any follow-on real-data
#: result was inspected.
FROZEN_DATE: Final[str] = "2026-11-16"

#: Changelog of deliberate post-freeze revisions (mirrored in
#: docs/followon_protocol.md §14). v1.0.1 is a CORRECTIVE revision: it
#: completes implementation coverage of already-frozen criteria after the
#: v1.0.0 integration run left parts of the frozen matrix unexecuted —
#: no target, band, model, grid, floor, or fold rule changed.
REVISIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "version": "1.0.1",
        "date": "2026-11-17",
        "kind": "corrective",
        "change": (
            "v1.0.0's integration run did not fully cover the frozen "
            "criteria: baseline MAEs were never emitted for the MOVE "
            "transfer criterion; the shift null sampled nominal shifts "
            "whose effective circular displacement could be a single "
            "update bin; the DELTA-combined block was never built; the "
            "Holm family was run with four of its five prespecified "
            "models. v1.0.1 completes the frozen matrix: adds the "
            "delta_combined ablation block (§11 'DELTA analogously'), "
            "pins the shift null to effective displacement min(shift, "
            "span − shift) ≥ 2 s, and requires all ten Holm family "
            "members (optional-model members absent from an environment "
            "are entered conservatively at p = 1 and marked "
            "not_executed). No frozen quantity changed."
        ),
    },
)

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

#: The three prespecified arms. REST and DELTA are subject-level; MOVE is
#: time-resolved (window-level) with subject-held-out evaluation.
ARMS: Final[tuple[str, ...]] = ("REST", "MOVE", "DELTA")

ARM_DEFINITIONS: Final[dict[str, str]] = {
    "REST": (
        "Resting-state neural features (REST blocks) predicting "
        "subject-level tapping phenotypes"
    ),
    "MOVE": (
        "Time-resolved movement ECoG (MOVE blocks) predicting concurrent "
        "glove movement (window-level decoding)"
    ),
    "DELTA": (
        "Movement-minus-rest neural modulation (MOVE − REST) predicting "
        "subject-level tapping phenotypes"
    ),
}

# ---------------------------------------------------------------------------
# Targets (frozen)
# ---------------------------------------------------------------------------

#: Frozen primary subject-level target. The implementation is the already
#: frozen ``curve_icon_dbso.kinematics`` tap-rate extraction and is NOT
#: changed by this contract; the floors below mirror it and are pinned by
#: tests.
PRIMARY_SUBJECT_TARGET: Final[str] = "tap_rate_mean"

#: Secondary kinematic targets (frozen definitions; implemented by the
#: glove-phenotyping phase exactly as written here).
SECONDARY_SUBJECT_TARGETS: Final[tuple[str, ...]] = (
    "tap_interval_cv",
    "tap_vigor_index",
    "tap_sequence_slope",
)

#: Amplitude-like quantity recorded for QC only — glove sensors are
#: uncalibrated across subjects, so it is WITHIN-SUBJECT-ONLY and excluded
#: from all cross-subject modeling.
WITHIN_SUBJECT_ONLY_TARGETS: Final[tuple[str, ...]] = ("tap_amplitude",)

#: MOVE-arm continuous target: the per-recording standardized movement
#: vigor envelope (frozen definition under ``targets.move_continuous``).
MOVE_CONTINUOUS_TARGET: Final[str] = "glove_vigor_envelope_z"

# Frozen tap-rate floors — MUST equal curve_icon_dbso.kinematics constants
# (enforced by tests/test_protocol_contract.py, not by importing scipy here).
TAP_RATE_LOW_HZ: Final[float] = 0.5
TAP_RATE_HIGH_HZ: Final[float] = 4.0
TAP_PROMINENCE_FLOOR: Final[float] = 0.20
TAP_MIN_CHANNEL_BLOCKS: Final[int] = 2
TAP_WELCH_SEGMENT_SECONDS: Final[float] = 1.0

# Frozen floors for the secondary cycle-based targets.
TAP_MIN_CYCLES_PER_BLOCK: Final[int] = 5
TAP_DETECTION_BAND_HZ: Final[tuple[float, float]] = (0.5, 4.0)
TAP_MIN_PEAK_SEPARATION_S: Final[float] = 0.15
TAP_DETECTION_FILTER_ORDER: Final[int] = 4

# ---------------------------------------------------------------------------
# Neural feature families and exact frequency boundaries
# ---------------------------------------------------------------------------

#: Standard analysis bands. Convention: every band is the half-open
#: interval [low, high) over Welch PSD bin centers — adjacent bands share
#: an edge without double counting. The topmost ACTIVE edge (see
#: :func:`resolve_spectral_bands`) is treated as inclusive.
STANDARD_BANDS_HZ: Final[dict[str, tuple[float, float]]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 35.0),
    "gamma_low": (35.0, 60.0),
}

#: High gamma: the standing choice is 60–100 (available for any fs with
#: 10 Hz Nyquist margin). The 90–200 alternative from the literature is
#: recorded but is NEVER auto-selected: adopting it requires a new
#: contract version (changelog) plus cohort fs ≥ 420 Hz.
HIGH_GAMMA_DEFAULT: Final[tuple[float, float]] = (60.0, 100.0)
HIGH_GAMMA_ALTERNATIVE: Final[tuple[float, float]] = (90.0, 200.0)
HIGH_GAMMA_ALTERNATIVE_MIN_FS_HZ: Final[float] = 420.0

#: HFO block (legacy 80–500 definition, separate feature block — it is
#: never summed with band powers even though it overlaps high gamma).
#: Cohort-uniform rule: active only when EVERY recording's sampling rate
#: clears the upper edge with the frozen Nyquist margin.
HFO_BAND_HZ: Final[tuple[float, float]] = (80.0, 500.0)

#: Nyquist safety margin required between any active top edge and fs/2.
NYQUIST_MARGIN_HZ: Final[float] = 10.0


def resolve_spectral_bands(min_fs_hz: float) -> dict:
    """Deterministic, acquisition-driven band availability (pure function).

    A band is active when its top edge clears ``fs/2 - NYQUIST_MARGIN_HZ``.
    The HFO block additionally requires the artifact QC gate at extraction
    time. Cohort uniformity: callers pass the MINIMUM sampling rate over
    all recordings, so feature space is identical for every subject; a
    recording below ``min_fs_hz`` is a QC exclusion, never a partial
    feature vector.

    Raises:
        ValueError: On a non-positive or non-finite ``min_fs_hz``.
    """
    if not math.isfinite(min_fs_hz) or min_fs_hz <= 0:
        raise ValueError(f"min_fs_hz must be a positive finite number, got {min_fs_hz!r}")
    ceiling = min_fs_hz / 2.0 - NYQUIST_MARGIN_HZ
    bands = {
        name: list(limits)
        for name, limits in STANDARD_BANDS_HZ.items()
        if limits[1] <= ceiling
    }
    gamma_high = list(HIGH_GAMMA_DEFAULT) if HIGH_GAMMA_DEFAULT[1] <= ceiling else None
    return {
        "min_fs_hz": float(min_fs_hz),
        "nyquist_margin_hz": NYQUIST_MARGIN_HZ,
        "bands_active": bands,
        "gamma_high": gamma_high,
        "hfo_block_feasible": bool(HFO_BAND_HZ[1] <= ceiling),
        "hfo_block_active_note": (
            "feasibility is necessary but not sufficient: the HFO block is "
            "active only if the frozen artifact QC gate also passes for the "
            "cohort"
        ),
    }


# Frozen feature transforms (implemented by the signal-processing phase).
FEATURE_TRANSFORMS: Final[dict[str, str]] = {
    "band_power": "Welch PSD integral over the band (units: signal^2)",
    "relative_band_power": "band power / total PSD power over the active band range",
    "log_relative_band_power": "log10(relative band power) — the primary spectral feature",
    "aperiodic_exponent": (
        "slope of the OLS fit of log10(PSD) vs log10(f) over the active band "
        "range bins (aperiodic 1/f estimate, no external dependency)"
    ),
    "aperiodic_offset": "intercept of the same fit",
    "aperiodic_adjusted_band_power": (
        "mean in-band PSD divided by the aperiodic fit's PSD predicted at the "
        "band center (dimensionless periodic surplus)"
    ),
}

FEATURE_DYNAMICS: Final[dict[str, str]] = {
    "hjorth_mobility": "standard Hjorth mobility on the QC-OK block signal",
    "hjorth_complexity": "standard Hjorth complexity on the QC-OK block signal",
    "line_length_per_s": "sum |diff(x)| divided by block duration (per second)",
    "spectral_entropy": (
        "normalized Shannon entropy of the PSD over the active band range"
    ),
    "beta_burst_occupancy": (
        "fraction of REST-block samples whose beta_high envelope exceeds the "
        "recording's own frozen threshold"
    ),
    "beta_burst_mean_duration_ms": (
        "mean contiguous above-threshold duration; bursts shorter than "
        "beta_burst_min_duration_ms are not bursts"
    ),
    "beta_burst_median_amplitude_ratio": (
        "median envelope/threshold ratio during bursts (dimensionless)"
    ),
}

#: Frozen beta-burst thresholds — computed from the recording's own REST
#: blocks (reference data only): no target information, no fold membership,
#: no held-out subject influence.
BURST_THRESHOLD_PERCENTILE: Final[float] = 75.0
BURST_MIN_DURATION_MS: Final[float] = 50.0

#: DELTA definition per feature family: log-ratio for power-type features,
#: signed native-unit difference for dynamics features.
DELTA_RULES: Final[dict[str, str]] = {
    "power_features": "log10(MOVE power) − log10(REST power) (dimensionless log ratio)",
    "dynamics_features": "MOVE value − REST value (signed difference, native units)",
    "same_block_pairing": (
        "MOVE features are averaged over MOVE blocks, REST features over "
        "REST blocks of the same recording, before the difference"
    ),
}

# ---------------------------------------------------------------------------
# Artifact / QC rules (frozen)
# ---------------------------------------------------------------------------

QC_RULES: Final[dict[str, str]] = {
    "non_finite": "any non-finite sample fails the channel-block/window immediately",
    "flatline": (
        "a window whose max − min ≤ 1e-12 fails (dead channel / clamped ADC)"
    ),
    "saturation": (
        "a channel-block/window with > 5% of samples at the recording's max "
        "or min sample value fails (rail detection)"
    ),
    "rapid_transients": (
        "a window whose fraction of samples with |diff| > 20 × MAD(diff over "
        "the recording) exceeds 0.005 fails"
    ),
    "implausible_band_power": (
        "a window whose band power z-score against the recording's own "
        "REST-window distribution exceeds 10 fails per band"
    ),
    "line_noise": (
        "notch at the detected line frequency ± 2 Hz plus harmonics up to "
        "Nyquist; the line frequency is the larger of the PSD maxima within "
        "± 2 Hz of 50 Hz and of 60 Hz; if the two are within 3 dB the run "
        "records the QC note 'line_freq_ambiguous' and uses the larger"
    ),
    "no_silent_repair": (
        "failed channels/windows are counted with machine-readable reasons "
        "and never imputed, zero-filled, or silently dropped"
    ),
    "target_qc": (
        "tap floors are frozen: >= 0.20 in-band prominence share per "
        "contributing channel-block and >= 2 contributing channel-blocks per "
        "recording for tap_rate_mean; >= 5 detected cycles per contributing "
        "channel-block for the cycle-based secondary targets"
    ),
}

#: Minimum usable data for a recording to enter each arm (after QC).
MIN_REST_DATA_S: Final[float] = 5.0
MIN_MOVE_DATA_S: Final[float] = 30.0

# ---------------------------------------------------------------------------
# Channel / anatomical aggregation (frozen)
# ---------------------------------------------------------------------------

AGGREGATION: Final[dict[str, str]] = {
    "rereferencing": (
        "bipolar rereferencing ONLY when an electrode-adjacency mapping is "
        "provided for the cohort; the documented fallback otherwise is a "
        "common median reference across QC-OK channels, computed per "
        "recording (target-independent)"
    ),
    "region_aggregation": (
        "when a channel→anatomical-region mapping exists, features are "
        "summarized per region (PM, M1, S1, other/unknown) by the MEDIAN "
        "over that region's QC-OK channels"
    ),
    "fallback_aggregation": (
        "without a region mapping: MEDIAN over all QC-OK channels, plus the "
        "QC-OK channel count carried as an explicit covariate column"
    ),
    "uniformity": (
        "feature blocks are cohort-uniform: a block is active for the whole "
        "cohort or excluded entirely (reported), never partially computed"
    ),
    "localization_status": (
        "channel→region mapping availability is UNKNOWN until the "
        "localization metadata is inspected; the fallback is the frozen "
        "default until then (no guessing)"
    ),
}

REGION_LABELS: Final[tuple[str, ...]] = ("PM", "M1", "S1", "other_unknown")

# ---------------------------------------------------------------------------
# Windowing: MOVE arm (frozen)
# ---------------------------------------------------------------------------

MOVE_WINDOW_S: Final[float] = 0.5
MOVE_UPDATE_S: Final[float] = 0.1
#: Preceding neural context: stacked lagged windows, expressed as offsets
#: BACK from the target sample (0.0 = the window ending at the target;
#: 0.5 = the window ending 500 ms earlier). All context is in the neural
#: past — never the future.
MOVE_LAGS_S: Final[tuple[float, ...]] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
MOVE_LAGS_SEMANTICS: Final[str] = (
    "each value is an offset BACK from the target sample; the decoder sees "
    "only neural windows strictly ending at or before the target time — "
    "never the future"
)
MOVE_WINDOW_RULE: Final[str] = (
    "windows are [t − 0.5, t) s ending at the target sample (causal); "
    "windows never cross REST/MOVE block boundaries or QC-invalid intervals"
)
MOVE_ENVELOPE_RULE: Final[str] = (
    "per QC-OK glove channel: 0.5–4 Hz Butterworth (order 4, zero-phase), "
    "Hilbert |envelope|, average across OK channels, 100 ms moving-mean "
    "smooth, mean within each 100 ms update bin, then per-recording "
    "z-score across MOVE samples (the standardization is part of the "
    "frozen target definition and uses only that recording's glove data)"
)

# ---------------------------------------------------------------------------
# Validation design (frozen) — the outer evaluation unit is ALWAYS subject
# ---------------------------------------------------------------------------

#: Subjects at or below this count use LOSO; above it, GroupKFold(5).
#: MUST equal ``curve_icon_dbso.evaluation.LOSO_MAX_GROUPS`` (pinned by test).
LOSO_MAX_SUBJECTS: Final[int] = 12
OUTER_GROUPKFOLD_SPLITS: Final[int] = 5
INNER_SPLITS: Final[int] = 3
#: Outer-training cohort below this size → inner selection is LOSO within
#: the outer-training subjects instead of GroupKFold(3).
INNER_LOSO_MIN_OUTER_TRAIN_SUBJECTS: Final[int] = 6
INNER_SELECTION_CRITERION: Final[str] = "inner_cv_mean_MAE (lower is better)"
INNER_TIE_RULE: Final[str] = (
    "ties resolve to the smaller grid index (frozen declaration order)"
)

# ---------------------------------------------------------------------------
# Models and bounded hyperparameter grids (frozen)
# ---------------------------------------------------------------------------

#: Hard ceiling on prespecified grid combinations per tunable model.
MAX_GRID_COMBOS: Final[int] = 7

MODELS_SUBJECT_LEVEL: Final[dict[str, dict]] = {
    "dummy_mean": {"kind": "baseline", "grid": {}, "spec": "DummyRegressor(strategy='mean')"},
    "dummy_median": {"kind": "baseline", "grid": {}, "spec": "DummyRegressor(strategy='median')"},
    "ridge_primary": {
        "kind": "primary",
        "spec": "sklearn Ridge(fit_intercept=True)",
        "grid": {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]},
    },
    "svr_linear": {
        "kind": "comparator",
        "spec": "sklearn LinearSVR-equivalent: SVR(kernel='linear')",
        "grid": {
            "C": [0.1, 1.0, 10.0],
            "epsilon_factor_of_train_MAD": [0.1, 0.3],
        },
    },
    "bayesian_ridge": {
        "kind": "comparator",
        "spec": "sklearn BayesianRidge() at library defaults",
        "grid": {},
    },
    "gp_matern32": {
        "kind": "comparator",
        "spec": (
            "GaussianProcessRegressor(kernel=C(1.0) * Matern(nu=1.5, "
            "length_scale=1.0, bounds='fixed') + WhiteKernel(noise_level=1.0, "
            "bounds='fixed'), normalize_y=True, n_restarts_optimizer=0, "
            "alpha=1e-6) — bounds fixed so nothing is tuned"
        ),
        "grid": {},
    },
    "xgboost_shallow": {
        "kind": "comparator_optional",
        "spec": (
            "XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, "
            "n_jobs=1, random_state=42) — the fixed DD-003 parity "
            "configuration; optional dependency, skipped cleanly if absent, "
            "never required for the core analysis"
        ),
        "grid": {},
    },
}

MODELS_MOVE: Final[dict[str, dict]] = {
    "lagged_ridge_primary": {
        "kind": "primary",
        "spec": "lagged Ridge / Wiener cascade over the frozen lag stack",
        "grid": {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]},
    },
    "svr_linear_move": {
        "kind": "comparator",
        "spec": "SVR(kernel='linear') on the same lagged features",
        "grid": {"C": [0.1, 1.0, 10.0]},
    },
    "xgboost_shallow_move": {
        "kind": "comparator_optional",
        "spec": "same fixed shallow XGBoost configuration as the subject-level arms",
        "grid": {},
    },
}


def grid_size(model_spec: dict) -> int:
    """Number of hyperparameter combinations in a frozen model spec."""
    total = 1
    for values in model_spec.get("grid", {}).values():
        total *= len(values)
    return total


# ---------------------------------------------------------------------------
# Metrics, baselines, promotion (frozen)
# ---------------------------------------------------------------------------

METRICS_SUBJECT_LEVEL: Final[tuple[str, ...]] = (
    "MAE", "RMSE", "R2", "Pearson_r", "Spearman_rho",
)
#: MOVE arm: per-held-out-subject window-level metrics FIRST, then median
#: and IQR across subjects. Pooled-window values are secondary only.
METRICS_MOVE: Final[dict[str, str]] = {
    "per_subject_primary": "within-subject Pearson r on window-level (target, prediction) pairs",
    "per_subject_secondary": "within-subject MAE on the standardized target",
    "summary": "median and IQR across held-out subjects — never pooled as independent windows",
    "pooled_reporting": (
        "pooled-window values may appear as explicitly-labeled secondary "
        "descriptives with the non-independence caveat"
    ),
}

BASELINES_SUBJECT_LEVEL: Final[tuple[str, ...]] = ("dummy_mean", "dummy_median")
#: Historical comparator (never re-fit, never overwritten): the frozen
#: five-feature kinematic result on the identical outer folds.
FROZEN_FIVE_FEATURE_REFERENCE: Final[str] = "frozen_first_run_five_feature_result"

BASELINES_MOVE: Final[dict[str, str]] = {
    "fold_global_mean": "training subjects' mean target (fold-local)",
    "block_mean_naive": (
        "held-out recording's own per-block target mean — a naive-target "
        "baseline describing temporal structure, never a model claim"
    ),
    "persistence": "one-update lag: prediction at t is the target at t − 0.1 s",
}

#: Promotion rule (frozen, DD-006-compatible): BOTH error metrics must
#: improve — never a favorable correlation alone.
PROMOTION_RULE: Final[dict[str, str]] = {
    "subject_level": (
        "a model promotes over the field only if, on the identical outer "
        "folds, it improves BOTH MAE and RMSE versus BOTH dummy baselines "
        "AND versus the frozen five-feature reference; improvement is "
        "reported with subject-cluster bootstrap CIs and the corrected "
        "permutation null; Holm-adjusted p-values across the prespecified "
        "family accompany the raw ones"
    ),
    "move": (
        "a MOVE decoder is discussed as superior only if it improves median "
        "across-subject MAE versus the persistence AND block-mean baselines "
        "on the identical outer folds"
    ),
    "no_outer_selection": (
        "no feature block, model, kernel, channel, lag, or target may be "
        "selected using outer-test performance"
    ),
}

#: Prespecified Holm family: primary target × prespecified models ×
#: {MAE, RMSE} within the arm being reported (≤ 20 tests per arm).
#: v1.0.1 clarification: the family is TEN tests. The optional XGBoost
#: members are executed when the dependency is present; in an environment
#: without it they are entered conservatively at p = 1 and recorded as
#: not_executed — the family size never shrinks after the fact.
HOLM_FAMILY_RULE: Final[str] = (
    "primary subject target × {ridge_primary, svr_linear, bayesian_ridge, "
    "gp_matern32, xgboost_shallow} × {MAE, RMSE} within one arm — ten "
    "tests; an optional-model member absent from the environment is "
    "entered conservatively at p = 1 and recorded as not_executed (the "
    "family size never shrinks after the fact)"
)

# ---------------------------------------------------------------------------
# Uncertainty and null procedures (frozen)
# ---------------------------------------------------------------------------

BOOTSTRAP_N_RESAMPLES: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 42
CI_LEVEL: Final[float] = 0.95
#: Tail probabilities require ≥ 1,000 permutations (SCI-009 floor 200).
#: MUST equal ``curve_icon_dbso.bootstrap.PERMUTATION_N_PERMUTATIONS``
#: (pinned by test).
PERMUTATION_MIN_FOR_REPORTING: Final[int] = 1000
PERMUTATION_FLOOR: Final[int] = 200
PERMUTATION_SEED: Final[int] = 42
MOVE_NULL_SHIFTS: Final[int] = 1000
#: Minimum circular-shift separation that breaks local neural–behavior
#: alignment while preserving the autocorrelation/spectrum.
MOVE_CIRCULAR_SHIFT_MIN_S: Final[float] = 2.0

NULL_PROCEDURES: Final[dict[str, str]] = {
    "subject_level": (
        "corrected subject-level label-permutation null through the COMPLETE "
        "nested grouped-CV workflow; each permutation's null statistic is "
        "scored against that permutation's own permuted targets (held-out "
        "rows included), never the original targets; conservative "
        "p = (1 + #at-least-as-favorable) / (1 + n_permutations); permutation "
        "stratified by recording count; subjects, never rows"
    ),
    "move_autocorrelation_preserving": (
        "circular shifts of the glove target WITHIN each recording (neural "
        "windows unchanged), each shift's EFFECTIVE circular displacement "
        "min(shift, span − shift) ≥ 2.0 s (v1.0.1 clarification — a "
        "nominal shift near span is circularly a tiny shift in the "
        "opposite direction), blocks shorter than twice the minimum "
        "displacement excluded and counted, ≥ 1,000 shifts; blockwise "
        "shifts (segments ≥ 2 s, same effective-displacement guarantee) "
        "as the secondary null; independent row "
        "shuffling is NEVER used"
    ),
    "descriptive_status": (
        "all p-values on this cohort are descriptive/hypothesis-generating; "
        "reported with the permutation_caveat for the actual subject count"
    ),
    "influence": (
        "leave-one-subject influence analysis (refit deltas) for the primary "
        "model, descriptive only"
    ),
}

# ---------------------------------------------------------------------------
# Feature-block ablations (frozen, subject-level arms)
# ---------------------------------------------------------------------------

ABLATION_BLOCKS: Final[dict[str, dict]] = {
    "clinical": {
        "features": ["age", "dbs_volt", "dbs_freq"],
        "description": "clinical/stimulation-only block",
    },
    "rest_spectral": {
        "features": [
            "log_relative_band_power[band]",
            "aperiodic_exponent",
            "aperiodic_offset",
            "aperiodic_adjusted_band_power[band]",
            "hfo_power/hfo_peak_freq (block active only if cohort-uniform rule permits)",
        ],
        "description": "REST-block spectral features over active bands",
    },
    "rest_dynamics": {
        "features": [
            "hjorth_mobility", "hjorth_complexity", "line_length_per_s",
            "spectral_entropy", "beta_burst_occupancy",
            "beta_burst_mean_duration_ms", "beta_burst_median_amplitude_ratio",
        ],
        "description": "REST-block dynamics features",
    },
    "delta_spectral": {
        "features": [
            "log_ratio_band_power[band]",
            "aperiodic_exponent_change",
            "aperiodic_adjusted_band_power_change[band]",
        ],
        "description": "movement-minus-rest spectral modulation",
    },
    "delta_dynamics": {
        "features": [
            "hjorth_mobility_delta", "hjorth_complexity_delta",
            "line_length_per_s_delta", "spectral_entropy_delta",
            "beta_burst_occupancy_delta", "beta_burst_mean_duration_ms_delta",
        ],
        "description": "movement-minus-rest dynamics modulation",
    },
    "combined": {
        "features": ["rest_spectral + rest_dynamics + clinical"],
        "description": "combined REST block",
    },
    "delta_combined": {
        "features": ["delta_spectral + delta_dynamics + clinical"],
        "description": (
            "combined DELTA block (§11 'DELTA analogously'; added to the "
            "implemented block set in the v1.0.1 corrective revision — "
            "the v1.0.0 integration run never built it)"
        ),
    },
}

#: MOVE-arm representation ablation (window-level).
ABLATION_MOVE_BLOCKS: Final[tuple[str, ...]] = (
    "multiband_power",
    "dmd_spatial_nodes (frozen rank 8, min(8, n_channels) dimensions; "
    "secondary ablation: DMD spatial edges)",
    "multiband_power + dmd_spatial_nodes (combined)",
)

# ---------------------------------------------------------------------------
# Methodology-transfer and interpretation criteria (frozen)
# ---------------------------------------------------------------------------

TRANSFER_CRITERIA: Final[dict[str, str]] = {
    "movement_state_transfer": (
        "MOVE transfer is declared successful (descriptive, exploratory) iff "
        "the primary lagged Ridge decoder achieves (a) median across-subject "
        "within-subject Pearson r > 0 with a subject-cluster bootstrap 95% CI "
        "excluding 0, AND (b) median across-subject MAE better than BOTH the "
        "persistence and block-mean baselines on identical outer folds"
    ),
    "resting_trait_evidence": (
        "REST evidence requires ridge_primary (combined REST block) to meet "
        "the full promotion rule AND a Holm-adjusted permutation p < 0.05 on "
        "MAE or RMSE — hypothesis-generating only"
    ),
    "state_change_evidence": (
        "DELTA evidence requires the DELTA combined block to beat the REST "
        "combined block on BOTH MAE and RMSE on identical outer folds AND "
        "meet the promotion rule versus baselines"
    ),
    "global_null": (
        "if none of the three evidence criteria is met, the outcome is "
        "reported as a global null under this contract version"
    ),
    "no_claims": (
        "no clinical utility, DBS-outcome prediction, or causal claims; "
        "thousands of windows are not thousands of independent participants"
    ),
}

# ---------------------------------------------------------------------------
# Stage rules: confirmatory / exploratory / post-hoc (frozen)
# ---------------------------------------------------------------------------

STAGE_RULES: Final[dict[str, str]] = {
    "dabi_cohort_stage": "exploratory",
    "dabi_cohort_reason": (
        "the cohort's targets and basic structure were already inspected in "
        "the recorded first authorized run; this freeze makes the follow-on "
        "protocol prespecified, not confirmatory — every follow-on result on "
        "this cohort is exploratory regardless of freezing"
    ),
    "confirmatory_requires": (
        "an external, independent cohort, evaluated under an UNCHANGED "
        "contract version, pre-registered before any inspection of that "
        "cohort"
    ),
    "post_hoc_label": "post_hoc",
    "post_hoc_definition": (
        "any target, band, model, kernel, channel, lag, threshold, or "
        "analysis choice made after inspecting follow-on outputs, or any "
        "analysis outside this document — labeled post_hoc in outputs, "
        "reported separately, never eligible for promotion or transfer claims"
    ),
    "version_change_rule": (
        "changes require a new protocol_version and a changelog entry in "
        "docs/followon_protocol.md; every run records protocol_version and "
        "protocol_hash; results from different versions are never mixed in "
        "one primary table"
    ),
}

# ---------------------------------------------------------------------------
# Provenance and privacy (frozen)
# ---------------------------------------------------------------------------

PROVENANCE_LABEL: Final[str] = (
    "DABI — authorized local copy; follow-on three-arm protocol; "
    "NOT DBS outcome"
)
PRIVACY_RULES: Final[dict[str, str]] = {
    "per_subject_outputs": (
        "all per-subject predictions, features, fold assignments, fitted "
        "models, and MLflow stores stay in gitignored local directories "
        "(data/, outputs/, artifacts/) or other local paths — never committed"
    ),
    "tracked_aggregates": (
        "only sanitized aggregates whose publication the recorded terms "
        "clearly permit may be committed, with provenance stated"
    ),
    "synthetic_tests": "tests use synthetic fixtures only",
}

# ---------------------------------------------------------------------------
# Unresolved data-layout issues (recorded, never guessed)
# ---------------------------------------------------------------------------

UNRESOLVED_DATA_LAYOUT: Final[tuple[str, ...]] = (
    "per-recording ECoG and glove sampling rates (governs "
    "resolve_spectral_bands and the HFO gate; read from acquisition "
    "metadata at conversion time, not guessed)",
    "glove channel count, units, and any cross-subject calibration "
    "(currently assumed absent — amplitude targets stay within-subject-only)",
    "channel→anatomical-region labels availability (aggregation fallback is "
    "the frozen default until a mapping is inspected)",
    "electrode adjacency for bipolar rereferencing (fallback common median "
    "reference until provided)",
    "number and duration of REST/MOVE blocks per recording (governs the "
    "minimum-usable-data gates and MOVE window counts)",
    "whether raw tables share the restmove_segs sample grid per modality "
    "(the .npz interchange contract requires a common grid, resolved at "
    "conversion time)",
    "clinical workbook fields beyond age/dbs_volt/dbs_freq (the clinical "
    "block stays frozen at three covariates until new fields are "
    "documented in a contract version bump)",
)

# ---------------------------------------------------------------------------
# Document assembly, hashing, validation
# ---------------------------------------------------------------------------

_SEMVER: Final[re.Pattern] = re.compile(r"^\d+\.\d+\.\d+$")


def protocol_document() -> dict:
    """The full frozen contract as one JSON-serializable dict.

    This document IS the protocol; its canonical JSON encoding is what
    :func:`protocol_hash` fingerprints.
    """
    return {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_date": FROZEN_DATE,
        "frozen_before_any_followon_result": True,
        "revisions": [dict(revision) for revision in REVISIONS],
        "headline": (
            "Three-arm follow-on protocol: REST, MOVE, DELTA — frozen before "
            "any follow-on real-data result was inspected"
        ),
        "basis": {
            "owner_instruction": (
                "docs/neural_decoding_implementation_prompts.md — Prompt 0, "
                "2026-11-16"
            ),
            "literature_review": "docs/literature_review.md",
            "preserves_first_result": (
                "the recorded first authorized real-data result (frozen "
                "five-feature kinematic analysis) is immutable history; this "
                "contract adds follow-on analyses only"
            ),
        },
        "arms": {
            name: {
                "definition": ARM_DEFINITIONS[name],
                "output_level": (
                    "subject" if name in ("REST", "DELTA") else "window_within_subject_holdout"
                ),
                "targets": {
                    "REST": [PRIMARY_SUBJECT_TARGET, *SECONDARY_SUBJECT_TARGETS],
                    "MOVE": [MOVE_CONTINUOUS_TARGET],
                    "DELTA": [PRIMARY_SUBJECT_TARGET, *SECONDARY_SUBJECT_TARGETS],
                }[name],
            }
            for name in ARMS
        },
        "targets": {
            "primary_subject": {
                "name": PRIMARY_SUBJECT_TARGET,
                "definition": (
                    "median dominant 0.5–4.0 Hz glove-channel-block tapping "
                    "frequency; implementation frozen in "
                    "curve_icon_dbso.kinematics — unchanged by this contract"
                ),
            },
            "secondary_subject": {
                "tap_interval_cv": (
                    "per contributing channel-block: SD/mean of successive "
                    f"cycle intervals with ≥ {TAP_MIN_CYCLES_PER_BLOCK} "
                    "intervals; median across contributing channel-blocks; "
                    f"≥ {TAP_MIN_CHANNEL_BLOCKS} contributing channel-blocks "
                    "required"
                ),
                "tap_vigor_index": (
                    "per contributing channel-block: RMS(diff(filtered)) / "
                    "(2π · f_tap · RMS(filtered)) on the "
                    f"{TAP_DETECTION_BAND_HZ[0]}–{TAP_DETECTION_BAND_HZ[1]} "
                    "Hz zero-phase filtered channel — dimensionless, "
                    "amplitude-independent vigor; median across contributing "
                    "channel-blocks"
                ),
                "tap_sequence_slope": (
                    "per contributing channel-block: OLS slope of per-cycle "
                    "amplitude vs cycle index normalized by the block's mean "
                    "amplitude (dimensionless per-cycle decay); ≥ "
                    f"{TAP_MIN_CYCLES_PER_BLOCK} cycles; median across "
                    "contributing channel-blocks"
                ),
            },
            "within_subject_only": {
                "tap_amplitude": (
                    "recorded for QC only — glove uncalibrated across "
                    "subjects, excluded from all cross-subject modeling"
                )
            },
            "move_continuous": {"name": MOVE_CONTINUOUS_TARGET,
                                "definition": MOVE_ENVELOPE_RULE},
            "frozen_floors": {
                "tap_rate_low_hz": TAP_RATE_LOW_HZ,
                "tap_rate_high_hz": TAP_RATE_HIGH_HZ,
                "tap_prominence_floor": TAP_PROMINENCE_FLOOR,
                "tap_min_channel_blocks": TAP_MIN_CHANNEL_BLOCKS,
                "tap_welch_segment_seconds": TAP_WELCH_SEGMENT_SECONDS,
                "tap_min_cycles_per_block": TAP_MIN_CYCLES_PER_BLOCK,
                "tap_detection_band_hz": list(TAP_DETECTION_BAND_HZ),
                "tap_min_peak_separation_s": TAP_MIN_PEAK_SEPARATION_S,
                "tap_detection_filter_order": TAP_DETECTION_FILTER_ORDER,
            },
        },
        "neural_features": {
            "bands_standard": {k: list(v) for k, v in STANDARD_BANDS_HZ.items()},
            "gamma_high_default": list(HIGH_GAMMA_DEFAULT),
            "gamma_high_alternative": {
                "range": list(HIGH_GAMMA_ALTERNATIVE),
                "status": (
                    "recorded only — adoption requires a new contract version "
                    f"and cohort fs ≥ {HIGH_GAMMA_ALTERNATIVE_MIN_FS_HZ} Hz; "
                    "never auto-selected"
                ),
            },
            "hfo_block": {"range": list(HFO_BAND_HZ)},
            "band_convention": (
                "half-open [low, high) over Welch PSD bin centers; the "
                "topmost active edge is inclusive; Nyquist margin "
                f"{NYQUIST_MARGIN_HZ} Hz"
            ),
            "band_resolution_rule": (
                "resolve_spectral_bands(min cohort fs) — deterministic, "
                "acquisition-driven, cohort-uniform"
            ),
            "transforms": dict(FEATURE_TRANSFORMS),
            "dynamics": dict(FEATURE_DYNAMICS),
            "burst_threshold_percentile": BURST_THRESHOLD_PERCENTILE,
            "burst_min_duration_ms": BURST_MIN_DURATION_MS,
            "delta_rules": dict(DELTA_RULES),
        },
        "qc_rules": dict(QC_RULES),
        "min_usable_data": {"rest_s": MIN_REST_DATA_S, "move_s": MIN_MOVE_DATA_S},
        "aggregation": dict(AGGREGATION),
        "region_labels": list(REGION_LABELS),
        "windowing_move": {
            "window_s": MOVE_WINDOW_S,
            "update_s": MOVE_UPDATE_S,
            "lags_s": list(MOVE_LAGS_S),
            "lags_semantics": MOVE_LAGS_SEMANTICS,
            "rule": MOVE_WINDOW_RULE,
        },
        "validation": {
            "outer_unit": "subject",
            "outer_rule": (
                f"LOSO for ≤ {LOSO_MAX_SUBJECTS} subjects, else "
                f"GroupKFold({OUTER_GROUPKFOLD_SPLITS}) — identical to the "
                "implemented evaluation splitter; identical outer folds "
                "across every model, feature block, and arm"
            ),
            "loso_max_subjects": LOSO_MAX_SUBJECTS,
            "outer_groupkfold_splits": OUTER_GROUPKFOLD_SPLITS,
            "inner_splits": INNER_SPLITS,
            "inner_loso_min_outer_train_subjects": INNER_LOSO_MIN_OUTER_TRAIN_SUBJECTS,
            "inner_selection_criterion": INNER_SELECTION_CRITERION,
            "inner_tie_rule": INNER_TIE_RULE,
            "window_fold_rule": (
                "MOVE arm: every window of a subject stays in exactly one "
                "outer fold (subject-keyed window→subject mapping)"
            ),
        },
        "grid_policy": {
            "max_combos_per_model": MAX_GRID_COMBOS,
            "selection": (
                "all imputation, scaling, reduction, and hyperparameter "
                "selection is fitted on inner-training subjects only"
            ),
        },
        "models_subject_level": MODELS_SUBJECT_LEVEL,
        "models_move": MODELS_MOVE,
        "metrics_subject_level": list(METRICS_SUBJECT_LEVEL),
        "metrics_move": dict(METRICS_MOVE),
        "baselines_subject_level": list(BASELINES_SUBJECT_LEVEL),
        "frozen_five_feature_reference": FROZEN_FIVE_FEATURE_REFERENCE,
        "baselines_move": dict(BASELINES_MOVE),
        "promotion_rule": dict(PROMOTION_RULE),
        "holm_family_rule": HOLM_FAMILY_RULE,
        "uncertainty": {
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "permutation_min_for_reporting": PERMUTATION_MIN_FOR_REPORTING,
            "permutation_floor": PERMUTATION_FLOOR,
            "permutation_seed": PERMUTATION_SEED,
            "move_null_shifts": MOVE_NULL_SHIFTS,
            "move_circular_shift_min_s": MOVE_CIRCULAR_SHIFT_MIN_S,
            "bootstrap_caveat": (
                "percentile intervals undercover at pilot n; they describe "
                "this cohort, not a population"
            ),
        },
        "null_procedures": dict(NULL_PROCEDURES),
        "ablation_blocks_subject_level": ABLATION_BLOCKS,
        "ablation_blocks_move": list(ABLATION_MOVE_BLOCKS),
        "transfer_criteria": dict(TRANSFER_CRITERIA),
        "stage_rules": dict(STAGE_RULES),
        "provenance": {"label": PROVENANCE_LABEL, "privacy": dict(PRIVACY_RULES)},
        "unresolved_data_layout": list(UNRESOLVED_DATA_LAYOUT),
    }


def canonical_json() -> str:
    """Canonical JSON encoding (sorted keys, compact separators)."""
    return json.dumps(protocol_document(), sort_keys=True, separators=(",", ":"))


def protocol_hash() -> str:
    """SHA-256 fingerprint of the canonical contract document.

    Recorded in every follow-on run's provenance alongside
    ``protocol_version``; pinned by ``tests/test_protocol_contract.py`` so
    a silent change to any frozen quantity fails CI.
    """
    return hashlib.sha256(canonical_json().encode("utf-8")).hexdigest()


def validate_protocol() -> None:
    """Enforce structural invariants of the frozen contract.

    Raises:
        ValueError: On any structural violation (ordering, bounds, floors,
            grid sizes, semver shape, identifier leakage). The pinned-hash
            test catches *content* changes; this catches a contract whose
            individual values are internally inconsistent even after a
            deliberate version bump.
    """
    doc = protocol_document()

    if not _SEMVER.match(PROTOCOL_VERSION):
        raise ValueError(f"PROTOCOL_VERSION must be semver, got {PROTOCOL_VERSION!r}")
    if tuple(doc["arms"]) != ARMS or set(ARMS) != {"REST", "MOVE", "DELTA"}:
        raise ValueError(f"arms must be exactly {('REST', 'MOVE', 'DELTA')}")

    # Bands: positive widths, ordered low→high, disjoint, inside (0, 500].
    # The disjointness ladder covers the standard bands + the default high
    # gamma; the HFO block is a SEPARATE legacy block that intentionally
    # overlaps high gamma (its features are never summed with band powers),
    # so it is bounds-checked on its own, as is the recorded alternative.
    ladder = sorted(
        ((name, (float(low), float(high)))
         for name, (low, high) in {**STANDARD_BANDS_HZ,
                                   "gamma_high": HIGH_GAMMA_DEFAULT}.items()),
        key=lambda item: item[1][0],
    )
    previous_high = 0.0
    for name, (low, high) in ladder:
        if not (0.0 < low < high <= 500.0):
            raise ValueError(f"band {name} has invalid edges: [{low}, {high}]")
        if low < previous_high:
            raise ValueError(f"band {name} overlaps the preceding band at {previous_high} Hz")
        previous_high = high
    for name, (low, high) in (("hfo", HFO_BAND_HZ),
                              ("gamma_high_alt", HIGH_GAMMA_ALTERNATIVE)):
        if not (0.0 < low < high <= 500.0):
            raise ValueError(f"{name} range has invalid edges: [{low}, {high}]")

    # Grid bounds.
    for family in (MODELS_SUBJECT_LEVEL, MODELS_MOVE):
        for name, spec in family.items():
            size = grid_size(spec)
            if size > MAX_GRID_COMBOS:
                raise ValueError(
                    f"model '{name}' grid has {size} combinations, over the "
                    f"frozen ceiling of {MAX_GRID_COMBOS}"
                )
            for param, values in spec.get("grid", {}).items():
                if not values or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError(f"model '{name}' grid param '{param}' is invalid")

    # Validation floors.
    if doc["validation"]["outer_unit"] != "subject":
        raise ValueError("the outer evaluation unit must always be subject")
    if LOSO_MAX_SUBJECTS < 2 or OUTER_GROUPKFOLD_SPLITS < 2 or INNER_SPLITS < 2:
        raise ValueError("fold counts must be ≥ 2")
    if INNER_LOSO_MIN_OUTER_TRAIN_SUBJECTS < 2 * INNER_SPLITS:
        raise ValueError(
            "the inner-LOSO fallback threshold must leave at least 2 subjects "
            "per inner fold when GroupKFold applies"
        )

    # Uncertainty floors.
    u = doc["uncertainty"]
    if u["bootstrap_n_resamples"] < 2000:
        raise ValueError("protocol bootstrap floor is 2,000 resamples")
    if u["permutation_min_for_reporting"] < 1000:
        raise ValueError("protocol permutation floor for reported p-values is 1,000")
    if u["permutation_floor"] < 200:
        raise ValueError("protocol permutation floor is 200")
    if u["move_null_shifts"] < 1000:
        raise ValueError("MOVE null floor is 1,000 shifts")
    if not (0.0 < u["ci_level"] < 1.0):
        raise ValueError("CI level must be in (0, 1)")
    if u["move_circular_shift_min_s"] <= 0:
        raise ValueError("the minimum circular shift must be positive")

    # Windowing bounds and causality (frozen prompt ranges).
    w = doc["windowing_move"]
    if not (0.5 <= w["window_s"] <= 1.0):
        raise ValueError("MOVE window must be within 0.5–1.0 s")
    if not (0.1 <= w["update_s"] <= 0.25):
        raise ValueError("MOVE update must be within 0.1–0.25 s")
    lags = [float(x) for x in w["lags_s"]]
    if any(lag < 0.0 for lag in lags):
        raise ValueError("preceding-context offsets must be non-negative")
    if max(lags) > 0.5:
        raise ValueError("neural context must not exceed 500 ms")
    if 0.0 not in lags:
        raise ValueError("the lag stack must include the window ending at the target")

    # Stage discipline.
    if doc["stage_rules"]["dabi_cohort_stage"] != "exploratory":
        raise ValueError("the DABI cohort's stage rule must remain exploratory")

    # No subject identifiers anywhere in the document.
    if re.search(r"DBS_bG\d", canonical_json()):
        raise ValueError("the contract document must never contain subject identifiers")


# Import-time structural check: a contract that violates its own invariants
# fails loudly the moment any consumer imports it.
validate_protocol()

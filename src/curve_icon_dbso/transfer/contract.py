"""Machine-readable v2 adaptive-transfer contract (version 2.1.0, frozen).

Frozen BEFORE any v2 real-data execution (owner decisions on the four
Phase-0 design questions recorded 2026-11-19; feasibility audit the
same day; pre-execution integrity refinement 2.1.0 the same day —
review findings, none from real data). Mirrored in prose by
``docs/adaptive_transfer_protocol.md``; pinned by
``tests/test_transfer_contract.py`` so a silent change to any frozen
quantity fails the suite. Changes require a new semver and an
append-only changelog entry in the protocol document.

Relationship to v1.0.1: this contract is additive and independent. The
v1.0.1 contract (``curve_icon_dbso.protocol``) and all of its recorded
results are immutable history; nothing here re-fits or re-scores them.
"""

from __future__ import annotations

import hashlib
import json
import re

# ---------------------------------------------------------------------------
# Identity and stage
# ---------------------------------------------------------------------------

CONTRACT_VERSION = "2.1.0"
STAGE = "exploratory"
PROTOCOL_DOC = "docs/adaptive_transfer_protocol.md"
COHORT_LABEL = "DABI authorized local copy — 31 converted recordings"
#: The v1.0.1 contract version this study must NOT modify (recorded history).
V1_CONTRACT_VERSION = "1.0.1"

# ---------------------------------------------------------------------------
# Cohort flow and exclusion rules (frozen)
# ---------------------------------------------------------------------------

MIN_MOVE_S = 30.0            # usable MOVE seconds per recording (v1.0.1 §4 floor)
MIN_REST_S = 5.0             # usable REST seconds (detection needs both classes)
MIN_TEST_S = 55.0            # untouched final test segment (owner decision, 2026-11-19)
MIN_CALIBRATION_S = 8.0      # shortest calibration prefix accepted (owner decision)
CALIBRATION_FRACTIONS = (0.10, 0.20, 0.30)  # chronological MOVE prefixes, nested
SIGNAL_QUALITY_FLAG_EXCLUDES = False  # flagged + reported per subject; never an exclusion
#: A track whose eligible-cohort fraction falls below this is reported
#: not-executable — never rescued by substituting models, bands, or targets.
MAX_INELIGIBLE_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Spatial representation (owner decision: grid-relative, anchor-anchored)
# ---------------------------------------------------------------------------

SPATIAL_REPRESENTATION = "grid_relative_anchor"
#: Anchor-relative ordinal positions on strip 1 (the anchored strip).
GRID_OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
ANCHOR_OFFSET = 0
#: Dual-strip recordings: the second strip's anchor assignment is
#: undocumented (feasibility audit 2026-11-19), so strip 2 is excluded
#: from the primary grid and never silently median-pooled into it.
SECOND_STRIP_POLICY = "excluded_unanchored"
#: Per-offset missingness indicators + per-window coverage fraction;
#: anatomical/positional absence is NEVER replaced by a cross-channel median.
MISSINGNESS_FEATURES = True
#: Per-recording, target-independent rereference (v1.0.1 §5 documented fallback).
REREFERENCE = "common_median"
#: Single prespecified spatial-representation ablation: adjacent-contact
#: bipolar pairs on the anchored strip (7 pairs for an 8-contact strip).
BIPOLAR_ABLATION = "adjacent_pairs_strip1"
#: The workbook motor-anchor position maps to ordinal position
#: ``anchor_position - 1`` on strip 1 (P-label convention).
ANCHOR_POSITION_RANGE = (1, 8)

# ---------------------------------------------------------------------------
# Windows, updates, lags, bands (frozen)
# ---------------------------------------------------------------------------

WINDOW_S_PRIMARY = 0.5
WINDOW_S_ABLATION = 1.0      # the single prespecified window comparison
UPDATE_S = 0.1
LAG_STACK_S = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)   # preceding neural context (regression)
DETECTION_LAGS_S = (0.0,)    # detection reads the window itself, no lag stack
#: Detection windows within this distance of a REST/MOVE edge are dropped
#: (transition dynamics are neither clean rest nor steady movement).
#: ENFORCED IN MACHINERY (v2.1.0): every retained feature row must read
#: only samples at least this far inside its (span-clipped) block edges.
BOUNDARY_GUARD_S = 1.0
#: Offline retrospective processing (owner decision, 2026-11-19):
#: calibration and test spans are processed INDEPENDENTLY end to end —
#: ECoG rereferencing/line treatment, features, and glove targets are all
#: built on clipped spans — so no sample of one span can alter another
#: span's artifacts. Zero-phase filtering remains allowed WITHIN a span
#: and is documented as retrospective; this contract makes no online or
#: causal-inference claim.
SPAN_LOCAL_PROCESSING = True
#: Both sides of every clipped target-segment edge are excluded by this
#: margin before bins are kept: it covers the 0.5 Hz-band-edge Hilbert
#: transient (1 / 0.5 Hz = 2 s) plus the band-pass and smoothing-kernel
#: transients, so a bin's envelope value depends only on samples strictly
#: interior to its own span.
TARGET_EDGE_EXCLUSION_S = 2.0
#: Cohort fs = 1200 Hz → Nyquist 600 Hz; every band below is active
#: (top edge ≤ fs/2 − 10 Hz activation rule, v1.0.1 §3 convention).
BANDS_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 35.0),
    "gamma_low": (35.0, 60.0),
    "gamma_high": (60.0, 80.0),
    "hfa": (80.0, 200.0),
}
#: Bands are a disjoint partition of 4–200 Hz; content above 200 Hz is
#: excluded a priori (the distributed recording is pre-filtered with
#: undocumented upstream filtering — recorded caveat, never guessed).
BAND_TOP_HZ_MAX = 200.0
BAND_GROUP_LOW = ("theta", "alpha", "beta_low", "beta_high")
BAND_GROUP_HIGH = ("gamma_low", "gamma_high", "hfa")
LINE_NOISE_RULE = "notch_detected_line_plus_harmonics"  # v1.0.1 §4 rule, reused

# ---------------------------------------------------------------------------
# Feature construction and normalization (frozen)
# ---------------------------------------------------------------------------

FEATURE_KIND = "log10_relative_band_power"   # per grid cell × band × lag
APERIODIC_PER_CELL = True                    # aperiodic exponent per cell × lag
WELCH_SEGMENT_S = 0.5
#: Causal trailing z-score per cell×band feature stream, past updates only.
CAUSAL_NORM_TRAILING_S = 5.0
CAUSAL_CLIP = 5.0
#: Zero-shot causal normalization uses PAST UNLABELED NEURAL samples only
#: (explicitly allowed by the owner brief); no labels, no target data.
ZERO_SHOT_CAUSAL_NORM_ALLOWED = True
#: Cross-subject model-side preprocessing: fold-local median imputation +
#: standardization, fitted on outer-training subjects only.
MODEL_SIDE_PREPROCESSING = "fold_local_median_impute_standardize"
#: Calibration→test purge. Minimum is max window (1.0 s) + max lag
#: (0.5 s) = 1.5 s; the conservative prespecified value below applies.
PURGE_S = 2.0

# ---------------------------------------------------------------------------
# Models and bounded grids (frozen)
# ---------------------------------------------------------------------------

RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DETECTION_MODEL = "ridge_logistic_class_weighted"
INNER_CV = "GroupKFold(3)"
INNER_SELECTION_DETECTION = "mean_balanced_accuracy"
INNER_SELECTION_REGRESSION = "mean_mae"
XGB_FIXED_PARAMS = {
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 0.05,
    "n_jobs": 1,
    "random_state": 42,
}
#: Kinematic autoregressive baseline: least-squares AR on the standardized
#: target stream, 1.0 s of history (10 update bins), fitted on
#: outer-training subjects only — never on held-out data, never on future
#: data. AR and persistence histories RESET at every MOVE-block boundary
#: (v2.1.0): a prediction never conditions on samples from a previous
#: block, across a purge, or across a REST gap.
AR_ORDER_BINS = 10
#: Few-shot subject-specific fit (v2.0.1): a CALIBRATION-SELECTED BLEND.
#: The transferred zero-shot residual model gives the base correction;
#: the calibration segment chooses — via chronological blocked CV
#: INSIDE the calibration segment only (never the final test interval) —
#: among {pure kinematic AR, transferred residual only, transferred
#: residual + anchored calibration correction, full calibration
#: correction}. The correction Ridge's alpha is likewise selected by
#: blocked CV inside the calibration segment; the zero-shot-selected
#: alpha is deliberately not reused. Detection analog: logistic re-fit
#: with the selected C fixed.
FEWSHOT_REFIT_RULE = "calibration_blend_selection"

# ---------------------------------------------------------------------------
# Validation, uncertainty, nulls (frozen)
# ---------------------------------------------------------------------------

OUTER_CV = "LOSO"  # true leave-one-subject-out, prespecified (v2 change from v1's GroupKFold(5); rationale in protocol §7)
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
DETECTION_NULL = "within_subject_block_kind_permutation"
DETECTION_NULL_DRAWS = 1000
REGRESSION_NULL_CIRCULAR = "circular_shift_effective_displacement_min_2s"  # v1.0.1 corrected convention
REGRESSION_NULL_BLOCKWISE = "blockwise_2s_segments"
REGRESSION_NULL_DRAWS = 1000
NULL_SEED = 42
PERMUTATION_CONVENTION = "conservative_p=(1+#{>= observed})/(1+n)"

# ---------------------------------------------------------------------------
# Endpoints, multiplicity, success criteria (frozen)
# ---------------------------------------------------------------------------

#: Zero-shot regression is scored with scale-free metrics only (per-subject
#: correlation between predicted and observed streams): per-recording target
#: standardization cannot be fitted on the final test interval, and no
#: held-out-subject labeled data exists in zero-shot. MAE-family claims are
#: few-shot endpoints, where calibration-fitted scale constants exist.
ZERO_SHOT_REGRESSION_METRICS = ("per_subject_pearson_r", "per_subject_spearman_rho")
#: Track B's quantitative primary lives in the 30% few-shot condition.
FEWSHOT_PRIMARY_FRACTION = 0.30

PRIMARY_FAMILY = (
    "detection_zero_shot_median_bac",        # Track A + C zero-shot detection
    "regression_zero_shot_median_r",         # Track C zero-shot regression
    "fewshot30_delta_mae",                   # Track B primary
    "fewshot30_gain_over_zero_shot",         # Track C few-shot value
)
MULTIPLICITY_RULE = "holm_within_primary_family_then_holm_within_secondary_families"
#: The primary family is adjusted EXACTLY ONCE: each endpoint contributes
#: its RAW permutation/null p-value; no pre-adjusted p enters the family
#: (the detection model-family Holm over {Ridge, XGBoost} is descriptive
#: only and never feeds the primary family — v2.1.0 clarification).
PRIMARY_FAMILY_SINGLE_ADJUSTMENT = True
#: Every displayed verdict Boolean is derived from the FINAL adjusted
#: result: a primary verdict requires its criterion AND its
#: primary-family Holm p < 0.05, so per-track verdicts can never
#: contradict ``global_null`` (v2.1.0 clarification).
VERDICTS_FROM_FINAL_ADJUSTMENT = True

SUCCESS_CRITERIA = {
    "detection_transfer": (
        "median subject-level balanced accuracy > 0.5 with subject-cluster "
        "bootstrap 95% CI excluding 0.5, majority-class baseline exceeded, and "
        "Holm-adjusted within-primary-family block-permutation p < 0.05"
    ),
    "regression_zero_shot": (
        "median per-subject Pearson r > 0 with bootstrap CI excluding 0 and "
        "Holm-adjusted circular-shift null p < 0.05"
    ),
    "fewshot_delta_mae": (
        "median paired delta-MAE (combined − kinematic AR baseline) < 0 in the "
        "30% condition with subject-cluster bootstrap CI excluding 0 and "
        "Holm-adjusted shift-null p < 0.05"
    ),
    "fewshot_gain": (
        "30%-condition performance gain over the zero-shot endpoint with "
        "bootstrap CI excluding 0 (detection: median BAC gain; regression: "
        "median Fisher-z r gain)"
    ),
    "event_association": (
        "two-stage subject-level association for the primary event outcome with "
        "Fisher-z bootstrap CI excluding 0, Holm-adjusted within the event "
        "secondary family"
    ),
    "event_prediction": (
        "held-out-subject delta-MAE of the neural event model versus the "
        "event-level AR baseline, CI excluding 0 and shift-null Holm p < 0.05"
    ),
    "global_null": "none of the above",
}

SECONDARY_FAMILY_DETECTION = ("auroc", "sensitivity", "specificity", "event_detection_rate")
SECONDARY_FAMILY_REGRESSION = ("incremental_r2", "innovation_r", "rmse", "per_subject_spread")
SECONDARY_FAMILY_ABLATIONS = (
    "window_1s_vs_500ms", "bipolar_vs_referential", "lag_subsets", "band_groups",
)
EVENT_PRIMARY_OUTCOME = "log_inter_tap_interval"
EVENT_SECONDARY_FAMILY = (
    "iti_cv", "within_block_amplitude_decrement", "tap_amplitude_z", "peak_velocity_z",
)
EVENT_NEURAL_FEATURE = "anchor_cell_hfa_log_power"
#: Association and prediction are reported as separate claims; the number of
#: taps never increases the independent subject sample size.
EVENT_SUBJECT_UNIT_RULE = "subject_is_replication_unit"
#: The event p-value family (v2.1.0): Holm over the association tests of
#: the per-tap outcomes plus the event-prediction shift null — all
#: adjusted once, together. The two block-level outcomes (iti_cv,
#: within_block_amplitude_decrement) have no prespecified per-tap
#: association test and therefore no p-value; they are reported
#: descriptively and are never silent members of a p-value family.
EVENT_P_VALUE_FAMILY = (
    "log_inter_tap_interval_association",
    "tap_amplitude_z_association",
    "peak_velocity_z_association",
    "event_prediction_shift_null",
)
#: Event-prediction success additionally requires the subject-cluster
#: bootstrap CI of the median paired delta-MAE to exclude 0 (frozen
#: criterion text; enforced in machinery since v2.1.0).
EVENT_PREDICTION_REQUIRES_BOOTSTRAP_CI = True

STOPPING_RULES = {
    "machinery_failure": (
        "a track with fewer than 50% of eligible recordings passing its data "
        "gates is reported not-executable; no model/band/target substitution"
    ),
    "no_result_driven_stopping": (
        "no track stops early for performance reasons; the full frozen matrix "
        "is executed and reported regardless of interim results"
    ),
    "protocol_amendment": (
        "any change after seeing v2 real-data results resets the affected "
        "analysis to post_hoc and requires a contract version bump"
    ),
}

PROHIBITED_POST_HOC = (
    "changing targets, bands, models, grids, lags, channels, thresholds, or "
    "spatial rules after inspecting v2 outputs",
    "selecting the calibration fraction using final-test performance",
    "presenting REST/MOVE classification as continuous-vigor decoding",
    "presenting pooled-window metrics as independent-subject evidence",
    "rerunning the v1 DMD method unchanged (a future DMD experiment must be "
    "event-aligned under a new contract)",
    "adding deep learning or contrastive representation before Tracks A–C are "
    "complete and the protocol is amended",
    "claiming physical-unit cross-subject kinematics (calibration UNKNOWN)",
    "claiming zero-shot and few-shot as one combined claim",
)

# ---------------------------------------------------------------------------
# Definitive-run gate (v2.1.0): only a run that satisfies EVERY frozen
# requirement below may be launched in definitive mode, named ``run_…``
# (not ``smoke_…``), and write the public results document. Any deviation —
# altered null draws, skipped ablations, skipped comparators — forces the
# non-definitive path, regardless of the draw count.
# ---------------------------------------------------------------------------

DEFINITIVE_RUN_REQUIREMENTS = {
    "detection_null_draws": DETECTION_NULL_DRAWS,
    "regression_null_draws": REGRESSION_NULL_DRAWS,
    "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
    "ablations_required": True,
    "xgboost_comparators_required": True,
    "public_results_doc_only_in_definitive_mode": True,
}

# ---------------------------------------------------------------------------
# Reporting requirements (frozen)
# ---------------------------------------------------------------------------

REPORTING_REQUIREMENTS = {
    "real_results_doc": "docs/results_adaptive_transfer_real.md",
    "machine_readable_summary": "aggregate results JSON (no subject identifiers)",
    "cohort_flow_artifact": True,
    "per_track_baseline_comparisons": True,
    "calibration_curve": True,
    "subject_cluster_cis": True,
    "corrected_permutation_results": True,
    "spatial_coverage_summary": True,
    "explicit_verdicts": "every frozen criterion gets a success/failure Boolean",
    "distinctions": (
        "detection vs regression; zero-shot vs few-shot; neural-only vs "
        "kinematic-only vs combined; association vs prediction; exploratory "
        "vs confirmatory; statistical evidence vs practical usefulness"
    ),
}
PRIVACY_RULES = {
    "never_commit": ("raw data", "derived per-subject values", "subject identifiers",
                     "coordinates", "filenames of the local copy"),
    "aggregate_only": "committed docs carry cohort-level aggregates exclusively",
    "v1_immutable": "v1.0.1 contract, results, and provenance are never modified",
}

#: Reused v1 machinery (recorded so the contract is explicit about what is
#: passthrough vs new): signal QC + rereferencing (signal_qc), Welch +
#: spectral features (neural_features), causal window math (move_modeling
#: off-by-one contract), glove envelope target + tap cycles
#: (glove_phenotyping/kinematics), subject-cluster bootstrap + conservative
#: permutation convention (bootstrap), fixed XGBoost params (evaluation).
REUSED_MODULES = (
    "curve_icon_dbso.signal_qc", "curve_icon_dbso.neural_features",
    "curve_icon_dbso.move_modeling", "curve_icon_dbso.glove_phenotyping",
    "curve_icon_dbso.kinematics", "curve_icon_dbso.bootstrap",
    "curve_icon_dbso.evaluation",
)


def contract_document() -> dict:
    """The complete frozen contract as one JSON-serializable document."""
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": STAGE,
        "protocol_doc": PROTOCOL_DOC,
        "cohort_label": COHORT_LABEL,
        "v1_contract_version_immutable": V1_CONTRACT_VERSION,
        "cohort_flow": {
            "min_move_s": MIN_MOVE_S,
            "min_rest_s": MIN_REST_S,
            "min_test_s": MIN_TEST_S,
            "min_calibration_s": MIN_CALIBRATION_S,
            "calibration_fractions": list(CALIBRATION_FRACTIONS),
            "signal_quality_flag_excludes": SIGNAL_QUALITY_FLAG_EXCLUDES,
            "max_ineligible_fraction": MAX_INELIGIBLE_FRACTION,
        },
        "spatial": {
            "representation": SPATIAL_REPRESENTATION,
            "grid_offsets": list(GRID_OFFSETS),
            "anchor_offset": ANCHOR_OFFSET,
            "second_strip_policy": SECOND_STRIP_POLICY,
            "missingness_features": MISSINGNESS_FEATURES,
            "rereference": REREFERENCE,
            "bipolar_ablation": BIPOLAR_ABLATION,
            "anchor_position_range": list(ANCHOR_POSITION_RANGE),
        },
        "windows": {
            "window_s_primary": WINDOW_S_PRIMARY,
            "window_s_ablation": WINDOW_S_ABLATION,
            "update_s": UPDATE_S,
            "lag_stack_s": list(LAG_STACK_S),
            "detection_lags_s": list(DETECTION_LAGS_S),
            "boundary_guard_s": BOUNDARY_GUARD_S,
            "purge_s": PURGE_S,
            "span_local_processing": SPAN_LOCAL_PROCESSING,
            "target_edge_exclusion_s": TARGET_EDGE_EXCLUSION_S,
        },
        "bands": {name: list(edges) for name, edges in BANDS_HZ.items()},
        "band_top_hz_max": BAND_TOP_HZ_MAX,
        "band_group_low": list(BAND_GROUP_LOW),
        "band_group_high": list(BAND_GROUP_HIGH),
        "features": {
            "kind": FEATURE_KIND,
            "aperiodic_per_cell": APERIODIC_PER_CELL,
            "welch_segment_s": WELCH_SEGMENT_S,
            "causal_norm_trailing_s": CAUSAL_NORM_TRAILING_S,
            "causal_clip": CAUSAL_CLIP,
            "zero_shot_causal_norm_allowed": ZERO_SHOT_CAUSAL_NORM_ALLOWED,
            "model_side_preprocessing": MODEL_SIDE_PREPROCESSING,
            "line_noise_rule": LINE_NOISE_RULE,
        },
        "models": {
            "ridge_alphas": list(RIDGE_ALPHAS),
            "detection_model": DETECTION_MODEL,
            "inner_cv": INNER_CV,
            "inner_selection_detection": INNER_SELECTION_DETECTION,
            "inner_selection_regression": INNER_SELECTION_REGRESSION,
            "xgb_fixed_params": dict(XGB_FIXED_PARAMS),
            "ar_order_bins": AR_ORDER_BINS,
            "fewshot_refit_rule": FEWSHOT_REFIT_RULE,
        },
        "validation": {
            "outer_cv": OUTER_CV,
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "detection_null": DETECTION_NULL,
            "detection_null_draws": DETECTION_NULL_DRAWS,
            "regression_null_circular": REGRESSION_NULL_CIRCULAR,
            "regression_null_blockwise": REGRESSION_NULL_BLOCKWISE,
            "regression_null_draws": REGRESSION_NULL_DRAWS,
            "null_seed": NULL_SEED,
            "permutation_convention": PERMUTATION_CONVENTION,
        },
        "endpoints": {
            "zero_shot_regression_metrics": list(ZERO_SHOT_REGRESSION_METRICS),
            "fewshot_primary_fraction": FEWSHOT_PRIMARY_FRACTION,
            "primary_family": list(PRIMARY_FAMILY),
            "multiplicity_rule": MULTIPLICITY_RULE,
            "primary_family_single_adjustment": PRIMARY_FAMILY_SINGLE_ADJUSTMENT,
            "verdicts_from_final_adjustment": VERDICTS_FROM_FINAL_ADJUSTMENT,
            "secondary_family_detection": list(SECONDARY_FAMILY_DETECTION),
            "secondary_family_regression": list(SECONDARY_FAMILY_REGRESSION),
            "secondary_family_ablations": list(SECONDARY_FAMILY_ABLATIONS),
        },
        "events": {
            "primary_outcome": EVENT_PRIMARY_OUTCOME,
            "secondary_family": list(EVENT_SECONDARY_FAMILY),
            "neural_feature": EVENT_NEURAL_FEATURE,
            "subject_unit_rule": EVENT_SUBJECT_UNIT_RULE,
            "p_value_family": list(EVENT_P_VALUE_FAMILY),
            "prediction_requires_bootstrap_ci": EVENT_PREDICTION_REQUIRES_BOOTSTRAP_CI,
        },
        "success_criteria": dict(SUCCESS_CRITERIA),
        "stopping_rules": dict(STOPPING_RULES),
        "definitive_run_requirements": dict(DEFINITIVE_RUN_REQUIREMENTS),
        "prohibited_post_hoc": list(PROHIBITED_POST_HOC),
        "reporting_requirements": {
            key: (value if isinstance(value, (str, bool)) else list(value))
            for key, value in REPORTING_REQUIREMENTS.items()
        },
        "privacy_rules": {key: list(value) if isinstance(value, tuple) else value
                          for key, value in PRIVACY_RULES.items()},
        "reused_modules": list(REUSED_MODULES),
    }


_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def canonical_json() -> str:
    """Canonical JSON encoding (sorted keys, compact separators)."""
    return json.dumps(contract_document(), sort_keys=True, separators=(",", ":"))


def contract_hash() -> str:
    """SHA-256 fingerprint of the canonical v2 contract document.

    Recorded in every v2 run's provenance; pinned by
    ``tests/test_transfer_contract.py``.
    """
    return hashlib.sha256(canonical_json().encode("utf-8")).hexdigest()


def validate_contract() -> None:
    """Structural invariants (the pinned-hash test catches content drift;
    this catches internal inconsistency even after a deliberate bump)."""
    if not _SEMVER.match(CONTRACT_VERSION):
        raise ValueError(f"CONTRACT_VERSION must be semver, got {CONTRACT_VERSION!r}")
    if STAGE != "exploratory":
        raise ValueError("the v2 contract is exploratory by definition")

    # Calibration fractions: ascending, nested, inside (0, 1); the primary
    # fraction must be the largest (identical test interval rule).
    fractions = CALIBRATION_FRACTIONS
    if not (0 < fractions[0] < fractions[-1] < 1):
        raise ValueError(f"calibration fractions must ascend inside (0,1): {fractions}")
    if FEWSHOT_PRIMARY_FRACTION != fractions[-1]:
        raise ValueError("the few-shot primary fraction must be the largest fraction")

    # Bands: positive widths, ordered low→high, disjoint, within (0, BAND_TOP_HZ_MAX].
    ladder = sorted(((name, (float(low), float(high)))
                     for name, (low, high) in BANDS_HZ.items()),
                    key=lambda item: item[1][0])
    previous_high = 0.0
    for name, (low, high) in ladder:
        if not (0.0 < low < high <= BAND_TOP_HZ_MAX):
            raise ValueError(f"band {name} has invalid edges: [{low}, {high}]")
        if low < previous_high:
            raise ValueError(f"band {name} overlaps the preceding band at {previous_high} Hz")
        previous_high = high

    # Grid: anchor inside offsets; purge clears the minimum window+lag bar.
    if ANCHOR_OFFSET not in GRID_OFFSETS:
        raise ValueError("the anchor offset must be a grid position")
    if GRID_OFFSETS[0] > 0 or GRID_OFFSETS[-1] < 0:
        raise ValueError("grid offsets must span the anchor on both sides")
    min_purge = WINDOW_S_PRIMARY if WINDOW_S_PRIMARY > WINDOW_S_ABLATION else WINDOW_S_ABLATION
    min_purge = min_purge + max(LAG_STACK_S)
    if PURGE_S < min_purge:
        raise ValueError(f"purge {PURGE_S} s is below the {min_purge} s minimum")
    if len(set(RIDGE_ALPHAS)) != len(RIDGE_ALPHAS) or len(RIDGE_ALPHAS) != 7:
        raise ValueError("the Ridge alpha grid is a fixed 7-point set")
    if len(set(PRIMARY_FAMILY)) != len(PRIMARY_FAMILY):
        raise ValueError("primary family members must be unique")
    if EVENT_PRIMARY_OUTCOME in EVENT_SECONDARY_FAMILY:
        raise ValueError("the event primary outcome cannot also be secondary")
    if SUCCESS_CRITERIA["global_null"] not in ("none of the above",):
        raise ValueError("global-null criterion text is fixed")
    if not PRIMARY_FAMILY_SINGLE_ADJUSTMENT or not VERDICTS_FROM_FINAL_ADJUSTMENT:
        raise ValueError("single-adjustment and final-verdict rules are fixed True")
    if not SPAN_LOCAL_PROCESSING or TARGET_EDGE_EXCLUSION_S < 1.0:
        raise ValueError(
            "span-local processing is mandatory and the target edge "
            "exclusion must cover the 0.5 Hz-band transient (≥ 1 s)")
    if set(EVENT_P_VALUE_FAMILY) & set(EVENT_SECONDARY_FAMILY):
        raise ValueError("the event p-value family holds tests, not outcomes")
    if DEFINITIVE_RUN_REQUIREMENTS["detection_null_draws"] != DETECTION_NULL_DRAWS \
            or DEFINITIVE_RUN_REQUIREMENTS["regression_null_draws"] != REGRESSION_NULL_DRAWS:
        raise ValueError("definitive draws must equal the frozen null draws")
    if not DEFINITIVE_RUN_REQUIREMENTS["public_results_doc_only_in_definitive_mode"]:
        raise ValueError("public results documents require definitive mode")

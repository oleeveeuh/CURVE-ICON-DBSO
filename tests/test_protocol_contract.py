"""Contract tests for the frozen follow-on protocol (REST / MOVE / DELTA).

The machine-readable contract lives in ``curve_icon_dbso.protocol``;
``docs/followon_protocol.md`` is its human mirror. This test file is the
freeze enforcement:

* the **pinned hash** fails any silent change to targets, bands, fold
  rules, model grids, QC floors, or stage rules — a deliberate change
  must bump ``PROTOCOL_VERSION``, update this pin, and add a changelog
  entry, so all three move together;
* structural invariants are re-checked independently of the module's own
  import-time validation;
* frozen constants are cross-checked against the implementations they
  govern (``kinematics.py`` tap floors, ``bootstrap.py`` null/bootstrap
  floors, ``evaluation.py`` fold rule) so the contract and the code can
  never drift apart silently;
* ``docs/followon_protocol.md`` must mirror the version, the hash, and
  the frozen quantities.

Everything here is offline and uses synthetic fixtures only (tiny
arrays for splitter checks) — no real data anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_icon_dbso import bootstrap as bootstrap_mod
from curve_icon_dbso import evaluation as evaluation_mod
from curve_icon_dbso import kinematics as kinematics_mod
from curve_icon_dbso.protocol import (
    ARMS,
    FROZEN_DATE,
    HFO_BAND_HZ,
    HIGH_GAMMA_ALTERNATIVE,
    HIGH_GAMMA_ALTERNATIVE_MIN_FS_HZ,
    MAX_GRID_COMBOS,
    MOVE_LAGS_S,
    MOVE_UPDATE_S,
    MOVE_WINDOW_S,
    PROTOCOL_VERSION,
    STANDARD_BANDS_HZ,
    canonical_json,
    grid_size,
    protocol_document,
    protocol_hash,
    resolve_spectral_bands,
    validate_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "followon_protocol.md"

#: The hash of contract v1.0.0, frozen 2026-11-16 before any follow-on
#: real-data result was inspected. Retained as recorded history: the
#: v1.0.0 integration run's provenance carries exactly this hash.
PINNED_V1_HASH = "ed07c397792073a265019b25491687bf27ae5d2836b24b9a8d9f6625d04565d1"

#: The hash of the CURRENT contract (v1.0.1, corrective revision
#: 2026-11-17 — completes implementation coverage of already-frozen
#: criteria; no frozen quantity changed). If this test fails, either
#: something changed the contract silently (restore it) or a deliberate
#: change is in progress (bump PROTOCOL_VERSION, update this pin AND the
#: hash recorded in docs/followon_protocol.md §14, add a changelog entry).
PINNED_CURRENT_HASH = "5e0e2566f9c5df24d8ac751cceaf96d27a15a781dfb70e90b6d2a0f78e1e92a7"


# ---------------------------------------------------------------------------
# Freeze enforcement
# ---------------------------------------------------------------------------


def test_protocol_hash_is_pinned():
    """Any silent change to a frozen quantity fails right here."""
    current = protocol_hash()
    assert current == PINNED_CURRENT_HASH, (
        "The follow-on analysis contract changed without a version bump. "
        f"Expected hash {PINNED_CURRENT_HASH} (v1.0.1, corrective revision "
        f"of the {FROZEN_DATE} freeze), got {current}. Restore the frozen "
        "quantities, or — if the change is deliberate — bump "
        "PROTOCOL_VERSION, update PINNED_CURRENT_HASH here and the hash in "
        "docs/followon_protocol.md §14, and add a changelog entry."
    )


def test_v1_hash_is_recorded_history_not_the_current_pin():
    """The v1.0.0 freeze hash survives as recorded history and must never
    silently equal or replace the current pin."""
    assert PINNED_V1_HASH != PINNED_CURRENT_HASH
    assert protocol_document()["protocol_version"] == "1.0.1"


def test_protocol_version_is_semver_and_frozen_date_recorded():
    assert re.match(r"^\d+\.\d+\.\d+$", PROTOCOL_VERSION)
    assert PROTOCOL_VERSION == "1.0.1"
    assert FROZEN_DATE == "2026-11-16"
    # The corrective revision is recorded in the contract itself.
    revisions = protocol_document()["revisions"]
    assert revisions[-1]["version"] == "1.0.1"
    assert revisions[-1]["kind"] == "corrective"
    assert "2026-11-17" == revisions[-1]["date"]
    assert "no frozen quantity changed" in revisions[-1]["change"].lower()


def test_canonical_json_is_deterministic_and_hash_matches_sha256():
    doc_json = canonical_json()
    assert doc_json == json.dumps(protocol_document(), sort_keys=True,
                                  separators=(",", ":"))
    assert protocol_hash() == hashlib.sha256(doc_json.encode("utf-8")).hexdigest()
    # Recomputing twice changes nothing.
    assert canonical_json() == canonical_json()


def test_validate_protocol_passes():
    # Also runs at import; asserted here so a future refactor that drops
    # the import-time call stays covered.
    validate_protocol()


def test_contract_module_is_dependency_free():
    """The contract must stay importable on any machine (stdlib only)."""
    source = (REPO_ROOT / "src" / "curve_icon_dbso" / "protocol.py").read_text()
    for forbidden in ("import numpy", "import pandas", "import scipy",
                      "import sklearn"):
        assert forbidden not in source, (
            f"protocol.py must not depend on {forbidden.split()[1]}"
        )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_three_arms_with_frozen_output_levels():
    assert ARMS == ("REST", "MOVE", "DELTA")
    arms = protocol_document()["arms"]
    assert arms["REST"]["output_level"] == "subject"
    assert arms["DELTA"]["output_level"] == "subject"
    assert arms["MOVE"]["output_level"] == "window_within_subject_holdout"


def test_primary_target_is_tap_rate_mean_and_frozen():
    doc = protocol_document()
    assert doc["targets"]["primary_subject"]["name"] == "tap_rate_mean"
    assert doc["targets"]["secondary_subject"] == {
        "tap_interval_cv": doc["targets"]["secondary_subject"]["tap_interval_cv"],
        "tap_vigor_index": doc["targets"]["secondary_subject"]["tap_vigor_index"],
        "tap_sequence_slope": doc["targets"]["secondary_subject"]["tap_sequence_slope"],
    }
    assert set(doc["targets"]["within_subject_only"]) == {"tap_amplitude"}
    assert doc["targets"]["move_continuous"]["name"] == "glove_vigor_envelope_z"
    # The primary target's implementation is the already-frozen kinematics
    # module — the contract records it, never redefines it.
    assert "curve_icon_dbso.kinematics" in doc["targets"]["primary_subject"]["definition"]


def test_tap_floors_match_the_frozen_kinematics_implementation():
    """The contract's tap floors MUST equal kinematics.py constants."""
    assert protocol_document()["targets"]["frozen_floors"] == {
        "tap_rate_low_hz": kinematics_mod.TAP_RATE_LOW_HZ,
        "tap_rate_high_hz": kinematics_mod.TAP_RATE_HIGH_HZ,
        "tap_prominence_floor": kinematics_mod.TAP_PROMINENCE_FLOOR,
        "tap_min_channel_blocks": kinematics_mod.TAP_MIN_CHANNEL_BLOCKS,
        "tap_welch_segment_seconds": kinematics_mod.TAP_WELCH_SEGMENT_SECONDS,
        "tap_min_cycles_per_block": 5,
        "tap_detection_band_hz": [0.5, 4.0],
        "tap_min_peak_separation_s": 0.15,
        "tap_detection_filter_order": 4,
    }


# ---------------------------------------------------------------------------
# Bands and acquisition-driven resolution
# ---------------------------------------------------------------------------


def test_standard_bands_match_the_frozen_boundaries():
    assert STANDARD_BANDS_HZ == {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta_low": (13.0, 20.0),
        "beta_high": (20.0, 35.0),
        "gamma_low": (35.0, 60.0),
    }


def test_band_ladder_is_ordered_disjoint_and_within_hard_range():
    ladder = sorted(STANDARD_BANDS_HZ.items(), key=lambda item: item[1][0])
    previous_high = 0.0
    for name, (low, high) in ladder:
        assert 0.0 < low < high <= 500.0, f"band {name} edges out of range"
        assert low >= previous_high, f"band {name} overlaps its predecessor"
        previous_high = high
    for name, (low, high) in (
        ("gamma_high_alt", HIGH_GAMMA_ALTERNATIVE),
        ("hfo", HFO_BAND_HZ),
    ):
        assert 0.0 < low < high <= 500.0, f"{name} range out of range"


@pytest.mark.parametrize("min_fs,expect_gamma_high,expect_hfo_feasible", [
    (2000.0, [60.0, 100.0], True),
    (1020.0, [60.0, 100.0], True),
    (1000.0, [60.0, 100.0], False),   # HFO needs fs >= 1020 (10 Hz margin)
    (500.0, [60.0, 100.0], False),
    (220.0, [60.0, 100.0], False),    # gamma_high needs fs >= 220
    (210.0, None, False),
    (180.0, None, False),             # gamma_low (<= 70 Hz ceiling) still active
])
def test_band_resolution_is_deterministic_in_sampling_rate(
    min_fs, expect_gamma_high, expect_hfo_feasible
):
    resolved = resolve_spectral_bands(min_fs)
    assert resolved["gamma_high"] == expect_gamma_high
    assert resolved["hfo_block_feasible"] is expect_hfo_feasible
    # Monotone: a lower fs can only shrink the active set.
    if min_fs >= 220.0:
        assert set(resolved["bands_active"]) >= {"theta", "alpha", "beta_low"}
    again = resolve_spectral_bands(min_fs)
    assert again == resolved


def test_band_resolution_rejects_invalid_sampling_rate():
    with pytest.raises(ValueError, match="positive finite"):
        resolve_spectral_bands(0.0)
    with pytest.raises(ValueError, match="positive finite"):
        resolve_spectral_bands(float("nan"))


def test_high_gamma_alternative_is_never_auto_selected():
    """The 90–200 Hz alternative must carry the new-version + fs ≥ 420 gate."""
    doc = protocol_document()
    alt = doc["neural_features"]["gamma_high_alternative"]
    assert alt["range"] == [90.0, 200.0]
    assert "never auto-selected" in alt["status"]
    assert str(HIGH_GAMMA_ALTERNATIVE_MIN_FS_HZ) in alt["status"]


# ---------------------------------------------------------------------------
# QC, aggregation, windowing
# ---------------------------------------------------------------------------


def test_qc_rules_forbid_silent_repair():
    qc = protocol_document()["qc_rules"]
    assert "never imputed, zero-filled, or silently dropped" in qc["no_silent_repair"]
    for key in ("non_finite", "flatline", "saturation", "rapid_transients",
                "implausible_band_power", "line_noise", "target_qc"):
        assert key in qc


def test_min_usable_data_gates_are_frozen():
    gates = protocol_document()["min_usable_data"]
    assert gates == {"rest_s": 5.0, "move_s": 30.0}


def test_windowing_is_frozen_causal_and_within_prompt_bounds():
    doc = protocol_document()
    windowing = doc["windowing_move"]
    assert windowing["window_s"] == MOVE_WINDOW_S == 0.5
    assert windowing["update_s"] == MOVE_UPDATE_S == 0.1
    lags = windowing["lags_s"]
    assert lags == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert 0.0 in lags and max(lags) <= 0.5 and min(lags) >= 0.0
    assert "never" in windowing["rule"]
    assert "future" in windowing["lags_semantics"]
    assert "BACK from the target" in windowing["lags_semantics"]
    assert MOVE_LAGS_S == tuple(lags)


def test_aggregation_fallback_is_documented_not_guessed():
    agg = protocol_document()["aggregation"]
    assert "fallback" in agg["rereferencing"]
    assert "UNKNOWN" in agg["localization_status"]
    assert "MEDIAN" in agg["fallback_aggregation"]


# ---------------------------------------------------------------------------
# Models and bounded grids
# ---------------------------------------------------------------------------


def test_subject_level_model_set_is_exactly_the_prespecified_seven():
    models = protocol_document()["models_subject_level"]
    assert set(models) == {
        "dummy_mean", "dummy_median",            # baselines
        "ridge_primary",                          # primary
        "svr_linear", "bayesian_ridge",           # comparators
        "gp_matern32", "xgboost_shallow",
    }
    assert models["ridge_primary"]["kind"] == "primary"
    assert models["gp_matern32"]["grid"] == {}  # fixed, nothing tuned
    assert "nu=1.5" in models["gp_matern32"]["spec"]
    assert models["xgboost_shallow"]["kind"] == "comparator_optional"


def test_move_model_set_is_the_prespecified_three():
    models = protocol_document()["models_move"]
    assert set(models) == {
        "lagged_ridge_primary", "svr_linear_move", "xgboost_shallow_move",
    }
    assert models["lagged_ridge_primary"]["kind"] == "primary"


def test_all_grids_are_bounded_and_nonempty_grids_are_finite():
    doc = protocol_document()
    for family in ("models_subject_level", "models_move"):
        for name, spec in doc[family].items():
            size = grid_size(spec)
            assert 1 <= size <= MAX_GRID_COMBOS, (
                f"{family}.{name} grid size {size} violates the frozen "
                f"ceiling of {MAX_GRID_COMBOS}"
            )


def test_ridge_grid_is_the_frozen_seven_point_log_grid():
    grid = protocol_document()["models_subject_level"]["ridge_primary"]["grid"]
    assert grid["alpha"] == [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    move_grid = protocol_document()["models_move"]["lagged_ridge_primary"]["grid"]
    assert move_grid == grid  # identical frozen grids


def test_svr_epsilon_is_training_fold_relative_never_test_relative():
    spec = protocol_document()["models_subject_level"]["svr_linear"]
    assert "epsilon_factor_of_train_MAD" in spec["grid"]
    assert set(spec["grid"]["epsilon_factor_of_train_MAD"]) == {0.1, 0.3}


# ---------------------------------------------------------------------------
# Validation design: the outer unit is always subject
# ---------------------------------------------------------------------------


def test_outer_rule_matches_the_implemented_splitter():
    """The contract's fold rule must equal the implemented splitter —
    LOSO at or below 12 subjects, 5-fold GroupKFold above it."""
    assert protocol_document()["validation"]["outer_unit"] == "subject"
    assert evaluation_mod.LOSO_MAX_GROUPS == 12
    for n_subjects in (2, 5, 12):
        groups = np.array([f"s{i}" for i in range(n_subjects)])
        x = pd.DataFrame({"f": np.arange(float(n_subjects))})
        splits = list(evaluation_mod._default_splitter(n_subjects).split(
            x, np.arange(float(n_subjects)), groups
        ))
        assert len(splits) == n_subjects  # LOSO: one fold per subject
    for n_subjects in (13, 31):
        groups = np.array([f"s{i}" for i in range(n_subjects)])
        x = pd.DataFrame({"f": np.arange(float(n_subjects))})
        splits = list(evaluation_mod._default_splitter(n_subjects).split(
            x, np.arange(float(n_subjects)), groups
        ))
        assert len(splits) == 5  # GroupKFold(5)
        seen: set[str] = set()
        for train_idx, test_idx in splits:
            test_subjects = set(groups[test_idx])
            assert test_subjects.isdisjoint(seen), "a subject spans two folds"
            assert not test_subjects & set(groups[train_idx])
            seen |= test_subjects
    assert protocol_document()["validation"]["loso_max_subjects"] == 12
    assert protocol_document()["validation"]["outer_groupkfold_splits"] == 5


def test_inner_validation_is_subject_grouped_with_frozen_fallback():
    doc = protocol_document()["validation"]
    assert doc["inner_splits"] == 3
    assert doc["inner_loso_min_outer_train_subjects"] == 6
    assert doc["inner_selection_criterion"].startswith("inner_cv_mean_MAE")
    assert "never" in doc["window_fold_rule"] or "exactly one" in doc["window_fold_rule"]


# ---------------------------------------------------------------------------
# Metrics, baselines, promotion, uncertainty, nulls
# ---------------------------------------------------------------------------


def test_promotion_rule_requires_both_error_metrics_and_the_frozen_reference():
    doc = protocol_document()
    rule = doc["promotion_rule"]["subject_level"]
    assert "MAE and RMSE" in rule
    assert "frozen five-feature reference" in rule
    assert "outer-test performance" in doc["promotion_rule"]["no_outer_selection"]
    assert doc["frozen_five_feature_reference"] == "frozen_first_run_five_feature_result"
    assert "Holm-adjusted" in rule  # multiplicity control travels with the rule
    assert "within one arm" in doc["holm_family_rule"]


def test_move_baselines_include_the_three_naive_references():
    baselines = protocol_document()["baselines_move"]
    assert set(baselines) == {"fold_global_mean", "block_mean_naive", "persistence"}


def test_move_metrics_summarize_across_subjects_not_pooled_windows():
    metrics = protocol_document()["metrics_move"]
    assert "median and IQR across held-out subjects" in metrics["summary"]
    assert "non-independence" in metrics["pooled_reporting"]


def test_uncertainty_floors_match_the_corrected_bootstrap_implementation():
    """The contract must carry the CORRECTED null discipline (null scored
    against permuted targets; >= 1,000 permutations when reporting p)."""
    doc = protocol_document()["uncertainty"]
    assert doc["bootstrap_n_resamples"] == bootstrap_mod.BOOTSTRAP_N_RESAMPLES == 2000
    assert doc["bootstrap_seed"] == bootstrap_mod.BOOTSTRAP_SEED == 42
    assert doc["ci_level"] == bootstrap_mod.CI_LEVEL == 0.95
    assert doc["permutation_min_for_reporting"] == (
        bootstrap_mod.PERMUTATION_N_PERMUTATIONS
    ) == 1000
    assert doc["permutation_floor"] == 200
    assert doc["permutation_seed"] == 42
    assert doc["move_null_shifts"] >= 1000
    assert doc["move_circular_shift_min_s"] == 2.0


def test_null_procedures_use_the_corrected_convention_and_preserve_autocorrelation():
    nulls = protocol_document()["null_procedures"]
    assert "own permuted targets" in nulls["subject_level"]
    assert "never the original targets" in nulls["subject_level"]
    assert "circular shifts" in nulls["move_autocorrelation_preserving"]
    assert "NEVER" in nulls["move_autocorrelation_preserving"]
    assert "subjects, never rows" in nulls["subject_level"]


# ---------------------------------------------------------------------------
# Ablations, transfer criteria, stage rules
# ---------------------------------------------------------------------------


def test_subject_level_ablation_blocks_are_the_prespecified_seven():
    """Seven blocks since the v1.0.1 corrective revision: the frozen §11
    'DELTA analogously' combined block is implemented, not just implied."""
    blocks = protocol_document()["ablation_blocks_subject_level"]
    assert set(blocks) == {
        "clinical", "rest_spectral", "rest_dynamics",
        "delta_spectral", "delta_dynamics", "combined", "delta_combined",
    }
    assert blocks["clinical"]["features"] == ["age", "dbs_volt", "dbs_freq"]
    assert blocks["combined"]["features"] == [
        "rest_spectral + rest_dynamics + clinical"
    ]
    assert blocks["delta_combined"]["features"] == [
        "delta_spectral + delta_dynamics + clinical"
    ]


def test_move_ablation_includes_the_dmd_representation_comparison():
    move_blocks = protocol_document()["ablation_blocks_move"]
    assert any("multiband_power" in block for block in move_blocks)
    assert any("dmd_spatial_nodes" in block for block in move_blocks)
    assert any("secondary" in block for block in move_blocks)


def test_transfer_criteria_cover_all_four_interpretation_outcomes():
    criteria = protocol_document()["transfer_criteria"]
    assert set(criteria) == {
        "movement_state_transfer", "resting_trait_evidence",
        "state_change_evidence", "global_null", "no_claims",
    }
    assert "Pearson r > 0" in criteria["movement_state_transfer"]
    assert "MAE and RMSE" in criteria["state_change_evidence"]


def test_stage_rules_keep_the_dabi_cohort_exploratory():
    """Honest stage discipline: the cohort was already inspected once, so
    follow-on work on it is exploratory no matter how well frozen."""
    rules = protocol_document()["stage_rules"]
    assert rules["dabi_cohort_stage"] == "exploratory"
    assert "external, independent cohort" in rules["confirmatory_requires"]
    assert rules["post_hoc_label"] == "post_hoc"
    assert "never" in rules["post_hoc_definition"]
    assert "protocol_version" in rules["version_change_rule"]
    assert "protocol_hash" in rules["version_change_rule"]


# ---------------------------------------------------------------------------
# Privacy and the human mirror document
# ---------------------------------------------------------------------------


def test_document_contains_no_identifiers_or_credential_patterns():
    doc_json = canonical_json()
    assert not re.search(r"DBS_bG\d", doc_json)
    assert "AKIA" not in doc_json
    assert "PRIVATE KEY" not in doc_json


def test_provenance_label_and_privacy_rules_are_frozen():
    provenance = protocol_document()["provenance"]
    assert "NOT DBS outcome" in provenance["label"]
    assert "gitignored" in provenance["privacy"]["per_subject_outputs"]
    assert "synthetic fixtures only" in provenance["privacy"]["synthetic_tests"]


def test_unresolved_data_layout_issues_are_recorded():
    issues = " ".join(protocol_document()["unresolved_data_layout"])
    for topic in ("sampling rates", "anatomical-region", "adjacency",
                  "calibration"):
        assert topic in issues, f"unresolved-layout list is missing: {topic}"


def test_followon_protocol_doc_mirrors_the_contract():
    """The human document must carry the version, the pinned hash, the
    targets, and every band key — so doc and module move together."""
    text = DOC_PATH.read_text()
    assert DOC_PATH.exists()
    assert PROTOCOL_VERSION in text
    assert PINNED_CURRENT_HASH in text
    assert PINNED_V1_HASH in text  # v1.0.0 stays as recorded history
    for target in ("tap_rate_mean", "tap_interval_cv", "tap_vigor_index",
                   "tap_sequence_slope", "tap_amplitude",
                   "glove_vigor_envelope_z"):
        assert target in text, f"docs/followon_protocol.md is missing target {target}"
    for band_key in ("theta", "alpha", "beta_low", "beta_high",
                     "gamma_low", "gamma_high"):
        assert f"`{band_key}`" in text, (
            f"docs/followon_protocol.md is missing band key {band_key}"
        )
    for edge in ("4–8", "8–12", "13–20", "20–35", "35–60", "60–100"):
        assert edge in text, f"docs/followon_protocol.md is missing band edge {edge}"
    for arm in ("REST", "MOVE", "DELTA"):
        assert arm in text
    assert "exploratory" in text and "post-hoc" in text


def test_doc_and_module_agree_on_the_recorded_freeze_hash():
    """The hash recorded in the doc's changelog must be the module's
    current hash (they may only move together via a version bump)."""
    text = DOC_PATH.read_text()
    hashes = set(re.findall(r"`([0-9a-f]{64})`", text))
    assert protocol_hash() in hashes

"""Aggregate summary + report generation (v2 contract §15).

Everything committed is cohort-level aggregate; per-subject tables stay
in the gitignored run directory. The results document is written by the
authorized driver run — never generated speculatively.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import contract
from .pipeline import StudyArtifacts


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


#: Key fragments that mark LOCAL-ONLY (per-subject / ordinal-position)
#: data anywhere in an artifact tree. The public summary is scrubbed
#: recursively so a wholesale copy of a richer internal dict can never
#: carry subject-level values into committable output (post-definitive-
#: run review, 2026-11-20: keyed per-subject delta-MAE maps reached the
#: public summary through ``artifacts.ablations``).
LOCAL_ONLY_KEY_FRAGMENTS = ("per_subject", "held_out", "ordinal")
#: Exact names EXEMPT from the fragment guard: frozen contract vocabulary
#: whose payloads are cohort aggregates, never subject-level values.
#: (``per_subject_spread`` is a member of SECONDARY_FAMILY_REGRESSION —
#: the IQR of per-subject performance, reported as one aggregate number.)
LOCAL_ONLY_KEY_EXACT_ALLOWLIST = ("per_subject_spread",)


def _is_local_only_key(key: str) -> bool:
    lowered = str(key).lower()
    if lowered in LOCAL_ONLY_KEY_EXACT_ALLOWLIST:
        return False
    return any(fragment in lowered for fragment in LOCAL_ONLY_KEY_FRAGMENTS)


def _contains_local_only_key(value, _prefix=()) -> list[str]:
    """Return the key paths of any local-only fields left in ``value``
    (the privacy test's oracle)."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            path = f"{_prefix}.{key}" if _prefix else str(key)
            if _is_local_only_key(key):
                found.append(path)
            found.extend(_contains_local_only_key(sub, path))
    elif isinstance(value, (list, tuple)):
        for index, sub in enumerate(value):
            found.extend(_contains_local_only_key(sub, f"{_prefix}[{index}]"))
    return found


def strip_local_only(value):
    """Recursively REMOVE every dict entry whose key carries a
    local-only fragment (``per_subject*``, ``held_out*``, ``*ordinal*``),
    except the frozen contract statistic names in the allowlist (their
    payloads are still scanned recursively). Aggregate fields pass
    through untouched; this is defense in depth — builders already keep
    per-subject rows out of the public summary."""
    if isinstance(value, dict):
        return {
            key: strip_local_only(sub)
            for key, sub in value.items()
            if not _is_local_only_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [strip_local_only(sub) for sub in value]
    return value


def build_run_summary(artifacts: StudyArtifacts) -> dict:
    """Aggregate machine-readable summary (contract §15 requirements)."""
    detection = artifacts.track_a["results"]
    summary = {
        "contract_version": contract.CONTRACT_VERSION,
        "contract_hash": contract.contract_hash(),
        "stage": contract.STAGE,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_flow": artifacts.cohort_flow,
        "track_a_detection": {
            name: {
                "median_bac": result["median_bac"],
                "iqr_bac": result["iqr_bac"],
                "bootstrap_median_bac_ci": result["bootstrap"]["median_bac_ci"],
                "null": {k: v for k, v in result["null"].items()
                         if k != "null"},
                "verdict": result["verdict"],
            } for name, result in detection.items()
        },
        "track_a_secondary_family": artifacts.track_a.get(
            "secondary_family", {}),
        "track_bc_zero_shot": {
            k: v for k, v in artifacts.tracks_bc.zero_shot.items()
            if not k.startswith("per_subject")
        },
        "track_c_detection_conditions": {
            "median_bac_zero_shot_full":
                artifacts.detection_conditions.get("median_bac_zero_shot_full"),
            "median_bac_zero_shot_test_interval":
                artifacts.detection_conditions.get(
                    "median_bac_zero_shot_test_interval"),
            "calibration_curve": artifacts.detection_conditions.get(
                "calibration_curve"),
        },
        "track_bc_fewshot": {
            key: {k: v for k, v in aggregate.items()}
            for key, aggregate in artifacts.tracks_bc.fewshot.items()
        },
        "calibration_curve": artifacts.tracks_bc.calibration_curve,
        "track_d_events": {
            "n_recordings_with_events": artifacts.track_d.get(
                "n_recordings_with_events"),
            "primary_association": artifacts.track_d.get("primary_association"),
            "primary_association_null": artifacts.track_d.get(
                "primary_association_null"),
            "secondary_tap_associations": artifacts.track_d.get(
                "secondary_tap_associations", {}),
            "secondary_family_per_recording_median":
                artifacts.track_d.get("secondary_family_per_recording_median"),
            "secondary_descriptive_note": artifacts.track_d.get(
                "secondary_descriptive_note"),
            "event_family": artifacts.track_d.get("event_family", {}),
            "event_prediction": artifacts.track_d.get("event_prediction"),
        },
        "ablations": artifacts.ablations,
        "primary_family": artifacts.primary_family,
        "verdicts": artifacts.verdicts,
        "reporting_distinctions": {
            "detection_vs_regression": (
                "Track A classifies REST vs MOVE; Tracks B/C decode a "
                "continuous standardized movement envelope — never mixed"),
            "zero_shot_vs_few_shot": (
                "zero-shot uses no held-out-subject labeled data; few-shot "
                "uses only the chronological calibration segment; separate "
                "claims, separate endpoints"),
            "neural_vs_kinematic_vs_combined": (
                "every regression claim compares the neural model against "
                "the kinematic AR baseline and the combined forecast"),
            "association_vs_prediction": (
                "Track D association (subject-replicated Spearman) and "
                "prediction (held-out error) are reported separately"),
            "exploratory_vs_confirmatory": (
                "this cohort is exploratory by construction; confirmatory "
                "status is reserved for an external cohort under an "
                "unchanged contract"),
            "statistical_vs_practical": (
                "small above-null effects may not be practically useful; "
                "every verdict carries its effect size and interval"),
        },
    }
    return _clean(strip_local_only(summary))


def write_summary(summary: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def write_cohort_flow(artifacts: StudyArtifacts, output_dir: Path) -> Path:
    """The LOCAL-ONLY cohort-flow artifact (gitignored run directory):
    carries the ordinal-position lists; the committable summary carries
    counts only (v2.1.0 privacy rule)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cohort_flow.json"
    path.write_text(json.dumps(_clean(artifacts.cohort_flow_local_only
                                      or artifacts.cohort_flow),
                              indent=2, sort_keys=True) + "\n")
    return path


def write_calibration_curve(artifacts: StudyArtifacts, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in artifacts.tracks_bc.calibration_curve:
        gain = entry.get("gain_over_zero_shot_r", {})
        rows.append({
            "calibration_fraction": entry["fraction"],
            "n_subjects": entry["n_subjects"],
            "median_usable_calibration_minutes":
                entry["median_usable_calibration_minutes"],
            "median_r_combined_test_interval":
                entry["median_r_combined_test_interval"],
            "median_r_zero_shot_test_interval":
                entry["median_r_zero_shot_test_interval"],
            "median_gain_r_over_zero_shot": gain.get("median_gain"),
            "gain_bootstrap_ci": gain.get("bootstrap_ci"),
        })
    path = output_dir / "calibration_curve.json"
    path.write_text(json.dumps(_clean(rows), indent=2) + "\n")
    return path


def _fmt_num(value) -> str:
    """Table-cell formatting that tolerates None/NaN aggregates."""
    if value is None:
        return "—"
    try:
        if not np.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.3f}" if abs(float(value)) < 1000 else str(value)


def render_results_doc(summary: dict, run_dir_label: str) -> str:
    """The committed results document (aggregate only)."""
    verdicts = summary["verdicts"]
    detection = summary["track_a_detection"]["ridge_logistic"]
    zero = summary["track_bc_zero_shot"]
    fewshot30 = summary["track_bc_fewshot"].get(
        f"fewshot_{int(contract.FEWSHOT_PRIMARY_FRACTION*100)}", {})
    lines: list[str] = []
    lines.append("# Adaptive-transfer v2 results (authorized real-data run)")
    lines.append("")
    lines.append(f"**Stage:** exploratory (contract {contract.CONTRACT_VERSION}, "
                 f"hash `{contract.contract_hash()}`). Run directory: "
                 f"`{run_dir_label}` (gitignored; per-subject values never "
                 "committed).")
    lines.append("")
    lines.append("## Cohort flow")
    lines.append("")
    flow = summary["cohort_flow"]
    lines.append(f"- recordings loaded: {flow['n_recordings_loaded']}; "
                 f"eligible: {flow['n_eligible']}")
    for reason, count in flow.get("n_excluded_by_reason", {}).items():
        lines.append(f"- excluded ({reason}): {count} recording(s)")
    lines.append(f"- signal-quality-flagged retained: "
                 f"{flow['n_signal_quality_flagged_retained']}")
    lines.append(f"- Track D ineligible: {flow['n_track_d_ineligible']}")
    lines.append("")
    lines.append("## Frozen verdicts (explicit, per criterion)")
    lines.append("")
    lines.append("Every verdict = criterion + bootstrap CI + FINAL "
                 "Holm-adjusted p (primary family for the four primary "
                 "endpoints; one event family for the event claims).")
    lines.append("")

    def _effect_and_interval(key: str, v: dict) -> tuple[str, str]:
        detail = v.get("detail", {}) or {}
        effect, interval = {
            "detection_transfer": ("median_bac", "ci"),
            "regression_zero_shot": ("median_r", "bootstrap_ci"),
            "fewshot_delta_mae": ("median_delta_mae", "bootstrap_ci"),
            "fewshot_gain": ("median_gain_r", "bootstrap_ci"),
            "event_association": ("observed_fisher_z_r",
                                  "bootstrap_ci_fisher_z_r"),
            "event_prediction": ("observed_median_delta_mae",
                                 "bootstrap_ci_median_delta_mae"),
        }[key]
        effect_value = v.get(effect, detail.get(effect))
        interval_value = v.get(interval, detail.get(interval))
        return _fmt_num(effect_value), str(interval_value)

    lines.append("| Criterion | Verdict | Effect | Interval | Holm p |")
    lines.append("|---|---|---|---|---|")
    for key, label in (
        ("detection_transfer", "Track A/C-zero detection"),
        ("regression_zero_shot", "Track C zero-shot regression (median r)"),
        ("fewshot_delta_mae", "Track B primary (delta-MAE, 30%)"),
        ("fewshot_gain", "Track C few-shot gain (30%)"),
        ("event_association", "Track D association"),
        ("event_prediction", "Track D prediction"),
    ):
        v = verdicts.get(key, {})
        effect, interval = _effect_and_interval(key, v)
        lines.append(
            f"| {label} | {'SUCCESS' if v.get('verdict') else 'failure'} | "
            f"{effect} | {interval} | {v.get('holm_p', '')} |")
    lines.append("")
    lines.append(f"**Global null: {'YES' if verdicts.get('global_null') else 'NO'}.**")
    lines.append("")
    lines.append("## Detection (Track A; zero-shot by construction)")
    lines.append("")
    baseline_exceeded = bool(
        summary["track_a_detection"]["ridge_logistic"]["verdict"].get(
            "above_majority_baseline"))
    lines.append(f"- Ridge logistic: median subject BAC "
                 f"{detection['median_bac']:.3f} "
                 f"(IQR {detection['iqr_bac'][0]:.3f}–{detection['iqr_bac'][1]:.3f}); "
                 f"bootstrap CI {detection['bootstrap_median_bac_ci'][0]:.3f}–"
                 f"{detection['bootstrap_median_bac_ci'][1]:.3f}; "
                 f"majority-class baseline: BAC 0.500; "
                 f"{'exceeded' if baseline_exceeded else 'not exceeded'}")
    xgb = summary["track_a_detection"].get("xgboost_fixed")
    if xgb is not None:
        lines.append(f"- comparator XGBoost (fixed): median BAC "
                     f"{xgb['median_bac']:.3f} (sensitivity "
                     f"{_fmt_num(xgb['null'].get('secondary_family', {}).get('detail', {}).get('sensitivity', {}).get('observed_median'))}, "
                     f"specificity "
                     f"{_fmt_num(xgb['null'].get('secondary_family', {}).get('detail', {}).get('specificity', {}).get('observed_median'))})")
    else:
        lines.append("- comparator XGBoost (fixed): not executed "
                     "(optional dependency absent)")
    null = detection["null"]
    lines.append(f"- block-permutation null: p_raw = {null.get('p_raw')}, "
                 f"Holm (model family, descriptive) = {null.get('p_holm')} "
                 f"({contract.DETECTION_NULL_DRAWS:,} draws, "
                 f"seed {contract.NULL_SEED})")
    lines.append("")
    lines.append("## Regression (Tracks B/C)")
    lines.append("")
    lines.append(f"- zero-shot median subject r (neural-only): "
                 f"{zero.get('median_r_neural_only')} "
                 f"(persistence {zero.get('median_r_persistence')}); "
                 f"circular-shift null p_raw = "
                 f"{(zero.get('null_median_r_neural_only') or {}).get('p_raw')}")
    lines.append(f"- few-shot 30%: median delta-MAE "
                 f"({fewshot30.get('median_mae_delta')}) = MAE(combined) − "
                 f"MAE(AR); circular-shift null p_raw = "
                 f"{(fewshot30.get('null_delta_mae_circular') or {}).get('p_raw')}; "
                 f"blockwise p_raw = "
                 f"{(fewshot30.get('null_delta_mae_blockwise') or {}).get('p_raw')}")
    lines.append("")
    lines.append("### Calibration curve")
    lines.append("")
    lines.append("| Track | Fraction | Subjects | Calibration minutes (median) | "
                 "few-shot performance | zero-shot (same interval) | gain CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in summary["calibration_curve"]:
        gain = entry.get("gain_over_zero_shot_r") or {}
        lines.append(
            f"| regression | {entry.get('fraction')} | {entry.get('n_subjects')} | "
            f"{_fmt_num(entry.get('median_usable_calibration_minutes'))} | "
            f"{_fmt_num(entry.get('median_r_combined_test_interval'))} | "
            f"{_fmt_num(entry.get('median_r_zero_shot_test_interval'))} | "
            f"{gain.get('bootstrap_ci')} |")
    for entry in (summary.get("track_c_detection_conditions") or {}).get(
            "calibration_curve", []):
        gain = entry.get("gain_over_zero_shot_bac") or {}
        lines.append(
            f"| detection | {entry.get('fraction')} | {entry.get('n_subjects')} | "
            f"{_fmt_num(entry.get('median_usable_calibration_minutes'))} | "
            f"{_fmt_num(entry.get('median_bac_fewshot'))} | "
            f"{_fmt_num(entry.get('median_bac_zero_shot_test_interval'))} | "
            f"{gain.get('bootstrap_ci')} |")
    lines.append("")
    lines.append("## Track D (event-level)")
    lines.append("")
    assoc = summary["track_d_events"]["primary_association"]
    lines.append(f"- association ({assoc.get('outcome')} vs "
                 f"{assoc.get('neural_feature')}): Fisher-z r "
                 f"{assoc.get('observed_fisher_z_r')}, CI "
                 f"{assoc.get('bootstrap_ci_fisher_z_r')} "
                 f"({assoc.get('n_subjects_with_events')} subjects; association "
                 "claim, not prediction)")
    for outcome_name, entry in (summary["track_d_events"].get(
            "secondary_tap_associations", {}) or {}).items():
        detail = entry.get("association", {})
        lines.append(f"- secondary association ({outcome_name} vs "
                     f"{detail.get('neural_feature')}): Fisher-z r "
                     f"{detail.get('observed_fisher_z_r')}, CI "
                     f"{detail.get('bootstrap_ci_fisher_z_r')}")
    event_family = summary["track_d_events"].get("event_family", {})
    if event_family:
        lines.append(f"- event p-value family (Holm within family): "
                     f"{json.dumps(_clean(event_family.get('p_holm_within_family', {})), sort_keys=True)}")
    pred = summary["track_d_events"]["event_prediction"]
    lines.append(f"- prediction: median delta-MAE "
                 f"{pred.get('observed_median_delta_mae')}, bootstrap CI "
                 f"{pred.get('bootstrap_ci_median_delta_mae')}, null p_raw "
                 f"{(pred.get('null') or {}).get('p_raw')}")
    lines.append("")
    lines.append("## Ablations (prespecified; all reported, none selected)")
    lines.append("")
    lines.append("Paired per-subject contrasts against the corresponding "
                 "primary condition (two-sided sign-flip null, Holm "
                 "within the contrast family); aggregates only.")
    lines.append("")
    contrast = (summary.get("ablations") or {}).get("contrast_family", {})
    if contrast:
        members = contrast.get("members", {})
        holm = contrast.get("p_holm_within_family", {})
        lines.append("| Ablation contrast | n | Median contrast | "
                     "Direction | Raw p | Holm p |")
        lines.append("|---|---|---|---|---|---|")
        for name, member in members.items():
            direction = ("positive favors variant"
                         if "positive" in member.get("direction_note", "")
                         else "negative favors variant")
            lines.append(
                f"| {name} | {member.get('n_pairs', '—')} | "
                f"{_fmt_num(member.get('median_contrast'))} | {direction} | "
                f"{_fmt_num(member.get('sign_flip_p_raw_two_sided'))} | "
                f"{_fmt_num(holm.get(name))} |")
        lines.append("")
        lines.append("Condition medians (aggregate): "
                     + "; ".join(
                         f"{name} median BAC "
                         f"{_fmt_num(detail.get('ridge_logistic', {}).get('median_bac'))}"
                         for name, detail in (summary.get("ablations") or {}).items()
                         if isinstance(detail, dict)
                         and "ridge_logistic" in detail)
                     + "; "
                     + "; ".join(
                         f"{name} median ΔMAE "
                         f"{_fmt_num(detail.get('median_delta_mae'))}"
                         for name, detail in (summary.get("ablations") or {}).items()
                         if isinstance(detail, dict)
                         and "median_delta_mae" in detail))
        lines.append("")
    lines.append("")
    lines.append("## Distinctions preserved")
    lines.append("")
    for key, text in summary["reporting_distinctions"].items():
        lines.append(f"- **{key.replace('_', ' ')}:** {text}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- exploratory single-center cohort; every p-value is "
                 "descriptive; percentile bootstrap intervals undercover at "
                 "this n")
    lines.append("- glove physical units are UNKNOWN across subjects; "
                 "magnitude claims are within-person only")
    lines.append("- spatial mapping is grid-relative anchored (no MNI "
                 "coordinates exist); second-strip contacts are EXCLUDED "
                 "from the primary grid (their anchor assignment is "
                 "undocumented) and are used in no analysis")
    lines.append("- zero-shot regression claims are correlation-only by the "
                 "frozen scale rule (target standardization may never be "
                 "fitted on a final test interval)")
    lines.append("- OFFLINE RETROSPECTIVE analysis (owner decision "
                 "2026-11-19): calibration and test spans are processed "
                 "independently end to end (no sample crosses a span "
                 "boundary), and zero-phase filtering is used WITHIN a "
                 "span; this contract makes no online or causal-inference "
                 "claim")
    return "\n".join(lines) + "\n"

"""DD-006 promotion rule — recomputed from the manifest alone (ADR-006).

Locked decision (owner, 2026-11-14): the synthetic artifact's promotion
decision must be machine-testable and RE-DERIVABLE from the metrics the
manifest itself records — lowest MAE (the prespecified primary metric)
among the fixed candidates, ties broken by candidate order, and
``beats_baseline`` true only when the selection beats BOTH naive baselines
on BOTH MAE and RMSE.

This test retrains the deterministic synthetic artifact and re-derives the
rule INDEPENDENTLY of the training/promotion code path (no
``reporting.beats_baseline_strict`` import): the point is that a future
edit to the promotion logic cannot silently diverge from the locked rule,
because the recomputed verdict must still match the manifest.
"""

from __future__ import annotations

import json

import pytest

from curve_icon_dbso.training import train_synthetic_artifact

PRIMARY_METRIC = "MAE"
STRICT_METRICS = ("MAE", "RMSE")


@pytest.fixture(scope="module")
def manifest(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("promotion-artifact")
    out = base / "artifact"  # must NOT pre-exist (artifacts are immutable by design)
    train_synthetic_artifact(out)
    return json.loads((out / "manifest.json").read_text())


def _recompute_selection(candidates: dict) -> str:
    """Lowest primary metric among candidates; ties -> first in order."""
    names = list(candidates.keys())  # JSON object order == fixed candidate order
    return min(names, key=lambda name: (candidates[name][PRIMARY_METRIC], names.index(name)))


def _recompute_beats_baseline(selected: str, candidates: dict, baselines: dict) -> bool:
    """Strict rule: beat BOTH naive baselines on BOTH MAE and RMSE."""
    if set(baselines) != {"Mean baseline", "Median baseline"}:
        return False  # missing baselines -> not claimable -> False
    for metric in STRICT_METRICS:
        for baseline_metrics in baselines.values():
            if not candidates[selected][metric] < baseline_metrics[metric]:
                return False
    return True


def test_manifest_records_everything_the_rule_needs(manifest):
    for key in ("candidates", "baselines", "beats_baseline", "promotion_label"):
        assert key in manifest, f"manifest is missing {key!r} — the rule is not recomputable"
    assert len(manifest["candidates"]) >= 1
    assert set(manifest["baselines"]) == {"Mean baseline", "Median baseline"}


def test_selected_model_is_the_recomputed_argmin(manifest):
    candidates = manifest["candidates"]
    # "metrics" carries ONLY the selection; its single key names it.
    recorded_selection = next(iter(manifest["metrics"]))
    recomputed = _recompute_selection(candidates)
    assert recorded_selection == recomputed, (
        f"manifest selected {recorded_selection!r} but the locked DD-006 rule "
        f"(lowest {PRIMARY_METRIC}, candidate-order tie-break) gives {recomputed!r}"
    )


def test_beats_baseline_and_promotion_label_match_the_recomputed_rule(manifest):
    selected = next(iter(manifest["metrics"]))
    recomputed_beats = _recompute_beats_baseline(
        selected, manifest["candidates"], manifest["baselines"]
    )
    assert manifest["beats_baseline"] is recomputed_beats, (
        f"manifest beats_baseline={manifest['beats_baseline']} but the strict "
        f"rule recomputes to {recomputed_beats}"
    )
    expected_label = "performance" if recomputed_beats else "demonstration_only"
    assert manifest["promotion_label"] == expected_label


def test_synthetic_verdict_is_stable_across_retraining(manifest, tmp_path):
    """Byte-deterministic training implies a stable promotion verdict; assert
    a second training produces an identical decision (artifacts are
    immutable, so the second training goes into a FRESH directory)."""
    out = tmp_path / "second-training"
    train_synthetic_artifact(out)
    second = json.loads((out / "manifest.json").read_text())
    for key in ("candidates", "baselines", "beats_baseline", "promotion_label"):
        assert second[key] == manifest[key], (
            f"promotion-relevant field {key!r} changed between trainings"
        )
    second = json.loads((out / "manifest.json").read_text())
    for key in ("candidates", "baselines", "beats_baseline", "promotion_label"):
        assert second[key] == manifest[key], f"promotion-relevant field {key!r} changed between trainings"

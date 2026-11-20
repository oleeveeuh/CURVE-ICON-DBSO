"""Contract tests for the frozen v2 adaptive-transfer protocol.

The machine-readable contract lives in
``curve_icon_dbso.transfer.contract``; ``docs/adaptive_transfer_protocol.md``
is its human mirror. This file is the freeze enforcement: the pinned
hash fails any silent change to targets, bands, grids, fold rules, QC
floors, nulls, or criteria; the structural invariants are re-checked
independently; the document must mirror the version and hash; and the
v1.0.1 contract must remain untouched (v2 is forbidden from modifying
it). Offline; synthetic only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from curve_icon_dbso import protocol
from curve_icon_dbso.transfer import contract

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "adaptive_transfer_protocol.md"

#: Pinned at the 2.1.0 freeze (2026-11-19, pre-execution integrity
#: refinement). Any contract change must bump the version, update this
#: pin, and add a changelog row.
PINNED_HASH = "027503a3d0f8359f89d56b16cb47e05adf73df670924216820f57b7fa433bb20"


def test_contract_hash_pinned() -> None:
    assert contract.contract_hash() == PINNED_HASH
    digest = hashlib.sha256(contract.canonical_json().encode("utf-8")).hexdigest()
    assert digest == PINNED_HASH


def test_validate_contract_passes() -> None:
    contract.validate_contract()


def test_stage_is_exploratory_and_separate_from_v1() -> None:
    assert contract.STAGE == "exploratory"
    assert contract.V1_CONTRACT_VERSION == "1.0.1"
    # The v1 contract module is untouched by v2 and still validates.
    assert protocol.PROTOCOL_VERSION == "1.0.1"
    protocol.validate_protocol()


def test_document_mirrors_version_and_hash() -> None:
    text = DOC.read_text()
    assert f"contract_version {contract.CONTRACT_VERSION}" in text
    assert PINNED_HASH in text
    # changelog has a row for the current version
    rows = re.findall(r"^\| (\d+\.\d+\.\d+) \|", text, re.M)
    assert contract.CONTRACT_VERSION in rows


def test_frozen_bands_are_disjoint_and_cohort_valid() -> None:
    ordered = sorted(contract.BANDS_HZ.items(), key=lambda kv: kv[1][0])
    previous_high = 0.0
    for name, (low, high) in ordered:
        assert 0.0 < low < high <= contract.BAND_TOP_HZ_MAX
        assert low >= previous_high, f"band {name} overlaps its predecessor"
        previous_high = high
    # cohort fs 1200 Hz: every band top edge clears Nyquist with margin
    for _name, (_low, high) in ordered:
        assert high <= 1200.0 / 2 - 10.0


def test_calibration_fractions_nested_and_floors_owner_approved() -> None:
    assert contract.CALIBRATION_FRACTIONS == (0.10, 0.20, 0.30)
    assert contract.MIN_CALIBRATION_S == 8.0
    assert contract.MIN_TEST_S == 55.0
    assert contract.FEWSHOT_PRIMARY_FRACTION == 0.30
    # the purge clears the minimum window + lag bar
    assert contract.PURGE_S >= contract.WINDOW_S_ABLATION + max(contract.LAG_STACK_S)


def test_primary_family_complete_and_event_separation() -> None:
    assert len(contract.PRIMARY_FAMILY) == 4
    assert contract.EVENT_PRIMARY_OUTCOME not in contract.EVENT_SECONDARY_FAMILY
    assert contract.EVENT_SUBJECT_UNIT_RULE == "subject_is_replication_unit"


def test_prohibited_post_hoc_list_present() -> None:
    text = json.dumps(contract.contract_document())
    assert "selecting the calibration fraction using final-test performance" in text
    assert "never_commit" in json.dumps(contract.PRIVACY_RULES)


def test_contract_hash_changes_with_content() -> None:
    doc = contract.contract_document()
    doc2 = json.loads(json.dumps(doc))
    doc2["stage"] = "confirmatory"
    tampered = json.dumps(doc2, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(tampered.encode()).hexdigest() != PINNED_HASH


@pytest.mark.parametrize("bad", [
    {"CALIBRATION_FRACTIONS": (0.5, 0.1)},        # not ascending
    {"PURGE_S": 0.5},                              # below window+lag minimum
    {"RIDGE_ALPHAS": (1.0, 2.0)},                  # not the 7-point grid
])
def test_structural_invariants_rejected(bad, monkeypatch) -> None:
    for attr, value in bad.items():
        monkeypatch.setattr(contract, attr, value)
    with pytest.raises(ValueError):
        contract.validate_contract()

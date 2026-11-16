"""Deterministic governance-documentation checks (Phase 0; amended 2026-11-15).

These tests pin the *structure* of the governance artifacts — never a
verification outcome. They fail if a dataset is marked VERIFIED without a
decision date and a register-backed evidence pointer, if a retired
institutional item reappears as an active gate or is falsely claimed as
approved, if the trust-boundary documentation is removed, or if a
subject-identifier/credential pattern leaks into documentation.

Everything here is offline and reads only tracked repository files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# The five active compliance items of the personal public-data gate
# (docs/approvals.md; the institutional gate was retired 2026-11-15).
REQUIRED_COMPLIANCE_HEADINGS = [
    "1. Public-source verification",
    "2. License and terms compliance",
    "3. Provenance boundary (no restricted, private, identifiable, or former-lab data)",
    "4. No endorsement or affiliation",
    "5. Re-trigger review on scope change",
]

REQUIRED_ITEM_FIELDS = [
    "- Owner:",
    "- Requires:",
    "- Status:",
    "- Evidence:",
    "- Decision date:",
    "- Re-check date:",
]

ALLOWED_STATUSES = {
    "OPEN",
    "VERIFIED",
    "REJECTED",
    "RETIRED",
    "SUPERSEDED",
    "NOT_APPLICABLE",
}


def _approvals_text() -> str:
    path = DOCS / "approvals.md"
    assert path.exists(), "docs/approvals.md (compliance checklist) is missing"
    return path.read_text()


def _approval_item_blocks() -> list[tuple[str, str]]:
    """Return (heading, body) for each '### N. ...' item in approvals.md."""
    text = _approvals_text()
    blocks = []
    for chunk in text.split("\n### ")[1:]:
        lines = chunk.splitlines()
        blocks.append((lines[0].strip(), "\n".join(lines[1:])))
    return blocks


def test_compliance_checklist_covers_all_active_items():
    headings = [heading for heading, _ in _approval_item_blocks()]
    for required in REQUIRED_COMPLIANCE_HEADINGS:
        assert required in headings, f"approval checklist is missing item: {required}"


def test_no_retired_institutional_item_remains_active():
    """The retired institutional gate must never reappear as an active
    item: PI/IRB/DUA/institutional-account approvals are out of scope
    for this independent personal project."""
    for heading, _ in _approval_item_blocks():
        for term in ("PI ", "IRB", "DUA", "institutional AWS"):
            assert term not in heading, (
                f"institutional item must stay retired; found active item: {heading}"
            )


def test_every_approval_item_records_all_required_fields():
    blocks = _approval_item_blocks()
    assert len(blocks) >= len(REQUIRED_COMPLIANCE_HEADINGS)
    for heading, body in blocks:
        if not re.match(r"^[1-5]\.", heading):
            continue
        for field in REQUIRED_ITEM_FIELDS:
            assert field in body, f"item '{heading}' is missing field: {field}"


def test_approval_statuses_use_the_allowed_vocabulary():
    for heading, body in _approval_item_blocks():
        if not re.match(r"^[1-5]\.", heading):
            continue
        match = re.search(r"^- Status: (.+)$", body, flags=re.MULTILINE)
        assert match, f"item '{heading}' has no Status line"
        status = match.group(1).strip().rstrip("*").strip()
        assert status in ALLOWED_STATUSES, f"item '{heading}' uses unknown status: {status}"


def test_no_dataset_is_marked_verified_without_date_and_register():
    """A VERIFIED item must cite a decision date and point at the
    compliance register in docs/data_access.md; placeholder '—' keeps
    the item OPEN in practice."""
    for heading, body in _approval_item_blocks():
        if not re.match(r"^[1-5]\.", heading):
            continue
        status = re.search(r"^- Status: (.+)$", body, flags=re.MULTILINE).group(1).strip()
        if status != "VERIFIED":
            continue
        date = re.search(r"^- Decision date: (.+)$", body, flags=re.MULTILINE).group(1).strip()
        assert date not in {"", "—"}, (
            f"item '{heading}' claims VERIFIED without a decision date"
        )
        evidence = re.search(r"^- Evidence: (.+)$", body, flags=re.MULTILINE).group(1).strip()
        assert "data_access.md" in evidence, (
            f"item '{heading}' claims VERIFIED without pointing at the register entry"
        )


def test_retired_institutional_items_are_not_applicable_and_not_approved():
    """The retired gate must be recorded NOT_APPLICABLE with the explicit
    statement that this does not mean approved — never silently dropped,
    never claimed as approved."""
    text = _approvals_text()
    assert "NOT_APPLICABLE" in text, (
        "retired institutional items must be recorded NOT_APPLICABLE"
    )
    normalized = " ".join(text.replace("**", "").split())
    assert "does not mean approved" in normalized, (
        "approvals.md must state that NOT_APPLICABLE does not mean approved"
    )
    assert "no institutional approval is claimed" in normalized, (
        "approvals.md must state that no institutional approval is claimed"
    )


def test_checklist_states_external_data_path_is_closed():
    text = _approvals_text()
    assert "closed" in text.lower(), (
        "approvals.md must state that the external-data path stays closed "
        "until a dataset passes the compliance register"
    )


def test_data_access_documents_the_boundary_rules_and_the_register():
    text = (DOCS / "data_access.md").read_text()
    for required in (
        "Dataset compliance register",
        "Verification status",
        "Two-track boundary",
        "Prohibited crossings",
        "Artifact classification",
        "Approved storage locations",
        "Retention and destruction",
        "Incident-response trigger",
        "Real-data script acknowledgement rule",
        "--acknowledge-data-use",
    ):
        assert required in text, f"data_access.md is missing section/rule: {required}"


def test_decision_log_covers_dd_001_through_dd_006():
    index = (DOCS / "decisions" / "README.md").read_text()
    for number in range(1, 7):
        assert f"ADR-00{number}" in index, f"decision index is missing ADR-00{number}"
        path = DOCS / "decisions" / f"ADR-00{number}-"
        matches = list((DOCS / "decisions").glob(f"ADR-00{number}-*.md"))
        assert matches, f"ADR-00{number} file is missing (expected prefix {path}*.md)"
        content = matches[0].read_text()
        if number <= 3:
            # Owner-approved design decisions; recorded verbatim from the plan.
            assert "Approved by the owner on 2026-11-13" in content, (
                f"ADR-00{number} must record the owner approval of 2026-11-13"
            )
        else:
            # Either still explicitly Open, or owner-approved with the decision
            # date recorded (ADR-004..006 were decided by the owner on
            # 2026-11-14). A silent edit that erases either state must fail.
            assert ("Open" in content) or ("Approved by the owner on 2026-11-14" in content), (
                f"ADR-00{number} must record either an explicit Open status or an "
                "owner decision with its date"
            )


@pytest.mark.parametrize("pattern_name,pattern", [
    ("subject-identifier", re.compile(r"DBS_bG\d")),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
])
def test_no_identifier_or_credential_patterns_in_public_docs(pattern_name, pattern):
    """Governance docs must not leak subject-code or credential patterns.
    (The extension plan's literal 'DBS_bG*' pattern description has no digit
    and does not match.)"""
    files = sorted(DOCS.rglob("*.md")) + [REPO_ROOT / "README.md"]
    for path in files:
        match = pattern.search(path.read_text())
        assert match is None, (
            f"{pattern_name} pattern found in {path.relative_to(REPO_ROOT)} — "
            "public documentation must not contain identifiers or credentials"
        )

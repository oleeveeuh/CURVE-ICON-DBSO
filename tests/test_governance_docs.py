"""Deterministic governance-documentation checks (Phase 0).

These tests pin the *structure* of the governance artifacts — never an
approval outcome. They fail if an approval is claimed without a decision
date and out-of-repository evidence, if the trust-boundary documentation
is removed, or if a subject-identifier/credential pattern leaks into
documentation. They deliberately do NOT assert that any approval is
PENDING: recording a real, evidenced approval must not break the suite.

Everything here is offline and reads only tracked repository files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# The six gating items required by the extension plan (§6) and Prompt 1.
REQUIRED_APPROVAL_HEADINGS = [
    "1. PI scope approval",
    "2. IRB / human-subjects determination",
    "3. DUA permissions",
    "4. Institutional AWS and security architecture",
    "5. Public release of cohort size and aggregate metrics",
    "6. Real-data model explanations",
]

REQUIRED_ITEM_FIELDS = [
    "- Owner:",
    "- Status:",
    "- Decision date:",
    "- Evidence location",
    "- Restrictions:",
    "- Expiration / review date:",
]

ALLOWED_STATUSES = {
    "PENDING",
    "APPROVED",
    "APPROVED WITH RESTRICTIONS",
    "DENIED",
    "EXPIRED",
    "SUPERSEDED",
}

STATUSES_REQUIRING_DATE = ALLOWED_STATUSES - {"PENDING"}
STATUSES_REQUIRING_EVIDENCE = {"APPROVED", "APPROVED WITH RESTRICTIONS"}


def _approvals_text() -> str:
    path = DOCS / "approvals.md"
    assert path.exists(), "docs/approvals.md (redacted approval checklist) is missing"
    return path.read_text()


def _approval_item_blocks() -> list[tuple[str, str]]:
    """Return (heading, body) for each '### N. ...' item in approvals.md."""
    text = _approvals_text()
    blocks = []
    for chunk in text.split("\n### ")[1:]:
        lines = chunk.splitlines()
        blocks.append((lines[0].strip(), "\n".join(lines[1:])))
    return blocks


def test_approval_checklist_covers_all_gating_items():
    headings = [heading for heading, _ in _approval_item_blocks()]
    for required in REQUIRED_APPROVAL_HEADINGS:
        assert required in headings, f"approval checklist is missing item: {required}"


def test_every_approval_item_records_all_required_fields():
    blocks = _approval_item_blocks()
    assert len(blocks) >= len(REQUIRED_APPROVAL_HEADINGS)
    for heading, body in blocks:
        if not re.match(r"^[1-6]\.", heading):
            continue
        for field in REQUIRED_ITEM_FIELDS:
            assert field in body, f"item '{heading}' is missing field: {field}"


def test_approval_statuses_use_the_allowed_vocabulary():
    for heading, body in _approval_item_blocks():
        if not re.match(r"^[1-6]\.", heading):
            continue
        match = re.search(r"^- Status: (.+)$", body, flags=re.MULTILINE)
        assert match, f"item '{heading}' has no Status line"
        status = match.group(1).strip().rstrip("*").strip()
        assert status in ALLOWED_STATUSES, f"item '{heading}' uses unknown status: {status}"


def test_no_approval_is_claimed_without_date_and_evidence():
    """An APPROVED/APPROVED-WITH-RESTRICTIONS item must cite a decision date
    and an out-of-repository evidence location; other non-pending statuses
    must cite a decision date. Placeholder '—' means the item stays pending."""
    for heading, body in _approval_item_blocks():
        if not re.match(r"^[1-6]\.", heading):
            continue
        status = re.search(r"^- Status: (.+)$", body, flags=re.MULTILINE).group(1).strip()
        if status not in STATUSES_REQUIRING_DATE:
            continue
        date = re.search(r"^- Decision date: (.+)$", body, flags=re.MULTILINE).group(1).strip()
        assert date not in {"", "—"}, f"item '{heading}' claims '{status}' without a decision date"
        if status in STATUSES_REQUIRING_EVIDENCE:
            evidence = re.search(
                r"^- Evidence location[^:]*: (.+)$", body, flags=re.MULTILINE
            ).group(1).strip()
            assert evidence not in {"", "—"}, (
                f"item '{heading}' claims '{status}' without an evidence location"
            )


def test_checklist_states_institutional_track_is_disabled():
    text = _approvals_text()
    assert "disabled" in text.lower(), (
        "approvals.md must state that the institutional real-data track is "
        "disabled until approvals are recorded"
    )


def test_data_access_documents_the_two_track_boundary_rules():
    text = (DOCS / "data_access.md").read_text()
    for required in (
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

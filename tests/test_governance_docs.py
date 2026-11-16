"""Deterministic policy-documentation checks (amended 2026-11-16).

These tests pin the repository's standing policy: repository work is
NEVER blocked by an approval or verification workflow. They fail if a
blocking approval/verification gate reappears in the governance
documents or the external-data script, if institutional requirements
are recorded as anything other than ``NOT_APPLICABLE``, if the
no-endorsement disclaimer is lost, or if a credential pattern leaks
into documentation. Subject-identifier patterns in documentation
produce a non-blocking warning (a reminder, never a gate).

Everything here is offline and reads only tracked repository files.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# Operative gate language that must NOT reappear in the governance docs.
# (Historical narrative may describe the retired framework; these markers
# are the active-checklist/status machinery itself. "[REQUIRED]" belongs
# to that family: unknown metadata is recorded UNKNOWN / TO BE VERIFIED,
# never as a prerequisite.)
FORBIDDEN_GATE_MARKERS = [
    "- Status:",
    "Status: OPEN",
    "Verification status:",
    "external-data path is CLOSED",
    "path stays closed",
    "stays CLOSED",
    "closed until",
    "must pass the compliance register",
    "refuses to run",
    "refuse to run",
    "[REQUIRED]",
]

REQUIRED_DATASET_REGISTER_FIELDS = [
    "- Dataset name:",
    "- Canonical URL:",
    "- License or terms:",
    "- Permitted uses:",
    "- Attribution requirements:",
]


def _approvals_text() -> str:
    path = DOCS / "approvals.md"
    assert path.exists(), "docs/approvals.md (data & compliance notices) is missing"
    return path.read_text()


def test_no_blocking_gate_language_in_governance_docs():
    """The standing policy: governance documents contain no approval
    checklist, no OPEN/VERIFIED status machinery, and no closed-path
    claims. Work is never gated."""
    for name in ("approvals.md", "data_access.md"):
        text = (DOCS / name).read_text()
        for marker in FORBIDDEN_GATE_MARKERS:
            assert marker not in text, (
                f"{name} contains blocking gate language: {marker!r} — "
                "repository work must never be gated on approvals"
            )


def test_external_data_script_has_no_gate():
    """scripts/evaluate_real_kinematics.py must not enforce any register
    status or required acknowledgement — reminders only, always runs."""
    source = (REPO_ROOT / "scripts" / "evaluate_real_kinematics.py").read_text()
    assert "_check_compliance_gate" not in source, (
        "the execution-time compliance register gate must stay removed"
    )
    assert "VERIFIED" not in source, (
        "the script must not check any register verification status"
    )
    assert "--acknowledge-data-use\", required=True" not in source, (
        "the acknowledgement flag must stay optional (a provenance note, "
        "never a gate)"
    )


def test_output_location_rule_is_a_warning_not_an_error():
    """resolve_output_dir must WARN (not raise) on in-repo output paths
    outside the designated gitignored directories, and stay silent for
    the designated ones — a reminder, never a blocker."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from curve_icon_dbso.kinematics import resolve_output_dir
    finally:
        sys.path.pop(0)
    designated = resolve_output_dir(REPO_ROOT / "outputs" / "x", REPO_ROOT)
    assert designated == (REPO_ROOT / "outputs" / "x").resolve()
    with pytest.warns(UserWarning, match="designated gitignored directory"):
        result = resolve_output_dir(REPO_ROOT / "plans" / "x", REPO_ROOT)
    assert result == (REPO_ROOT / "plans" / "x").resolve()


def test_approvals_file_records_institutional_items_not_applicable():
    """The retired institutional requirements are recorded NOT_APPLICABLE
    with the explicit statement that this does not mean approved — never
    silently dropped, never claimed as approved."""
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


def test_approvals_file_keeps_the_no_endorsement_disclaimer():
    """The independent-personal-project disclaimer must survive any
    documentation rework."""
    normalized = " ".join(_approvals_text().replace("**", "").split())
    assert "independent personal project" in normalized
    assert "not sponsored by" in normalized
    assert "endorsement" in normalized


def test_data_access_documents_the_guidance_sections_and_the_register():
    text = (DOCS / "data_access.md").read_text()
    for required in (
        "Dataset register",
        "Two-track boundary",
        "Out-of-scope material",
        "Artifact classification",
        "Recommended storage locations",
        "Retention and destruction",
        "Incident-response trigger",
        "Real-data script reminders (non-blocking)",
    ):
        assert required in text, f"data_access.md is missing section/rule: {required}"
    for field in REQUIRED_DATASET_REGISTER_FIELDS:
        assert field in text, f"data_access.md register is missing field: {field}"
    assert "UNKNOWN" in text and "TO BE VERIFIED" in text, (
        "unknown dataset facts must be recorded UNKNOWN / TO BE VERIFIED"
    )
    assert "never stop" in text, (
        "the register must state that incomplete fields never stop work"
    )


def test_data_access_describes_non_blocking_script_reminders():
    """The script-behavior contract: reminders that never prevent a run."""
    text = (DOCS / "data_access.md").read_text()
    assert "never prevents a run" in text
    assert "warning (never an error)" in text or "never an error" in text


def test_decision_log_covers_dd_001_through_dd_006():
    index = (DOCS / "decisions" / "README.md").read_text()
    for number in range(1, 7):
        assert f"ADR-00{number}" in index, f"decision index is missing ADR-00{number}"
        matches = list((DOCS / "decisions").glob(f"ADR-00{number}-*.md"))
        assert matches, f"ADR-00{number} file is missing"
        content = matches[0].read_text()
        if number <= 3:
            # Owner design decisions; recorded verbatim from the plan.
            assert "Approved by the owner on 2026-11-13" in content, (
                f"ADR-00{number} must record the owner decision of 2026-11-13"
            )
        else:
            # Either still explicitly Open, or owner-decided with the date
            # recorded (ADR-004..006 were decided by the owner on
            # 2026-11-14). A silent edit that erases either state fails.
            assert ("Open" in content) or ("Approved by the owner on 2026-11-14" in content), (
                f"ADR-00{number} must record either an explicit Open status or an "
                "owner decision with its date"
            )


@pytest.mark.parametrize("pattern_name,pattern", [
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
])
def test_no_credential_patterns_in_public_docs(pattern_name, pattern):
    """Credential patterns in public documentation are a security issue and
    stay blocking (unlike privacy notices, which are non-blocking)."""
    files = sorted(DOCS.rglob("*.md")) + [REPO_ROOT / "README.md"]
    for path in files:
        match = pattern.search(path.read_text())
        assert match is None, (
            f"{pattern_name} pattern found in {path.relative_to(REPO_ROOT)}"
        )


def test_subject_identifier_pattern_in_public_docs_is_a_non_blocking_notice():
    """A subject-code pattern in documentation produces a WARNING — a
    reminder of the owner's privacy responsibilities, never a gate (the
    test passes either way; docs/approvals.md)."""
    pattern = re.compile(r"DBS_bG\d")
    files = sorted(DOCS.rglob("*.md")) + [REPO_ROOT / "README.md"]
    hits = []
    for path in files:
        if pattern.search(path.read_text()):
            hits.append(f"{path.relative_to(REPO_ROOT)}: subject-identifier pattern")
    if hits:
        for hit in hits:
            warnings.warn(
                f"non-blocking privacy notice — {hit} "
                "(docs/approvals.md owner responsibilities)",
                UserWarning,
                stacklevel=1,
            )
    assert True, "privacy notices never block repository work"

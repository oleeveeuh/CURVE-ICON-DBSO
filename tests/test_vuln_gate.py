"""Behavior of the exceptions-aware HIGH/CRITICAL gate (CON-004, CIC-007)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "vuln_gate.py"
EXCEPTIONS = REPO_ROOT / "docs" / "vulnerability-exceptions.yml"

TODAY = date.today()
FUTURE = (TODAY + timedelta(days=30)).isoformat()
PAST = (TODAY - timedelta(days=1)).isoformat()


def _report(tmp_path: Path, findings: list[dict], check: str = "vuln") -> Path:
    """Build a minimal trivy-shaped report (matches where the gate reads
    component from: PkgName for vuln, CauseMetadata.File for config)."""
    key = "Vulnerabilities" if check == "vuln" else "Misconfigurations"
    result = {"Target": "thing"}
    if check == "vuln":
        result[key] = [
            {
                "VulnerabilityID": f.get("id"),
                "PkgName": f.get("component"),
                "Severity": f.get("severity"),
                "FixedVersion": "9.9.9",
            }
            for f in findings
        ]
    else:
        result[key] = [
            {
                "ID": f.get("id"),
                "Severity": f.get("severity"),
                "CauseMetadata": {"File": f.get("component")},
            }
            for f in findings
        ]
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"Results": [result]}))
    return path


def _exceptions(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "exceptions.yml"
    import yaml

    path.write_text(yaml.safe_dump({"exceptions": entries}))
    return path


def _run(report: Path, exceptions: Path, check: str = "vuln") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--report",
            str(report),
            "--exceptions",
            str(exceptions),
            "--check",
            check,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _vuln(finding_id: str, severity: str, component: str = "pkg") -> dict:
    return {"id": finding_id, "component": component, "severity": severity}


VULN_ENTRY = {
    "id": "CVE-2026-0001",
    "scope": "vuln",
    "component": "pkg",
    "reason": "fixture exception",
    "approver": "owner",
    "expires": FUTURE,
}


def test_unmatched_high_blocks(tmp_path):
    code = _run(
        _report(tmp_path, [_vuln("CVE-2026-0001", "HIGH")]),
        _exceptions(tmp_path, []),
    )
    assert code.returncode == 1
    assert "unresolved HIGH/CRITICAL" in code.stdout


def test_critical_blocks(tmp_path):
    code = _run(
        _report(tmp_path, [_vuln("CVE-2026-0002", "CRITICAL")]),
        _exceptions(tmp_path, []),
    )
    assert code.returncode == 1


def test_valid_exception_allows_the_finding(tmp_path):
    code = _run(
        _report(tmp_path, [_vuln("CVE-2026-0001", "HIGH")]),
        _exceptions(tmp_path, [VULN_ENTRY]),
    )
    assert code.returncode == 0
    assert "EXCEPTED" in code.stdout


def test_expired_exception_does_not_allow_the_finding(tmp_path):
    expired = {**VULN_ENTRY, "expires": PAST}
    code = _run(
        _report(tmp_path, [_vuln("CVE-2026-0001", "HIGH")]),
        _exceptions(tmp_path, [expired]),
    )
    assert code.returncode == 1
    assert "EXPIRED exception" in code.stdout


def test_component_mismatch_blocks(tmp_path):
    code = _run(
        _report(tmp_path, [_vuln("CVE-2026-0001", "HIGH", component="other-pkg")]),
        _exceptions(tmp_path, [VULN_ENTRY]),
    )
    assert code.returncode == 1


def test_low_and_medium_findings_do_not_block(tmp_path):
    code = _run(
        _report(tmp_path, [_vuln("CVE-2026-0003", "LOW"), _vuln("CVE-2026-0004", "MEDIUM")]),
        _exceptions(tmp_path, []),
    )
    assert code.returncode == 0


def test_stale_exception_is_flagged_when_the_finding_is_fixed(tmp_path):
    code = _run(
        _report(tmp_path, []),  # finding gone — the fix shipped
        _exceptions(tmp_path, [VULN_ENTRY]),
    )
    assert code.returncode == 0
    assert "STALE exception" in code.stdout


def test_config_scope_findings_are_gated_separately(tmp_path):
    """A config finding needs a config-scoped exception with the matching
    id; a wrong-scope or wrong-id exception must not cover it."""
    config_finding = {"id": "AWS-9999", "component": "main.tf", "severity": "HIGH"}
    wrong_scope = _run(
        _report(tmp_path, [config_finding], check="config"),
        _exceptions(
            tmp_path,
            [{**VULN_ENTRY, "scope": "vuln", "component": "main.tf"}],
        ),
        check="config",
    )
    assert wrong_scope.returncode == 1

    allowed = _run(
        _report(tmp_path, [config_finding], check="config"),
        _exceptions(
            tmp_path,
            [{
                "id": "AWS-9999",
                "scope": "config",
                "component": "main.tf",
                "reason": "fixture exception",
                "approver": "owner",
                "expires": FUTURE,
            }],
        ),
        check="config",
    )
    assert allowed.returncode == 0


def test_real_exceptions_register_passes_against_the_real_config_scan(tmp_path):
    """End-to-end: regenerate the report the CI job would produce and run the
    real gate + real register (runs only if trivy is installed locally; the
    CI job runs this same pairing on every PR)."""
    import shutil

    trivy = shutil.which("trivy")
    if trivy is None:
        print("trivy not installed locally; skipping live end-to-end gate check")
        return
    report = tmp_path / "config-report.json"
    subprocess.run(
        [trivy, "config", "--format", "json", "--output", str(report), "infra"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    code = _run(report, EXCEPTIONS, check="config")
    assert code.returncode == 0, (
        f"real config scan + real exceptions register must pass:\n{code.stdout}"
    )

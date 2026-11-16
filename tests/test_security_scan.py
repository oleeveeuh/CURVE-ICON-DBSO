"""Negative fixtures for the scanner (CIC-007, plan §11.2; amended 2026-11-16).

Plants controlled fake findings in temporary directories and asserts the
severity behavior — credential findings (secrets, credential file types)
are BLOCKING (exit 1); privacy notices (subject identifiers, data file
types) are NON-BLOCKING (exit 0 with a WARN line) per docs/approvals.md.
Also asserts the scanner never echoes the secret content into its own
output, and that the tracked repository tree scans clean.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER = REPO_ROOT / "scripts" / "security_scan.py"


def _run(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCANNER), "--paths", *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        check=False,
    )


def _tree(tmp_path: Path, *files: tuple[str, str]) -> Path:
    for rel, content in files:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


FAKE_AKIA = "AKIA" + "A" * 16  # fake access key id (pattern-valid, not real)
FAKE_SECRET = "a" * 40


def test_planted_aws_access_key_is_caught_without_echoing_content(tmp_path):
    tree = _tree(tmp_path, ("src/app.py", f'client = Client("{FAKE_AKIA}")'))
    result = _run(tree)
    assert result.returncode == 1
    assert "aws-access-key" in result.stdout
    assert FAKE_AKIA not in result.stdout, "scanner must never print the secret itself"


def test_planted_private_key_is_caught(tmp_path):
    tree = _tree(tmp_path, ("cert.pem", "-----BEGIN RSA PRIVATE KEY-----\nabc\n"))
    result = _run(tree)
    assert result.returncode == 1
    assert "private-key-block" in result.stdout


def test_planted_github_token_is_caught(tmp_path):
    tree = _tree(tmp_path, (".env", "GITHUB_TOKEN=ghp_" + "x" * 36))
    result = _run(tree)
    assert result.returncode == 1
    assert "github-token" in result.stdout


def test_planted_subject_identifier_is_a_non_blocking_notice(tmp_path):
    """A subject-code match prints a WARN line but never fails the scan —
    privacy notices never block repository work (docs/approvals.md)."""
    tree = _tree(tmp_path, ("outputs/results.csv", "code,DBS_bG01,mae\n1,0.5,0.2\n"))
    result = _run(tree)
    assert result.returncode == 0, (
        f"subject identifiers must be non-blocking: {result.stdout}"
    )
    assert "WARN" in result.stdout
    assert "subject-identifier" in result.stdout


def test_planted_data_file_type_is_a_non_blocking_notice(tmp_path):
    """A data file type (privacy guidance) warns without failing."""
    tree = _tree(tmp_path, ("data/cohort.mat", "MATLAB binary junk"))
    result = _run(tree)
    assert result.returncode == 0, f"data file types must be non-blocking: {result.stdout}"
    assert "WARN" in result.stdout
    assert "data-file-type" in result.stdout


def test_credential_file_type_is_blocking(tmp_path):
    """A credential file type (security) still fails the scan."""
    tree = _tree(tmp_path, ("certs/server.pem", "not really a key, but the type counts"))
    result = _run(tree)
    assert result.returncode == 1
    assert "credential-file-type" in result.stdout


def test_legacy_identifiers_are_exempt_but_secrets_are_not(tmp_path):
    """The archived legacy scripts keep historical subject codes as evidence
    (docs/data_access.md); identifiers there are exempt, secrets are not."""
    tree = _tree(
        tmp_path,
        ("legacy/HFO_feature_extraction.py", 'subjects = ["DBS_bG01", "DBS_bG02"]'),
    )
    result = _run(tree)
    assert result.returncode == 0, f"legacy identifiers must be exempt: {result.stdout}"

    tree2 = _tree(tmp_path, ("legacy/secret.py", f'key = "{FAKE_AKIA}"'))
    result2 = _run(tree2)
    assert result2.returncode == 1, "secrets inside legacy/ must still be caught"


def test_generic_credential_assignment_is_caught(tmp_path):
    tree = _tree(tmp_path, ("config/settings.py", 'password = "hunter2hunter2"'))
    result = _run(tree)
    assert result.returncode == 1
    assert "generic-credential-assignment" in result.stdout


def test_clean_tree_passes(tmp_path):
    tree = _tree(
        tmp_path,
        ("src/app.py", "x = compute(y)\n"),
        ("docs/readme.md", "# Clean project\nNo findings expected.\n"),
        ("infra/main.tf", 'resource "aws_s3_bucket" "b" {}\n'),
    )
    result = _run(tree)
    assert result.returncode == 0, f"clean tree must pass: {result.stdout}"


def test_this_repository_tree_scans_clean():
    """The real CI gate: the tracked tree must produce no BLOCKING findings
    (no secrets, no credential file types). Privacy notices are printed but
    do not fail."""
    result = subprocess.run(
        [sys.executable, str(SCANNER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"repository tree failed its own security scan:\n{result.stdout}\n{result.stderr}"
    )

"""Negative fixtures for the scanner (CIC-007, plan §11.2; amended
2026-11-16 pre-publication review).

Plants controlled fake findings in temporary directories and asserts the
severity behavior — credential findings (secrets, credential file types),
subject identifiers, and non-allowlisted data file types are ALL BLOCKING
(exit 1); an explicitly allowlisted data file stays visible as a WARN but
does not fail. Also asserts the scanner never echoes the secret content
into its own output, and that the tracked repository tree scans clean.
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


def test_planted_subject_identifier_is_blocking(tmp_path):
    """A subject-code match FAILS the scan (2026-11-16 review: a real
    identifier in the public track is a privacy breach, not a warning)."""
    tree = _tree(tmp_path, ("outputs/results.csv", "code,DBS_bG01,mae\n1,0.5,0.2\n"))
    result = _run(tree)
    assert result.returncode == 1, (
        f"subject identifiers must be blocking: {result.stdout}"
    )
    assert "subject-identifier" in result.stdout
    assert "DBS_bG01" not in result.stdout, "scanner must not echo the identifier"


def test_planted_data_file_type_is_blocking(tmp_path):
    """A data file type fails the scan unless explicitly allowlisted."""
    tree = _tree(tmp_path, ("data/cohort.mat", "MATLAB binary junk"))
    result = _run(tree)
    assert result.returncode == 1, f"data file types must be blocking: {result.stdout}"
    assert "data-file-type" in result.stdout


def test_allowlisted_data_file_warns_without_failing(tmp_path):
    """The auditable exception path: an exact-path allowlist entry keeps the
    file visible as a WARN while passing the scan. Outside the repository
    (fixture trees) the exact path is the absolute path."""
    tree = _tree(tmp_path, ("data/cohort.mat", "MATLAB binary junk"))
    result = _run_with_allowlist(tree, allowed=str(tree / "data" / "cohort.mat"))
    assert result.returncode == 0, f"allowlisted data file must pass: {result.stdout}"
    assert "WARN" in result.stdout
    assert "allowlisted data file" in result.stdout
    # A DIFFERENT data file (not on the allowlist) still fails.
    tree2 = _tree(tmp_path, ("data/other.mat", "more junk"))
    result2 = _run_with_allowlist(tree2, allowed=str(tree / "data" / "cohort.mat"))
    assert result2.returncode == 1, "non-allowlisted data files must still fail"


def _run_with_allowlist(*paths: Path, allowed: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCANNER), "--allow-data-file", allowed,
         "--paths", *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        check=False,
    )


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
    (no secrets, no subject identifiers, no non-allowlisted data file
    types). Runs on every push and pull_request, so PR content is covered
    by the whole-tree gate before merge."""
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

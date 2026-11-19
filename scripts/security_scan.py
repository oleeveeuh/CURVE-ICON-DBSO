#!/usr/bin/env python3
"""Secret scanner with blocking privacy boundaries (CIC-007, amended
2026-11-16 pre-publication review).

Scans GIT-TRACKED files for:

1. credential patterns (AWS keys, private keys, GitHub/Slack tokens,
   generic credential assignments) and credential file types
   (.pem/.key/.p12/.pfx/.crt/.tfstate) — **blocking**: a leaked
   credential is a security vulnerability, so these fail the scan;
2. subject identifiers matching ``DBS_bG<digits>`` (plan §5.3) —
   **blocking** as of 2026-11-16: a real subject code in the public
   track is a privacy failure, not a style suggestion. The only
   exemption is the archived ``legacy/`` directory (historical evidence
   documented in docs/data_access.md);
3. data/model/database file types (raw data, spreadsheets, serialized
   models, databases) — **blocking** unless the exact repo-relative
   path is explicitly allowlisted (``--allow-data-file`` / module
   constant). The tracked tree must contain no data files; the
   allowlist is the auditable exception path, never a default.

Blocking scope covers pull requests too: CI runs this scanner over the
WHOLE tracked tree on every push and pull_request, so any PR content
that would introduce a finding fails before merge — there is no separate
diff-only mode to maintain.

Design constraints:

* stdlib only (no third-party dependency in the trust-critical path);
* findings print file:line + pattern NAME only — never the matched
  content, so the scanner itself cannot leak a secret into logs;
* exit code 1 for ANY blocking finding (credential OR privacy), 0
  otherwise;
* ``--paths`` accepts explicit files/directories for fixture tests;
  the default scans ``git ls-files`` (exactly what CI would commit).

This script doubles as the negative-test fixture target:
tests/test_security_scan.py plants fake secrets/identifiers in temporary
directories and asserts the severity behavior.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (pattern name, compiled regex) — findings report the NAME, not the value.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "aws-secret-key",
        re.compile(
            r"(?i)(?:aws)?_?secret_?access_?key[\"']?\s*[:=][\"']?[A-Za-z0-9/+=]{40}"
        ),
    ),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "github-token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|ghs_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})\b"),
    ),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "generic-credential-assignment",
        re.compile(
            r"(?i)\b(?:password|api_?key|secret)\b[\"']?\s*[:=]\s*[\"'][^\"']{8,}[\"']"
        ),
    ),
]

# Subject identifiers (plan §5.3). BLOCKING since the 2026-11-16
# pre-publication review: a subject code must never reach the public
# track, so a match fails the scan (the archived legacy/ directory keeps
# its documented exemption). The digit-anchored form keeps the literal
# "DBS_bG*" pattern descriptions in documentation from self-matching.
SUBJECT_IDENTIFIER_PATTERN = re.compile(r"DBS_bG\d")

# The sanctioned exception: the archived legacy scripts keep historical
# subject codes as evidence of the placeholder-data fallback
# (docs/data_access.md "Privacy rules observed in this repository").
# Secrets are still scanned there; only IDENTIFIERS are exempt.
IDENTIFIER_EXEMPT_PREFIXES = ("legacy/",)

# The scanner's own negative-test file plants fake secrets, identifiers and
# credentials to prove the patterns fire (tests/test_security_scan.py).
# Scanning it would flag its fixtures forever; its file TYPE is still
# checked, and the planted strings are test data, never real credentials.
CONTENT_EXEMPT_FILES = ("tests/test_security_scan.py",)

# Credential file types — BLOCKING (security: files of these types are
# credentials or infrastructure state that routinely embeds credentials).
CREDENTIAL_SUFFIXES = (
    ".tfstate",
    ".tfstate.backup",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
)

# Data/model/database file types — BLOCKING unless explicitly allowlisted
# (2026-11-16 review: raw data, spreadsheets, databases, and serialized
# models never belong in the public track; the allowlist below is the
# auditable exception path and stays empty by default).
DATA_SUFFIXES = (
    ".csv",
    ".mat",
    ".edf",
    ".xlsx",
    ".xls",
    ".parquet",
    ".pkl",
    ".joblib",
    ".nii",
    ".sqlite",
    ".sqlite3",
    ".db",
)

# Repo-relative paths (forward slashes) explicitly allowed to be data-type
# files. Keep this EMPTY unless a specific, documented file genuinely must
# be tracked — every entry is an auditable privacy exception.
DATA_FILE_ALLOWLIST: frozenset[str] = frozenset({
    # Measured aggregate load-test statistics (per-interval aggregates only,
    # never payloads) and the deterministic synthetic demo metrics — the two
    # committed evidence files the README and docs/deployment_report.md cite.
    "locust/results_stats_history.csv",
    "docs/synthetic_demo_metrics.csv",
})

MAX_FILE_BYTES = 5_000_000  # skip absurd files; lock files are well under this


def tracked_files() -> list[Path]:
    """Files git tracks (the commit surface); falls back to a plain walk."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        names = [n for n in result.stdout.decode().split("\0") if n]
        return [REPO_ROOT / n for n in names]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Outside a git checkout (e.g. fixture directories): walk everything.
        return []


def iter_files(paths: list[str]) -> list[Path]:
    """Expand --paths arguments; default to the tracked file list."""
    if not paths:
        return tracked_files()
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            out.append(path)
    return out


def rel_display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def is_identifier_exempt(path: Path) -> bool:
    """True when the file lives inside a sanctioned ``legacy/`` archive
    directory (any depth) — identifiers there are historical evidence;
    secrets are still scanned everywhere."""
    return any(parent.name == "legacy" for parent in path.parents)


def scan_files(
    files: list[Path], allowlisted_data: frozenset[str] = DATA_FILE_ALLOWLIST
) -> tuple[list[str], list[str]]:
    """Return (blocking, notice) human-readable finding lines.

    Lines carry file:line (content patterns) or file (types) plus the
    finding NAME — never the matched content. Subject identifiers and
    non-allowlisted data file types are BLOCKING; an allowlisted data
    file prints a WARN notice so its presence stays visible.
    """
    blocking: list[str] = []
    notices: list[str] = []
    for path in files:
        rel = rel_display(path)
        if path.is_dir() or not path.is_file():
            continue
        rel_posix = rel.replace("\\", "/")
        # Allowlist match key: repo-relative inside the repository, absolute
        # outside it (fixture trees in temporary directories).
        try:
            match_key = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            match_key = path.resolve().as_posix()
        suffix = path.suffix.lower()
        if suffix in CREDENTIAL_SUFFIXES:
            blocking.append(f"{rel}: credential-file-type ({suffix})")
        elif suffix in DATA_SUFFIXES:
            if match_key in allowlisted_data:
                notices.append(f"{rel}: allowlisted data file ({suffix}) — "
                               "documented exception, still worth a look")
            else:
                blocking.append(
                    f"{rel}: data-file-type ({suffix}) — raw data, spreadsheets, "
                    "databases and serialized models never belong in the public "
                    "track; move the file out or allowlist it explicitly via "
                    "--allow-data-file (docs/data_access.md)"
                )
        # No `continue` after the type checks: a private key inside a .pem
        # is still worth reporting by pattern; keep scanning the content.
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                blocking.append(f"{rel}: file-too-large-to-scan ({path.stat().st_size} bytes)")
                continue
            text = path.read_text(errors="replace")
        except OSError as exc:
            blocking.append(f"{rel}: unreadable-file ({exc.__class__.__name__})")
            continue

        exempt_ids = is_identifier_exempt(path)
        content_exempt = rel_posix in CONTENT_EXEMPT_FILES
        for lineno, line in enumerate(text.splitlines(), start=1):
            if content_exempt:
                continue  # the negative-test fixture file: patterns are its content
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    blocking.append(f"{rel}:{lineno}: secret-pattern ({name})")
            if not exempt_ids and SUBJECT_IDENTIFIER_PATTERN.search(line):
                blocking.append(
                    f"{rel}:{lineno}: subject-identifier (DBS_bG<digits>) — "
                    "a real subject code must never enter the public track "
                    "(docs/data_access.md incident response)"
                )
    return blocking, notices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[],
        help="explicit files/directories to scan (default: git-tracked files)",
    )
    parser.add_argument(
        "--allow-data-file",
        action="append",
        default=[],
        metavar="PATH",
        help="repo-relative data-type file to allow (repeatable); every "
             "allowlisted file still prints a WARN notice",
    )
    args = parser.parse_args(argv)

    files = iter_files(args.paths)
    if not files:
        print("security-scan: no files to scan", file=sys.stderr)
        return 2

    allowlisted = frozenset(
        Path(p).as_posix() for p in (*DATA_FILE_ALLOWLIST, *args.allow_data_file)
    )
    blocking, notices = scan_files(files, allowlisted_data=allowlisted)
    for line in notices:
        print(f"security-scan: WARN — {line}")
    if blocking:
        print(f"security-scan: FAIL — {len(blocking)} blocking finding(s)")
        for line in blocking:
            print(f"  {line}")
        print(
            "security-scan: matched content is never printed; remove the item "
            "from the public track and follow docs/data_access.md incident response."
        )
        return 1
    print(f"security-scan: OK ({len(files)} files scanned; no secrets, "
          f"subject identifiers, or non-allowlisted data files; "
          f"{len(notices)} allowlisted-file notice(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())

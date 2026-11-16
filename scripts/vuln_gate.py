#!/usr/bin/env python3
"""Exceptions-aware gate for HIGH/CRITICAL findings (CON-004, CIC-007).

Consumes a trivy JSON report (``--scan-type image`` vulnerabilities or
``--scan-type config`` misconfigurations) plus the time-limited exception
register (docs/vulnerability-exceptions.yml) and:

* exits 1 when any unresolved HIGH or CRITICAL finding exists —
  "unresolved" = no valid, unexpired exception covers it;
* exits 0 when every HIGH/CRITICAL finding is covered by a live
  exception, printing which exception covers which finding;
* warns about STALE exceptions (expired, or no matching finding anymore)
  so the register gets cleaned up instead of rotting;
* exits 2 on malformed inputs (bad report/exception file) so CI treats
  tooling failure as a failure, not as a pass.

The gate never edits reports and never prints payloads — findings are
summarized by ID/component/severity only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}


def load_report(path: Path, check: str) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"vuln-gate: cannot read report {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    results = data.get("Results") or []
    findings: list[dict] = []
    for result in results:
        if check == "vuln":
            items = result.get("Vulnerabilities") or []
            for item in items:
                findings.append({
                    "id": item.get("VulnerabilityID", "UNKNOWN"),
                    "component": item.get("PkgName", "unknown"),
                    # Debian source package (e.g. util-linux for bsdutils,
                    # libmount1, ...): lets one exception cover a source
                    # package's binary packages when a fix has not been
                    # released upstream at all.
                    "src": item.get("SrcName") or item.get("PkgName", "unknown"),
                    "severity": (item.get("Severity") or "UNKNOWN").upper(),
                })
        else:  # config
            items = result.get("Misconfigurations") or []
            target = result.get("Target", "unknown")
            for item in items:
                cause = item.get("CauseMetadata") or {}
                findings.append({
                    "id": item.get("ID", "UNKNOWN"),
                    "component": cause.get("File") or target,
                    "severity": (item.get("Severity") or "UNKNOWN").upper(),
                })
    return findings


def load_exceptions(path: Path) -> list[dict]:
    try:
        import yaml  # locked dependency (PyYAML); imported lazily for tests
    except ImportError as exc:
        print(f"vuln-gate: PyYAML is required to read {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"vuln-gate: cannot read exceptions {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    entries = data.get("exceptions")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        print(f"vuln-gate: {path}: 'exceptions' must be a list", file=sys.stderr)
        raise SystemExit(2)

    parsed: list[dict] = []
    for index, entry in enumerate(entries):
        try:
            expires = entry["expires"]
            if isinstance(expires, str):
                expires = date.fromisoformat(expires)
            elif isinstance(expires, datetime):
                expires = expires.date()
            parsed.append({
                "id": str(entry["id"]),
                "scope": str(entry["scope"]),
                "component": entry.get("component"),
                "reason": str(entry.get("reason", "")),
                "expires": expires,
            })
        except (KeyError, TypeError, ValueError) as exc:
            print(
                f"vuln-gate: {path}: exception entry #{index} is invalid ({exc}); "
                "each needs id, scope, expires (YYYY-MM-DD), reason",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True, type=Path, help="trivy JSON report")
    parser.add_argument(
        "--exceptions",
        required=True,
        type=Path,
        help="exceptions register (docs/vulnerability-exceptions.yml)",
    )
    parser.add_argument("--check", choices=["vuln", "config"], default="vuln")
    args = parser.parse_args(argv)

    today = date.today()
    findings = load_report(args.report, args.check)
    exceptions = load_exceptions(args.exceptions)

    blocking = [f for f in findings if f["severity"] in BLOCKING_SEVERITIES]
    live = [e for e in exceptions if e["scope"] == args.check and e["expires"] >= today]

    unmatched: list[dict] = []
    for finding in blocking:
        cover = next(
            (
                e
                for e in live
                if e["id"] == finding["id"]
                and (e["component"] is None or e["component"] == finding["component"])
                and (
                    e.get("component_source") is None
                    or e["component_source"] == finding.get("src")
                )
            ),
            None,
        )
        if cover is None:
            unmatched.append(finding)
        else:
            print(
                f"vuln-gate: EXCEPTED {finding['severity']} {finding['id']} "
                f"({finding['component']}) until {cover['expires'].isoformat()} — {cover['reason']}"
            )

    # Stale-exception hygiene: expired entries, or entries whose finding is
    # gone entirely (fix shipped — remove the exception). Reported BEFORE the
    # failure verdict so a failing run also shows register hygiene problems.
    expired = [e for e in exceptions if e["scope"] == args.check and e["expires"] < today]
    ids_present = {f["id"] for f in findings}
    fixed = [e for e in live if e["id"] not in ids_present]

    for entry in expired:
        print(f"vuln-gate: EXPIRED exception in register: {entry['id']} (until {entry['expires'].isoformat()}) — remove it")
    for entry in fixed:
        print(f"vuln-gate: STALE exception (finding no longer present): {entry['id']} — remove it from the register")

    if unmatched:
        print(f"vuln-gate: FAIL — {len(unmatched)} unresolved HIGH/CRITICAL finding(s) without a valid exception:")
        for finding in unmatched:
            print(f"  {finding['severity']} {finding['id']} in {finding['component']}")
        print(
            "vuln-gate: to ship anyway, add a time-limited entry to "
            "docs/vulnerability-exceptions.yml (id, scope, component, reason, expires, approver) "
            "recording the owner's decision — or fix and redeploy."
        )
        return 1

    print(
        f"vuln-gate: OK — {len(blocking)} HIGH/CRITICAL finding(s), all covered by live exceptions; "
        f"{len(findings)} total {args.check} finding(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

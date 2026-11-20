#!/usr/bin/env python3
"""Reporting-only regeneration from a PRESERVED transfer run directory.

Reads the run's ``run_summary.json`` (LOCAL-ONLY, gitignored), scrubs
every local-only (per-subject / held-out / ordinal-position) field, and
re-renders the public results document with the corrected presentation.
NO scientific recomputation — the numbers are exactly the preserved
run's; the run directory itself is never written to.

Outputs:
  --doc-out      public document (default docs/results_adaptive_transfer_real.md)
  --summary-out  sanitized aggregate JSON (default: alongside the doc,
                 gitignored path under outputs/)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curve_icon_dbso.transfer.summary import (  # noqa: E402
    _contains_local_only_key,
    render_results_doc,
    strip_local_only,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regenerate-transfer-report",
        description="Re-render the public transfer report from a preserved "
                    "run's summary (reporting-only; no recomputation).")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Preserved gitignored run directory (read-only)")
    parser.add_argument("--doc-out", type=Path,
                        default=ROOT / "docs" / "results_adaptive_transfer_real.md")
    parser.add_argument("--summary-out", type=Path,
                        default=ROOT / "outputs" / "transfer_v2"
                        / "sanitized_run_summary.json")
    args = parser.parse_args(argv)

    raw = json.loads((args.run_dir / "run_summary.json").read_text())
    residual = _contains_local_only_key(raw)
    sanitized = strip_local_only(raw)
    # the sanitized document is rendered from the SAME summary the driver
    # would produce post-fix; provenance block re-attached verbatim
    sanitized.setdefault("provenance", raw.get("provenance", {}))
    residual_after = _contains_local_only_key(sanitized)
    doc = render_results_doc(sanitized, str(args.run_dir))

    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(doc)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
    print(f"[regenerate] local-only fields removed from summary: {len(residual)}")
    print(f"[regenerate] residual after scrub: {len(residual_after)}")
    for path in residual_after[:10]:
        print(f"  RESIDUAL: {path}")
    print(f"[regenerate] wrote {args.doc_out}")
    print(f"[regenerate] wrote {args.summary_out}")
    return 0 if not residual_after else 1


if __name__ == "__main__":
    raise SystemExit(main())

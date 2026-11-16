#!/usr/bin/env python3
"""Drift demonstration (OBS-006/OBS-007): PSI per feature, synthetic only.

Loads the synthetic serving artifact, then computes Population Stability
Index (PSI) per feature for two deterministic samples:

* a clean synthetic sample  — PSI near zero on every feature;
* an intentionally SHIFTED synthetic sample — PSI crosses the documented
  0.2 warning threshold on the shifted features.

Privacy: the report contains per-feature PSI floats and aggregate bucket
shares only — never raw feature values, never per-row data (OBS-006).
The reference aggregates are baked into the artifact at training time.

NOT validated clinical drift monitoring: the reference is a synthetic
cohort and the 0.2 threshold is an educational convention (OBS-007), so
a warning here is a demonstration signal, not a clinical or model-decay
claim of any kind.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from curve_icon_dbso import artifact as artifact_mod  # noqa: E402
from curve_icon_dbso import drift as drift_mod  # noqa: E402
from curve_icon_dbso.api.payloads import shifted_payload_rows, synthetic_payload_rows  # noqa: E402

DISCLAIMER = (
    "Educational demonstration on SYNTHETIC data only — NOT validated "
    "clinical drift monitoring; PSI > 0.2 is a documented warning "
    "convention (OBS-007), not a clinical or model-decay claim."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact-dir",
        default=str(REPO_ROOT / "artifacts" / "synthetic-demo"),
        help="trained artifact directory (default: artifacts/synthetic-demo)",
    )
    parser.add_argument("--n-rows", type=int, default=500)
    parser.add_argument(
        "--cohort-seed",
        type=int,
        default=42,
        help="cohort/sampling seed — MUST match the artifact's training seed "
        "(42) so the clean sample draws from the reference distribution",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    _, manifest = artifact_mod.load_artifact(args.artifact_dir)
    aggregates = manifest.get("reference_aggregates")
    if not aggregates:
        print(
            "drift-demo: this artifact predates the Phase 5 reference "
            "aggregates — retrain with scripts/train_synthetic_model.py",
            file=sys.stderr,
        )
        return 2
    if manifest.get("training_seed") != args.cohort_seed:
        print(
            f"drift-demo: --cohort-seed {args.cohort_seed} does not match the "
            f"artifact's training seed ({manifest.get('training_seed')}); the "
            "clean sample would NOT be reference-distributed",
            file=sys.stderr,
        )
        return 2

    import pandas as pd

    # Same cohort seed as the artifact reference: a clean sample is a
    # resample of the reference distribution, so PSI ~ 0 is the honest
    # expectation (the generator derives its cohort from this seed).
    clean = pd.DataFrame(synthetic_payload_rows(args.n_rows, seed=args.cohort_seed))
    shifted = pd.DataFrame(
        shifted_payload_rows(args.n_rows, seed=args.cohort_seed, shift_fraction=0.5)
    )

    clean_report = drift_mod.compute_drift_report(aggregates, clean)
    shifted_report = drift_mod.compute_drift_report(aggregates, shifted)

    report = {
        "disclaimer": DISCLAIMER,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": date.today().isoformat(),
        "artifact": {
            "model_version": manifest["model_version"],
            "schema_version": manifest["schema_version"],
            "schema_hash": manifest["schema_hash"],
            "data_provenance": manifest["data_provenance"],
        },
        "psi_threshold": drift_mod.DEFAULT_PSI_THRESHOLD,
        "clean_synthetic_sample": {
            "n_rows": args.n_rows,
            "seed": args.cohort_seed,
            "psi_per_feature": clean_report["features"],
            "drifted_features": clean_report["drifted_features"],
        },
        "shifted_synthetic_sample": {
            "n_rows": args.n_rows,
            "seed": args.cohort_seed,
            "psi_per_feature": shifted_report["features"],
            "drifted_features": shifted_report["drifted_features"],
        },
    }

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"drift-demo: report written to {args.output}", file=sys.stderr)

    if not shifted_report["drifted_features"]:
        print("drift-demo: expected the shifted sample to cross the threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

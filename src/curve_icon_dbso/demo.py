"""Deterministic synthetic end-to-end demo.

Runs the full corrected pipeline — synthetic cohort generation,
clinical composites, prespecified feature matrix, leave-one-subject-out
evaluation with baselines, and provenance-labeled outputs — entirely
offline. Used by the quick start in the README and by the test suite
as the end-to-end determinism check.

The output is **pipeline verification on synthetic data**. It
demonstrates that the code runs and produces correctly aligned,
leakage-controlled metrics; it says nothing about real DBS outcomes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import features as features_mod
from . import reporting
from .config import DemoConfig
from .errors import CurveIconError
from .evaluation import baseline_models, default_models, run_group_cv
from .synthetic import generate_synthetic_cohort


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="curve-icon-demo",
        description=(
            "Run the corrected subject-grouped evaluation pipeline on a "
            "deterministic SYNTHETIC cohort (offline; no real data involved)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for outputs (default: outputs/synthetic_demo)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument(
        "--n-subjects", type=int, default=None,
        help="Number of synthetic subjects (default: 8)",
    )
    args = parser.parse_args(argv)

    try:
        defaults = DemoConfig()
        config = DemoConfig(
            seed=args.seed if args.seed is not None else defaults.seed,
            n_subjects=args.n_subjects if args.n_subjects is not None else defaults.n_subjects,
            output_dir=args.output_dir if args.output_dir is not None else defaults.output_dir,
        )
        _run(config)
    except CurveIconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _run(config: DemoConfig) -> None:
    cohort = generate_synthetic_cohort(n_subjects=config.n_subjects, seed=config.seed)
    enriched = features_mod.add_clinical_composites(cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)

    models = {**baseline_models(), **default_models(config.models)}
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)

    paths = reporting.save_outputs(
        output_dir=config.output_dir,
        results=results,
        n_subjects=int(groups.nunique()),
        n_recordings=int(len(cohort)),
        provenance=reporting.PROVENANCE_SYNTHETIC,
    )

    print(reporting.format_summary(
        results,
        n_subjects=int(groups.nunique()),
        n_recordings=int(len(cohort)),
        provenance=reporting.PROVENANCE_SYNTHETIC,
        cohort_note="(synthetic cohort — pipeline verification only)",
    ))
    print("\nOutputs written:")
    for label, path in paths.items():
        print(f"  {label:>8}: {path}")


if __name__ == "__main__":
    raise SystemExit(main())

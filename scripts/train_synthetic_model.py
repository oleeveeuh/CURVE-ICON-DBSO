#!/usr/bin/env python3
"""Train and package the deterministic SYNTHETIC serving artifact (Phase 2).

Thin wrapper around :func:`curve_icon_dbso.training.main` (also runnable
as ``python -m curve_icon_dbso.training``, which is the form the
container build uses). Creates a versioned artifact directory holding
``model.pkl``, ``schema.json`` and ``manifest.json``; refuses to
overwrite an existing artifact.

Research demonstration only: the model is trained on synthetic data and
the artifact is labeled synthetic — pipeline verification, NOT study
results, and never for clinical use.
"""

from curve_icon_dbso.training import main

if __name__ == "__main__":
    raise SystemExit(main())

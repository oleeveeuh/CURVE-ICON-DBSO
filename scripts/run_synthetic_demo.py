#!/usr/bin/env python3
"""Run the corrected evaluation pipeline on a deterministic SYNTHETIC cohort.

This is the repository's offline quick start: no private data, no
network, fully seeded. Outputs are labeled synthetic — pipeline
verification only.
"""

from curve_icon_dbso.demo import main

if __name__ == "__main__":
    raise SystemExit(main())

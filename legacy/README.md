# Legacy scripts (archived, superseded, known buggy)

These are the original analysis scripts, kept for provenance only.
They are **not** installed, **not** linted, and must not be run or
cited. Defects (documented in `docs/methodology.md`):

| Script | Defects |
|---|---|
| `run_pipeline.py` | Saved predictions in fold order but paired them with subjects/targets in original row order (misaligned metrics — the source of the contradictory R² values in the historical figures); feature selection, imputation and scaling fit on the full dataset before CV; **silently substituted random placeholder HFO features** when the aggregated HFO file was missing. |
| `visualization.py` | Rendered the misaligned saved predictions; labeled arbitrary PDP bands as "confidence bands (±1 SE)"; hard-coded "best model" selection. |
| `HFO_feature_extraction.py` | Broad `except` blocks returned `(0, 0)` so failures were indistinguishable from true zeros; undocumented one-file-per-subject selection; hardcoded sampling-rate default. |
| `extract_UPDRS.py` | Silent column-skipping; produced `_x`/`_y` merge-suffix artifacts that later appeared as "top biomarkers". |

Use `src/curve_icon_dbso/` and `scripts/` instead.

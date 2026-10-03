# Figures

Every figure in this folder states its own data provenance, cohort
size, and evaluation method. Regenerate them with:

```bash
python scripts/make_figures.py
```

| File | Content | Provenance |
|---|---|---|
| `pipeline_architecture.png` | What the corrected pipeline actually computes, including what was removed from the historical version | schematic (no data) |
| `cohort_modality_flow.png` | Modality merge and the 8-subject / 9-recording analysis cohort | historical pilot structure; counts unverified against source data |
| `synthetic_oof_verification.png` | Correctly aligned out-of-fold predictions vs baselines | **synthetic** — pipeline verification only, not study results |

`archive/` contains the historical figures produced by the legacy
pipeline. They are retained as evidence of the defects documented in
`docs/methodology.md` (misaligned predictions, leakage, placeholder
fallbacks). Their numbers are invalid and must not be cited.

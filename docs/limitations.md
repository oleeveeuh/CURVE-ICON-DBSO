# Limitations

Read this before drawing any conclusion from this repository.

## Statistical

- **n = 8 subjects (9 recordings).** This is the dominant limitation.
  LOSO evaluation yields eight held-out predictions; R² and Pearson r
  are close to meaningless at this size (negative expected R² under
  the null; sign flips from one subject to the next).
- **Bootstrap intervals undercover at this n.** The subject-cluster
  bootstrap (2,000 resamples) produces *descriptively* wide intervals,
  but the percentile method is known to be too narrow at pilot sample
  sizes; the intervals describe this cohort, not a population.
- **Permutation p-values are descriptive only.** At n = 8 they summarize
  how unusual the observed metric is against this cohort's own
  label-permutation null and cannot support a confirmatory claim; near
  n ≈ 31 (the kinematic cohort) they remain hypothesis-generating.
  Unequal recording counts stratify the permutation (conservative).
- **No model comparison is meaningful.** Differences between models
  are within noise; the corrected pipeline reports baselines so this
  is visible rather than hidden.
- **No feature selection could be validated.** Selecting "top"
  features from eight subjects is unfalsifiable; the pipeline uses a
  prespecified set instead.
- **Single center, retrospective, complete-case cohort.** The merge
  to subjects with all modalities induces unknown selection effects.

## Measurement

- HFO features summarize ~minutes of resting/task ECoG into two
  numbers per subject (band power, peak frequency). They are not
  validated HFO *detectors* (no event-level detection, no
  ripple/fast-ripple classification, no artifact rejection).
- State confounds (REST vs MOVE, anesthesia, medication) are not
  modeled; the repeated-recordings subject shows why this matters.
- **The kinematic prominence floor is binning-sensitive (prespecification
  caveat).** With the prespecified 1 s Welch windows the 0.5–4.0 Hz
  band holds only ~4 PSD bins, so a peak share below the 0.20 floor is
  nearly unreachable: the floor passes almost every channel and the
  ≥ 2 channel-block rule does the exclusion work. The alternative
  "share of total-spectrum power" reading would discriminate sharply.
  **Resolved 2026-11-16: the owner retained the frozen in-band
  reading** (`kinematics.py` unchanged); the sensitivity stands as a
  caveat on QC decisions from datasets with similar PSD binning.
- The historical cortical-thickness "features" included accidental
  pandas merge-suffix columns (`*_x`, `*_y`); any interpretation of
  them as biomarkers is void.

## Engineering / reproducibility

- The pilot dataset is available to the owner locally under the
  distributor authorization; the first authorized real-data run is
  complete (2026-11-16) and its corrected permutation analysis is
  **negative** — no model beat both baselines on both MAE and RMSE,
  and no statistic survives the exploratory multiplicity correction
  (raw p = .064 best, Holm-adjusted .642; the earlier p ≈ .045 was
  withdrawn after the null-scoring defect was found and fixed). The
  sanitized report is [`docs/results_kinematic_real.md`](results_kinematic_real.md);
  all results remain exploratory single-cohort descriptions, and any
  variant re-analysis on these 31 subjects is confirmatory in name
  only — a new cohort or untouched evaluation set is required.
- Historical results in `figures/archive/` are invalid (misaligned
  OOF pairing, full-dataset preprocessing leakage, a random
  placeholder fallback) and are retained solely to document the
  correction.
- NumPy RNG determinism is best-effort across versions; identical
  seeds give identical results for a fixed environment (this is what
  the tests check).

## What would have to change for real conclusions

1. Larger, ideally multicenter cohort (order 10² subjects).
2. Pre-registered analysis plan and prespecified primary endpoint.
3. External validation cohort; prospective design.
4. Clinically validated HFO detection with artifact QC.
5. Uncertainty quantification that actually models the cohort
   (e.g., hierarchical bootstrap), not band-aids on point estimates.

## Follow-on three-arm study (2026-11-16, frozen contract v1.0.0; corrected v1.0.1, 2026-11-17)

- **v1.0.0 statement, superseded in part.** The original v1.0.0 summary
  ("global null; transfer NOT successful; DMD does not outperform band
  power") overstated its own coverage: a post-run review found the MOVE
  transfer criterion's MAE half never computed (only baseline
  correlations were emitted), a shift null whose effective circular
  displacement could be a single update bin (its p = .001 is
  **withdrawn** pending the corrected null), the DELTA-combined block
  never built (so "state-change evidence: none" was not fully tested),
  and a Holm family run over 8 of its 10 members (the reported adjusted
  minimum .83 is superseded; with the two missing members entered
  conservatively at p = 1 the family minimum is 1.0). What remained
  defensible from v1.0.0: no tested model met the promotion rule, REST
  feature expansion did not help, separate DELTA spectral/dynamics
  blocks did not promote, MOVE correlations were extremely small
  (persistence r ≈ .92 dwarfs every decoder's r ≈ .06), and no single
  subject dominated the combined-block error.
- The **v1.0.1 corrective rerun** (identical inputs, verified by SHA-256
  input manifests) supersedes it and closes the matrix: **global null,
  formally** — no promotion in any block (ten-test Holm family minimum
  adjusted p = 1.0), state-change evidence none (`delta_combined`
  comparators beat REST-combined on error metrics but none meets the
  promotion rule), movement-state transfer NOT successful in code
  (decoder MAE .721 vs persistence .282 / block-mean .629), corrected
  circular and blockwise shift nulls both p = .002. DMD diagnostics
  repaired: the ≈3.2e3 multi-step number is near-unitary eigenvalue
  compounding (~600 steps), not a linearity claim — the interpretable
  one-step relative error is ≈ .15. Definitive verdicts and numbers:
  `docs/results_followon_real.md` (v1.0.1 section).
- These negatives are specific to this cohort, this 500 ms/100 ms protocol, and this envelope definition; the frozen contract's external-validation requirements (independent cohort, unchanged version) still apply. The earlier "order 10² subjects / ≈800 subjects" figures were illustrative correlation-power calculations for detecting r = .1, not minimum sample sizes for this nested predictive pipeline; a replication design sizes on the paired MAE difference versus persistence, its uncertainty, expected usable-window yield, and an external-validation margin.

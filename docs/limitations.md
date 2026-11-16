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

- The restricted dataset is not available for verification; the
  corrected evaluation logic is verified on deterministic synthetic
  data only.
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

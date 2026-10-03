# Methodology

Status: **exploratory retrospective feasibility analysis; research
prototype; not for clinical use.**

## Research question

Do features derived from intracranial ECoG (high-frequency
oscillations, HFO) and routine clinical measures carry any signal
about the magnitude of motor improvement after DBS, measurable at
pilot scale (n = 8 subjects)?

## Cohort

Structure from the historical pilot outputs (unverified against the
source data, but consistent across the tracked artifacts):

- **8 unique subjects, 9 recordings** — one subject contributes two
  recordings (REST and MOVE states); the rest contribute one.
- Modality merge: UPDRS assessments + ECoG recording metadata +
  cortical-thickness measures + HFO features, joined on subject.
- Subjects lacking any modality were excluded, which is what reduced
  the pool to eight.

Every repeated recording of a subject is treated as belonging to one
evaluation unit. Both the subject count and the recording count are
reported with every result.

## Features actually implemented

Channel level (`curve_icon_dbso.hfo`), per ECoG channel:

- 4th-order Butterworth band-pass (default 80–500 Hz, zero-phase
  `filtfilt`), validated against Nyquist.
- Welch PSD (1 s Hann/hamming segments, 50 % overlap): **band power**
  (PSD integral over the band) and **peak frequency** (argmax within
  band).
- Per-channel QC status + failure reason; failed channels are never
  zero-filled.

Subject level (`curve_icon_dbso.features`): mean of OK channels for
the two HFO features; UPDRS-derived bradykinesia / rigidity / tremor
composites per side and their asymmetry indices; disease duration;
age. The modeling feature set is **prespecified** (7 features —
listed in `features.PRESPECIFIED_FEATURES`), fixed a priori from the
hypothesis, with no data-driven selection at any point.

**Not implemented (and therefore not claimed):** phase-amplitude
coupling, artifact rejection (historical artifact columns were
hard-coded zeros), SNR, temporal-stability metrics, spatial-
distribution summaries, bootstrap confidence intervals, and
per-patient uncertainty. Claims of "18 neural biomarkers" or "108
features" in the historical README were unsupported: the tracked
artifacts show 32 engineered columns, of which 2 are HFO-derived.

## Evaluation design (corrected)

Subjects — not rows — are the evaluation units.

1. **Splitter:** leave-one-subject-out (LOSO) via
   `LeaveOneGroupOut` for cohorts of ≤ 12 subjects. Every fold holds
   out all recordings of exactly one subject. (Above 12 subjects the
   default becomes 5-fold `GroupKFold`.)
2. **Fold-local preprocessing:** median imputation and standard
   scaling live inside an sklearn `Pipeline` and are fit on each
   training fold only. **No feature selection is performed** — the
   feature set is prespecified.
3. **Baselines:** `DummyRegressor` (mean and median) run through the
   identical procedure. A model is not described as useful unless it
   beats both on MAE and RMSE.
4. **Aligned out-of-fold predictions:** a preallocated array is
   written back with `oof[test_idx] = pipe.predict(X.iloc[test_idx])`.
   This is the fix for the historical bug in which fold-order
   predictions were zipped with original-row-order subjects — the
   direct cause of the contradictory historical metrics (figure:
   R² = −0.672; README: R² = 0.713). A regression test
   (`tests/test_alignment.py`) pins this with deliberately
   nonsequential fold indices.
5. **Metrics:** MAE, RMSE, R², Pearson r, plus subject and recording
   counts and per-subject OOF predictions/errors.
6. **No tuning:** model hyperparameters are fixed and documented
   (`config.py`); nothing is optimized against held-out subjects.

## Interpretation guardrails

- With 8 held-out subjects, R² and correlation are **highly
  unstable**: a single subject can change the sign of R². Expected
  R² under the null is *negative* at this n.
- Feature-importance / SHAP-style explanations, where computed, are
  descriptive and hypothesis-generating only; mean-SHAP sign is not a
  validated direction of effect, and rankings are not reliable at
  this sample size. The corrected pipeline therefore ships none by
  default.
- Mixed-effects models are not used: with eight subjects (one with
  two recordings) there is insufficient repeated-measure structure.

## Historical results are invalidated

All previously published numbers and figures from the legacy
pipeline are considered invalid (misaligned predictions + leakage +
a random placeholder fallback). They are archived, clearly marked,
under `figures/archive/`. No study performance is currently claimed;
the only results in this repository come from the deterministic
synthetic verification run and are labeled as such.

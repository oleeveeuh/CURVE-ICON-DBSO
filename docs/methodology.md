# Methodology

Status: **exploratory retrospective feasibility analysis; research
prototype; not for clinical use.**

> ** November 2026 — status note.** The original DBS-outcome analysis
> (target: `updrs_improvement`) is **paused**: the clinical table
> (`UPDRS_data.csv`) was unavailable at analysis time and could not be
> recovered (not derivable from any distributed script — verified by
> inspection). A prespecified kinematic analysis below replaces it as
> the active analysis; the outcome question resumes if the clinical
> file is recovered. The prespecified target, predictors and estimator
> below were frozen **before** any model was fit to real data and
> before any real-data feature/target correlation was computed.

## Kinematic analysis (active; prespecified)

**Research question (pivoted).** Do resting-state intracranial HFO
features and basic patient covariates relate to concurrent motor
performance — specifically finger-tapping rate — measured with a
sensor glove during intraoperative recordings? This is an exploratory
neural–motor relationship study. It is **not** DBS-outcome prediction:
glove kinematics measure motor state during the recording session
(under intraoperative medication/anesthesia conditions), not treatment
response.

**Primary target (single, prespecified).** `tap_rate_mean` — the
median dominant tapping frequency (Hz) of the glove signal during
MOVE blocks:

1. For each MOVE block (`restmove_segs` column 4 = 1; columns 1–2 =
   start/end samples, treated as 0-based half-open) and each of the 5
   glove channels: detrend the block, Welch PSD (1 s Hamming windows,
   50 % overlap), restrict to 0.5–4.0 Hz (physiological tapping range),
   take the dominant peak frequency and its share of total in-band
   power.
2. A channel–block contributes only if that share is ≥ 0.20
   (prespecified prominence floor — guards against flat/non-oscillating
   fingers being counted as "taps").
3. `tap_rate_mean` = median over all contributing channel–blocks.
4. A recording passes QC only if ≥ 2 channel–blocks contribute;
   otherwise it is excluded from modeling with an explicit QC record
   (never imputed silently, never zero-filled).

**Predictors (prespecified, five — fixed before any model fit).**

| Feature | Definition |
|---|---|
| `hfo_power_rest_mean` | Mean 80–500 Hz Welch band power over **REST** blocks (column 4 = 0), averaged over blocks and QC-OK ECoG channels |
| `hfo_peak_freq_rest_mean` | Mean in-band PSD peak frequency over the same REST-block, OK-channel records |
| `age` | Localization workbook Sheet1, first row per subject |
| `dbs_volt` | Sheet1 `VOLT STIM`, first row per subject |
| `dbs_freq` | Sheet1 `FREQ STIM`, first row per subject |

The HFO features deliberately use REST blocks only: the hypothesis is
that the resting HFO state relates to movement performance measured in
a different block (cross-state design, less circular than predicting
movement from features computed during movement). The full-recording
HFO features remain available but are not part of the prespecified
set.

**Evaluation.** Unchanged from the corrected design below:
leave-one-subject-out cross-validation, fold-local median imputation +
scaling inside an sklearn Pipeline, `DummyRegressor` mean/median
baselines, fixed hyperparameters, aligned out-of-fold predictions,
MAE/RMSE/R²/Pearson with subject and recording counts. Each recording
is one row; subjects contribute at most one recording in this dataset.

**Provenance.** Artifacts are labeled `DABI — authorized local copy
(owner agreement); exploratory kinematic target; NOT DBS outcome` and
are written outside the repository by default. No participant-level
values are committed.

**Known caveats (stated up front).** n ≈ 31 recordings is still small;
R² and correlation remain unstable. Glove sensors are uncalibrated
across subjects, which is why the target is a dimensionless rate
rather than amplitude. Tapping conditions are intraoperative
(medication/anesthesia state varies by recording; the `ANAESTHESIA`
column records it but is not a predictor). At n ≈ 31 with 5
predictors, any "finding" is hypothesis-generating only.

**Prespecification note (resolved 2026-11-16).** The prominence floor
("dominant peak carries ≥ 0.20 of total in-band power") is sensitive to
the PSD binning: with the prespecified 1 s Welch windows the frequency
resolution is 1 Hz, so the 0.5–4.0 Hz band contains only ~4 bins and a
peak-bin share below 0.20 is essentially unreachable — the floor as
worded passes nearly every channel and the ≥ 2 channel-block QC rule
carries the exclusion. An alternative reading (peak share of
*total-spectrum* power) would discriminate sharply but deviates from
the frozen text. **Resolved by the project owner (2026-11-16): the
frozen in-band wording is retained** — `kinematics.py` implements it
unchanged, and the binning sensitivity above remains the standing
caveat on any external-data QC decision that rests on it.

## Uncertainty quantification (implemented 2026-11-13)

Both procedures resample or permute **subjects, never rows**
(`bootstrap.py`), and both are deterministic given their seed.

**Subject-cluster bootstrap (SCI-008).** MAE and RMSE confidence
intervals from 2,000 subject-level resamples (seed 42, 95% percentile
intervals), each resample drawing subjects with replacement and
carrying all of that subject's recordings. At pilot n the percentile
method undercovers: the intervals describe this cohort, not the
population.

**Label-permutation null (SCI-009).** At least 200 permutations of the
target at the subject level, each rerun through the **complete**
grouped-CV workflow (same folds, fold-local preprocessing, aligned OOF
predictions). Subjects are shuffled as blocks, stratified by recording
count (a 2-recording subject's target block cannot move onto a
1-recording subject; singleton strata keep their targets, which makes
the null slightly conservative). The empirical p-value uses the
conservative convention
`p = (1 + #{null at least as favorable as observed}) / (1 + n)` —
never zero, equality counts as favorable. **Permutation results are
descriptive at n = 8 and remain hypothesis-generating near n ≈ 31**;
they summarize how unusual the observed metric is against the cohort's
own label-permutation null.

**Stage labeling (SCI-011).** Every run is labeled `exploratory` or
`confirmatory` in its outputs and lineage records; this correction
phase is exploratory by definition. Any change made after viewing real
results resets the analysis to exploratory unless a new held-out
evaluation exists.

## Run log (append-only; DOC-002)

| Date | Commit | Stage | Data | Entry |
|---|---|---|---|---|
| 2026-10-03 | `382342f`–`eb45b11` | exploratory | synthetic | Corrected pipeline (LOSO, fold-local preprocessing, aligned OOF, baselines) committed and verified on synthetic fixtures; historical results retracted and archived (`bc13544`, `9b63042`). |
| 2026-11-13 | `eb45b11` + working tree | exploratory | synthetic | Phase 1 hardening: `uv.lock` environment lock; subject-cluster bootstrap CIs (2,000 × seed 42); label-permutation null (≥ 200, conservative p); MLflow lineage (`tracking.py`, local store, synthetic-only); optional fixed-parameter XGBoost parity exercise (DD-003); kinematic extraction implemented and verified on synthetic fixtures (`kinematics.py`). No real-data run has occurred. |
| 2026-11-14 | `eb45b11` + working tree | exploratory | synthetic | Phase 2 serving path: versioned serving schema v1.0.0 (the prespecified 7-feature outcome contract; `schema.py`); deterministic synthetic artifact build with LOSO metrics, baselines and DD-006 promotion rule (`training.py`, `artifact.py` — selected Linear Regression, `beats_baseline: true`, byte-identical reruns); FastAPI service (`/health`, `/ready`, `/model-info`, `/predict`; no `/explain` per ADR-002); non-root container definition with readiness health check. All verification on synthetic data only; no real-data run has occurred. |
| 2026-11-14 | working tree | exploratory | synthetic | Phases 3–4: CI security gates and OIDC delivery workflows; single-root Terraform for the public synthetic AWS track (CloudFront → ALB → ECS Fargate, scale-to-zero, budgets, versioned private artifact bucket, 14-day logs) with one-time remote-state bootstrap. Owner decisions DD-004…006 ratified (ADR-004…006 Approved 2026-11-14). Region constraint: the new-AWS-experience project account's SCP denies OIDC providers and other Regions — deploys for this account are local by recorded decision. All verification synthetic-only; no real-data run. |
| 2026-11-15 | working tree | exploratory | synthetic | Phases 4–5 deployed and verified end to end (image by digest, CloudFront smoke tests, idempotent second plan), then Phase 5: 4 alarms + SNS topic + 2 low-cardinality log metric filters + dashboard; Locust load test (34,177 requests, 0 failures, p95 110 ms, zero restarts, flat memory — synthetic payloads only); versioned synthetic reference aggregates in the artifact + PSI drift demonstration (clean max 0.005, shifted all 7 features above the educational 0.2 threshold; not clinical drift monitoring); owner-run FAILURE_DEMO rollback exercise (unhealthy ~40 s → circuit-breaker rollback ~70 s, zero dropped requests, converged). Teardown the same day (owner decision): destroy verified by the after-destroy orphan check (all empty; remote-state bucket + lock table retained). Measured record: `docs/deployment_report.md`. No real data touched at any point. |

Entries are appended after runs complete; no entry is ever rewritten.

## Original outcome analysis (paused)

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
distribution summaries, and per-patient uncertainty. (Cohort-level
bootstrap confidence intervals and the label-permutation null ARE
implemented since 2026-11-13 — see "Uncertainty quantification"
above; per-subject uncertainty intervals are not.) Claims of "18
neural biomarkers" or "108 features" in the historical README were
unsupported: the tracked artifacts show 32 engineered columns, of
which 2 are HFO-derived.

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

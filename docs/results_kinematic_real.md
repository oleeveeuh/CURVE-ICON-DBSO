# Real-data kinematic evaluation — sanitized negative-results report

**Written for:** reviewers and future readers of this repository (aggregates only; no subject codes, no per-subject values, no redistributed data).

**Stage: exploratory.** Every number below is from the cohort's own
cross-validation; nothing here is a confirmatory or clinical claim.
**This analysis predicts concurrent finger-tapping rate** (taps/s) from
resting ECoG features and recorded covariates in a single session. It
does **not** predict DBS response, treatment outcome, or clinical
benefit, and no such claim is made or implied.

**Headline (negative):** no candidate model satisfied the prespecified
usefulness rule (beat BOTH the mean and median baselines on BOTH MAE
and RMSE). With the corrected label-permutation null, no statistic is
significant even before multiplicity correction (smallest raw
p = .064), and the smallest Holm-adjusted p across the tested family
is .642. The prespecified resting-ECoG feature set did not establish
practically useful out-of-subject prediction of tapping rate in this
cohort.

## History and correction of record

The first authorized run (2026-11-16, commit `a3f4984`, 200
permutations) contained a statistical implementation defect: each
permutation's null statistic was scored against the ORIGINAL targets
instead of that permutation's permuted targets, invalidating every
reported p-value — including the gradient-boosting RMSE/R² p ≈ .0448
circulated on 2026-11-16. **That value is withdrawn.** The scoring was
fixed (`src/curve_icon_dbso/bootstrap.py`; regression test in
`tests/test_permutation.py` mirrors scikit-learn's
`permutation_test_score` definition) and the full evaluation was
re-run from the same recorded inputs (5,000 permutations, seed 42).
The observed cross-validated metrics were never affected by the defect
and are identical across both runs. The flawed first-run outputs
remain only in gitignored local storage and are superseded by
`outputs/kinematic_results_corrected/`.

## Cohort flow

| Stage | Count |
|---|---|
| Subject folders available in the authorized local copy | 32 |
| Excluded before conversion | 1 — the folder contained no standard-format preprocessed recording file |
| Recordings converted and passing prespecified tap QC | 31 |
| Subjects (one recording each; `subject` = group) | 31 |

The excluded subject is identified nowhere in this report or its
outputs (the code appears once in a historical commit message; see
`docs/data_access.md`). Rest-event QC across the cohort: 1,408 rest
blocks passed, 2 failed (excluded upstream of feature extraction).
Tap-event QC: all 31 recordings `ok`; contributing tap blocks per
recording: min 15, median 20, max 30.

## Predictors and target (prespecified; `src/curve_icon_dbso/kinematics.py`)

| Variable | Definition | Observed range (n = 31) | Missing |
|---|---|---|---|
| `hfo_power_rest_mean` | Mean HFO-band power of resting ECoG, averaged over qualifying strip channels (unit: spectral power as extracted by the prespecified pipeline) | [1.64, 3573], median 11.2, IQR 6.8–29.5 — strongly right-skewed | 2 |
| `hfo_peak_freq_rest_mean` | Mean peak frequency within the HFO band (Hz) | [99.3, 420], median 189.3, IQR 141.5–253.2 | 2 |
| `age` | Years | [52, 78] | 0 |
| `dbs_volt` | Stimulation voltage covariate as recorded in the localization workbook (recorded values span [0, 1]) | [0, 1] | 0 |
| `dbs_freq` | Stimulation frequency covariate as recorded in the same workbook (recorded values span [0, 1]) | [0, 1] | 0 |
| `tap_rate_mean` (**target**) | Mean finger-tapping rate (taps/s), QC-gated to the prespecified 0.5–4 Hz window | [1.0, 3.0], mean 1.81, median 2.00, SD 0.67 | 0 |

Outlier checks: the target lies in a physiologically ordinary band with
no extreme values; `hfo_power_rest_mean` is right-skewed (max ≈ 320×
the median), which motivates the tree-based candidates and the median
baseline; no other feature shows isolated extremes beyond the skew
noted. The four missing feature values (two recordings × two HFO
features) are handled by fold-local median imputation inside the
cross-validation pipeline (never full-dataset imputation); covariates
and target are complete.

## Evaluation protocol

- **Splitter:** 5-fold GroupKFold, subject-grouped — each held-out fold
  is a set of whole subjects (fold sizes 6/6/6/6/7; every subject's
  single recording is held out exactly once). The generated outputs
  label the scheme explicitly; no output claims leave-one-subject-out.
- **Preprocessing:** fold-local median imputation + standardization,
  refit inside each training fold (leakage-controlled; no feature
  selection; the feature set is prespecified).
- **Models:** fixed, documented hyperparameters — RandomForest(200
  trees, depth 3), GradientBoosting(50, depth 2, lr 0.05), Linear
  Regression — plus DummyRegressor mean and median baselines. No
  tuning anywhere.
- **Uncertainty:** subject-cluster bootstrap (2,000 resamples, seed 42)
  percentile 95% intervals; label-permutation null through the complete
  grouped-CV workflow (5,000 permutations, seed 42), conservative
  p = (1 + #{null ≥ as favorable}) / (1 + 5000).
- **Provenance:** commit `493b9a6` + working tree (the scoring fix);
  lock hash `1f81065a…`, feature-schema hash `58809e8c…`, input table
  SHA-256 `e7a4e46f…` — identical lock/schema/input hashes to the
  withdrawn first run, so the only change is the corrected null
  scoring.

## Results — all models and baselines (out-of-fold, n = 31)

| Model | MAE [95% CI] | RMSE [95% CI] | R² | Pearson r | Beats rule |
|---|---|---|---|---|---|
| Mean baseline | 0.559 [0.426, 0.692] | 0.680 [0.539, 0.802] | −0.074 | −0.370 | — |
| Median baseline | 0.484 [0.306, 0.645] | 0.684 [0.546, 0.798] | −0.087 | — | — |
| Random Forest | 0.582 [0.438, 0.724] | 0.708 [0.549, 0.870] | −0.165 | 0.059 | **false** |
| Gradient Boosting | 0.523 [0.393, 0.664] | 0.650 [0.494, 0.812] | +0.018 | 0.349 | **false** |
| Linear Regression | 0.765 [0.507, 1.098] | 1.163 [0.612, 1.730] | −2.145 | −0.011 | **false** |

The prespecified rule: beat BOTH baselines on BOTH MAE and RMSE.
Gradient Boosting's MAE (0.523) beats the mean baseline (0.559) but
not the median baseline (0.484); no model passes on both metrics, so
**no candidate is promoted** (`beats_baseline: false` everywhere, per
DD-006/SCI-014). Bootstrap intervals overlap heavily across models and
baselines — consistent with "no detectable signal", not "a narrow
miss".

## Corrected permutation null (5,000 permutations)

Raw conservative p-values, with Holm step-down adjustment across the
prespecified family (5 models × {MAE, RMSE}). R² is reported only as a
monotone transform of squared error — it is **not** an independent
finding and shares RMSE's null ordering exactly.

| Model | MAE raw / Holm | RMSE raw / Holm |
|---|---|---|
| Gradient Boosting | .0648 / .6419 | .0642 / .6419 |
| Random Forest | .3305 / 1.0 | .3047 / 1.0 |
| Mean baseline | .6021 / 1.0 | .5819 / 1.0 |
| Linear Regression | .7686 / 1.0 | .7566 / 1.0 |
| Median baseline | 1.0 / 1.0 | 1.0 / 1.0 |

**Multiplicity statement:** multiple models and metrics were inspected
in this exploratory screen, so the family above is the full inspected
set and the Holm column is the reported inference. Nothing approaches
significance under either view. All permutation results are labeled
descriptive/exploratory for a single-cohort n = 31 design and remain
hypothesis-generating regardless.

## Conclusion

1. **Negative:** the prespecified resting-ECoG feature set did not
   establish practically useful out-of-subject prediction of concurrent
   tapping rate in this cohort; the best candidate failed the
   prespecified baseline rule and its uncorrected p = .064 does not
   survive the multiplicity correction its own exploratory screen
   requires (.642 adjusted).
2. This is a result about **these 31 recordings and this feature set** —
   not about DBS therapy, not about HFOs as biomarkers, and not a
   validated predictor of anything clinical.
3. Confirmation of any future variant requires a new cohort or an
   untouched evaluation set; tuning on these 31 subjects and reporting
   the same subjects would be confirmatory only in name.
4. Reasonable post-hoc sensitivity work (LOSO/jackknife influence,
   artifact/QC sensitivity, acquisition covariates, alternative
   spectral-power features, longer windows) is documented as
   exploratory follow-up in `docs/limitations.md`; none of it may be
   presented as confirmatory performance on this cohort.

## Machine-readable outputs

`outputs/kinematic_results_corrected/` (gitignored, local):
`metrics.csv`, `oof_predictions_per_subject.csv` (per-subject outputs
stay outside the repository, per `docs/data_access.md`), and
`summary.json` (full run contract: provenance, protocol context,
bootstrap intervals, corrected permutation summaries, and the
interpretation caveat).

# Adaptive-transfer protocol v2 (frozen analysis contract)

**Stage: exploratory, explicitly separate from v1.0.1.** Machine-readable
contract: `src/curve_icon_dbso/transfer/contract.py` —
`contract_version 2.1.0`, freeze hash
`027503a3d0f8359f89d56b16cb47e05adf73df670924216820f57b7fa433bb20`
(recompute any time with `python -c "from
curve_icon_dbso.transfer.contract import contract_hash; print(contract_hash())"`).
Pinned by `tests/test_transfer_contract.py`; any later scientific change
requires a contract version bump plus an append-only §16 changelog entry.

**Freeze discipline.** This document and the module it mirrors were
written and frozen **before any v2 real-data execution** and before any
v2 feature/target correlation was computed on real data. The Phase-0
feasibility audit (2026-11-19, aggregate facts only) informed the four
design decisions below; those decisions were made by the project owner
on 2026-11-19 and are recorded here verbatim:

| Decision | Owner choice (2026-11-19) |
|---|---|
| Spatial strategy | Grid-relative topology anchored at the motor contact (no MNI projection — none exists) |
| Target strategy | Within-person standardized dynamics (physical-unit cross-subject claims unsupported) |
| Deployment claim | Zero-shot and few-shot as **separate** endpoints |
| Calibration floors | First 10/20/30 % of MOVE; ≥ 8 s calibration, ≥ 55 s test; all 31 recordings eligible |
| Analysis mode (post-review, pre-execution) | **Offline retrospective** — spans processed independently end to end; zero-phase filtering allowed WITHIN a span, documented; NO online/causal-inference claim |

**Relationship to v1.0.1 (immutable).** The v1.0.1 contract
(`curve_icon_dbso.protocol`, hash `5e0e2566…e92a7`), its results
(`docs/results_followon_real.md`), and its provenance are frozen
history. Nothing in v2 re-fits, re-scores, reinterprets, or overwrites
them. v2 asks a **different, deployment-shaped question**: does
spatially aligned neural information transfer to unseen subjects with
and without brief patient calibration, measured against temporal
baselines that v1 showed dominate naive decoders? v1's recorded v1.0.1
verdict (transfer NOT successful; decoder MAE .721 vs persistence
.282) is the *motivation* for v2's stricter baselines and calibration
arms, never a result v2 may revisit.

## 1. Research questions and tracks

| Track | Question | Endpoint family |
|---|---|---|
| A — movement detection | Does a cross-subject decoder detect REST vs MOVE from held-out subjects' sensorimotor ECoG above the null? (Faithful transfer of the cross-patient rest-vs-movement methodology, literature review §9.) | classification, zero-shot by construction |
| B — kinematic decoding | Does ECoG predict movement information **beyond the target's own temporal history**? | regression, zero-shot (scale-free) + few-shot (error-based) |
| C — adaptation | Do zero-shot and few-shot performance differ, and how much calibration does a patient need? | calibration curve, separate claims |
| D — event-level biology | Are single-tap kinematic outcomes associated with (association claim) and predictable from (prediction claim) event-aligned ECoG? | event-level, subject-replicated |

The four pre-registered primary endpoints (§11) form ONE multiplicity
family. This protocol never optimizes toward significance: a negative
result on any or all endpoints is a valid, reportable outcome.

## 2. Cohort flow and exclusion rules (frozen)

Source: the 31 converted per-recording files of the authorized local
copy (aggregate facts, feasibility audit 2026-11-19: 31/31 fs = 1200 Hz
ECoG **and** glove on a shared sample grid; 23 recordings one
8-contact strip, 8 recordings two strips; uniform 5-sensor glove).

Eligibility gates (a recording is excluded, with a machine-readable
reason, if ANY fails):

1. usable MOVE data ≥ 30 s (v1.0.1 §4 floor; usable = non-finite/flat/
   saturated/rule-failing channels removed, never imputed);
2. usable REST data ≥ 5 s (detection needs both classes);
3. motor-anchor position present and within [1, 8];
4. at the 30 % split: untouched test segment ≥ 55 s and calibration
   prefix ≥ 8 s (owner floors — feasibility: 31/31 pass at every
   fraction);
5. Track D additionally requires ≥ 2 contributing channel-blocks under
   the frozen tap floors (v1.0.1 §2), else Track D reports the
   recording as event-ineligible (Tracks A–C unaffected).

The one recording carrying a workbook signal-quality flag of 0 is
**retained** (no defined QC rule it fails) and reported by flag in
per-subject heterogeneity. Exclusions are counted, never silent;
eligibility fractions below 50 % for any track trigger the §13
machinery-failure stopping rule. The cohort-flow artifact records:
eligible → gated-in per track → windows kept → per-subject counts
(aggregate only; per-subject rows stay in gitignored local outputs).

## 3. Primary and secondary hypotheses (frozen)

- **H1 (Track A / C-zero detection).** Grid-relative multiband
  features detect REST vs MOVE in unseen subjects above the
  block-permutation null.
- **H2 (Track B primary).** In the 30 % calibration condition, the
  combined model (kinematic AR + neural residual) reduces MAE relative
  to the kinematic AR baseline on held-out subjects (paired
  delta-MAE < 0).
- **H3 (Track C zero-shot regression).** Neural predictions correlate
  with held-out subjects' movement dynamics (scale-free correlation
  endpoint — §6 unit rationale).
- **H4 (Track C few-shot value).** Performance improves with patient
  calibration over the zero-shot endpoint.
- Secondary hypotheses (separate Holm families): detection AUROC /
  sensitivity / specificity / event-detection rate; regression
  incremental R², innovation correlation, RMSE, per-subject spread;
  ablation contrasts (§12); event-level secondary outcomes (§10).
- **Global null** is declared when none of the four primary endpoints
  meets its criterion. Heterogeneity across subjects is reported
  regardless of aggregate outcome.

## 4. Targets and units (frozen)

- **Detection target:** block-kind label (REST = 0 / MOVE = 1) from
  the recording's segment table. Never presented as continuous-vigor
  decoding.
- **Continuous target (Track B):** the v1-frozen
  `glove_vigor_envelope_z` construction (per QC-OK glove channel:
  0.5–4 Hz band-pass, Hilbert envelope, channel average, 100 ms
  moving-mean smooth, mean per 100 ms update bin, per-recording
  z-score), **reimplemented verbatim in v2 code** and in few-shot
  conditions standardized with **calibration-segment-only** constants
  (§9). **Span-local construction (v2.1.0):** the envelope for each
  span (each calibration prefix; the test interval) is filtered from
  that span's clipped raw glove samples only, and bins within
  `TARGET_EDGE_EXCLUSION_S = 2.0` s of a clipped segment edge are
  excluded (counted, never filled) — the margin covers the 0.5 Hz
  band-edge Hilbert transient, the band-pass startup, and the
  smoothing kernel, so a bin's value depends only on samples strictly
  interior to its own span. No future test sample can alter a
  calibration target; this is pinned by adversarial mutation tests.
  Units: dimensionless (within-person standardized). Physical
  units across subjects: **UNKNOWN — unsupported** (feasibility:
  per-recording movement-period RMS spans ~2.8 orders of magnitude;
  no calibration metadata). No cross-subject amplitude claim is
  permitted under any result.
- **Event-level targets (Track D):** primary = log₁₀ inter-tap
  interval (s; scale-free, cross-subject comparable); secondary family
  = inter-tap-interval CV, within-block amplitude decrement
  (dimensionless per-cycle slope), per-tap amplitude z and peak-velocity
  z (within-person only, never cross-subject magnitudes).

## 5. Spatial representation (frozen; owner decision)

Grid-relative, motor-anchor-anchored topology on the anchored strip:
ordinal positions −4…+4 relative to the anchor contact (anchor = 0);
per-recording common-median rereference (target-independent, v1.0.1 §5
fallback); per-channel QC (v1.0.1 §4 rules; failed channels never
imputed). Per-offset **missingness indicators** plus a per-window
coverage fraction are features; a missing grid position is explicitly
represented, never silently replaced by a cross-channel median. The
second strip of dual-strip recordings is **excluded from the primary
grid** (its anchor assignment is undocumented — feasibility audit
2026-11-19) and counted in the spatial-coverage summary. The anchor
position maps to ordinal position `anchor_position − 1` on strip 1
(P-label convention). No MNI coordinates, native-space transforms, or
per-contact anatomical labels exist in any available metadata; none
are invented, and the coordinate space of the single recorded landmark
triple is **UNKNOWN**. Any data-dependent spatial weighting (none is
prespecified) would be fitted on outer-training subjects only.
Prespecified ablation: adjacent-contact bipolar pairs on the anchored
strip instead of the referential grid (§12).

## 6. Signal processing (frozen)

- Windows: 500 ms (primary) and 1,000 ms (single prespecified ablation
  — not an unrestricted search); 100 ms updates; target-bin center =
  `(bin + 0.5)·0.1 s`; a lagged window ends at `target_time − lag`
  (sample-exclusive half-open `[start, end)`) — the v1 off-by-one
  contract, reused and re-tested.
- Lag stack (regression): {0, 0.1, 0.2, 0.3, 0.4, 0.5} s of *preceding*
  neural context. Detection: the window itself (lag 0). No window ever
  contains the target instant or later samples.
- Detection additionally drops windows within 1.0 s (the maximum
  window) of a REST/MOVE boundary guard — transition dynamics belong
  to neither class. **Enforced in machinery (v2.1.0):** every retained
  feature row's full read range (earliest lagged sample through the
  lag-0 window end) must sit ≥ 1 s inside BOTH edges of its
  span-clipped block; each meta row records
  `window_earliest_sample` AND `window_latest_sample`, and the
  enforced distances are pinned by adversarial tests, not by the
  constant's value.
- Windows never cross REST/MOVE block, calibration, purge, or test
  boundaries; any straddle is dropped with a machine-readable reason.
- **Span-local processing (v2.1.0; offline retrospective — owner
  decision recorded in the header table):** calibration and test
  spans are processed INDEPENDENTLY end to end — per-span ECoG
  rereferencing, line-frequency detection, notch treatment, feature
  builds, and targets — so no sample of one span can alter another
  span's artifacts. Zero-phase filtering remains allowed WITHIN a
  span and is documented as retrospective; this contract makes no
  online or causal-inference claim.
- Bands (cohort fs = 1200 Hz → Nyquist 600 Hz; disjoint partition of
  4–200 Hz): theta 4–8, alpha 8–12, low beta 13–20, high beta 20–35,
  low gamma 35–60, high gamma 60–80, HFA 80–200 Hz. Content above
  200 Hz is excluded a priori (the distributed recording is
  pre-filtered with undocumented upstream filtering — recorded, not
  guessed). Welch PSD (0.5 s Hamming segments, 50 % overlap, the v1
  estimator); features = log₁₀ relative band power per grid cell ×
  band × lag, plus per-cell aperiodic exponent; line noise notched at
  the detected line frequency ± 2 Hz + harmonics (v1.0.1 §4 rule).
- Normalization: causal trailing z-score per cell × band feature
  stream over the past 5 s of updates only (past unlabeled neural
  samples — explicitly allowed zero-shot), clipped at ± 5.
  Cross-subject model-side preprocessing (median imputation +
  standardization) lives inside the sklearn pipeline, fitted on
  outer-training subjects only. Artifact mitigation: v1.0.1 §4
  per-window QC (flatline / saturation / rapid transients / implausible
  band power), failures counted and never imputed.
- The 1,000 ms vs 500 ms comparison is the single prespecified window
  ablation (§12).

## 7. Validation (frozen)

- **Outer unit: subject.** True leave-one-subject-out (31 folds), the
  prespecified v2 change from v1's GroupKFold(5): every recording is
  held out exactly once, features are label-independent and computed
  once, and Ridge/logistic fits on ≤ 30 subjects are computationally
  feasible (verified on synthetic fixtures before any real-data run;
  a timing budget accompanies the run command). Fold machinery
  refuses subject-straddling partitions outright.
- **Inner:** `GroupKFold(3)` over outer-training subjects (inner-LOSO
  under 6 outer-training subjects). All imputation, scaling, spatial
  weights, burst thresholds, and hyperparameter selection are fitted
  on inner-training subjects only. Selection: inner mean balanced
  accuracy (detection) / inner mean MAE (regression); ties resolve to
  the smaller frozen grid index. Selection never sees outer-test data.
- Pooled windows are never treated as independent subjects: every
  headline number is subject-level first (median / IQR across
  subjects); pooled-window values are secondary descriptives with the
  explicit non-independence caveat.

## 8. Models and fixed grids (frozen)

- Detection — primary: **class-weighted Ridge logistic regression**
  (alpha grid {1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}; 7 points;
  inner-selected). Comparator: fixed shallow XGBoost (100 trees,
  depth 2, lr 0.05, n_jobs 1, seed 42 — identical inputs, no tuning).
  No deep learning is added under this contract.
- Regression — primary: **lagged Ridge / Wiener** over the frozen lag
  stack (same 7-point alpha grid, inner-selected). Fixed comparators:
  neural-only Ridge (identical features → target directly), kinematic
  autoregressive baseline (least-squares AR, order 10 bins = 1.0 s of
  standardized-target history, fitted on outer-training subjects
  only), combined AR + neural residual (the Track B primary model),
  and the fixed shallow XGBoost on exactly the same leakage-safe
  inputs. Few-shot subject-specific fits: a **calibration-selected
  blend** (v2.0.1, pre-execution) — the transferred zero-shot residual
  model gives the base correction, and the calibration segment chooses
  via chronological blocked CV **inside the calibration segment only**
  among {pure kinematic AR, transferred residual only, transferred
  residual + anchored calibration correction (a Ridge on the
  residual-of-the-residual with blocked-CV-selected alpha), full
  calibration correction}. The final test interval is never touched,
  and the zero-shot-selected alpha is deliberately not reused — a value
  chosen for a different task and cohort over-shrinks the tiny
  calibration fit. The detection analog refits the logistic with the
  selected C fixed.

## 9. Zero-shot vs few-shot validation (frozen; separate claims)

- **Zero-shot.** No labeled data from the held-out subject may
  influence fitting, tuning, normalization parameters, spatial
  learning, or threshold selection. Causal trailing normalization uses
  past unlabeled neural samples only (explicitly allowed). Scoring:
  detection uses balanced accuracy (scale-free); regression uses
  per-subject correlation between predicted and observed streams
  (scale-free). **Rationale (frozen):** per-recording target
  standardization fitted on the final test interval is a leakage
  violation, and zero-shot provides no legitimate scale constants for
  the held-out subject — so error-magnitude claims are few-shot
  endpoints. This limitation is stated in every zero-shot regression
  result.
- **Few-shot.** Calibration conditions = chronological first
  10 % / 20 % / 30 % of each recording's eligible MOVE data (nested
  prefixes; ≤ 8 s floor). The calibration span runs from recording
  start through the end of the selected MOVE prefix (interleaved
  REST blocks inside the span are legitimately labeled calibration
  data); the **final test interval is identical across conditions**:
  everything after the 30 % prefix end plus the purge. A purge gap of
  **2.0 s** (conservative prespecified value; the minimum is max
  window 1.0 s + max lag 0.5 s = 1.5 s) separates calibration from
  test. Subject-specific parameters fitted **only** from the
  calibration segment: target scale constants (mean/SD of the
  envelope over calibration MOVE samples), the calibration-block
  means, the Ridge/logistic refit (calibration-selected blend, §8,
  v2.0.1).
  The best calibration fraction is **never** selected using final-test
  performance; all three conditions are reported together as the
  calibration curve (performance, usable calibration minutes, and
  gain over the corresponding zero-shot endpoint per condition).
- Windows are constructed within spans only; the dataset builder drops
  with a recorded reason any window crossing a span, block, purge, or
  QC boundary.
- **Single-class calibration segments (machinery note, §2 skip
  discipline).** A recording whose calibration prefix contains only one
  block kind (e.g. the recording begins inside a MOVE block, so the
  10 %/20 % prefixes hold no REST rows) cannot support the binary
  detection few-shot refit for that fraction: the subject-fraction
  point is skipped with the machine-readable reason
  `calibration_segment_single_class`, the skip is counted in the
  calibration curve (`n_single_class_calibration_skips`), and the
  zero-shot endpoints are unaffected. The regression track is
  unaffected (Ridge has no class requirement). No threshold, target,
  or model changes.

## 10. Track D — event-level analysis (frozen; enabled)

Tap events are extractable without outcome-dependent manual editing
(feasibility: the frozen cycle-detection rule yields 762–2,706 cycles
per recording). Events are keyed by the frozen detection rule (0.5–4
Hz band-pass, 0.15 s minimum separation, orientation by larger peak
count, ≥ 5 cycles per contributing channel-block); a recording enters
Track D only with ≥ 2 contributing channel-blocks. Per-tap outcomes:
**primary = log₁₀ inter-tap interval**; secondary family = inter-tap
interval CV, within-block amplitude decrement, per-tap amplitude z,
peak velocity z (the latter two within-person only). Analysis:

- **Association claim:** two-stage subject-replicated model — per
  subject, the within-subject Spearman association between the event
  outcome and the anchor-cell HFA log-power feature at an
  event-aligned pre-onset window (−0.5 s to onset, causal); across
  subjects, Fisher-z aggregation with subject-cluster bootstrap CI and
  a within-subject block-permutation null. Subject — not tap — is the
  independent replication unit; the number of taps never increases the
  subject sample size. Holm within the event family.
- **Prediction claim:** held-out-subject validation — Ridge on
  event-aligned features versus an event-level AR(1) baseline (AR(1)
  resets at every block×channel boundary — taps from different sensor
  channels are distinct series), paired delta-MAE with subject-cluster
  bootstrap CI and shift null (mirrors Track B).
- Association and prediction are reported as **separate** claims;
  neither licenses causal or clinical interpretation.
- **One event p-value family (v2.1.0):** the log-ITI, tap-amplitude-z,
  and peak-velocity-z association nulls plus the event-prediction
  shift null form a single Holm family, adjusted exactly once. The
  per-tap z outcomes are analyzed through these association claims —
  the mean of within-recording z-scores (≈ 0 by construction) is never
  an endpoint. ITI CV and within-block amplitude decrement are
  block-level descriptives with no prespecified per-tap association
  test: no p-value, no family membership, reported as medians.

## 11. Metrics, baselines, promotion (frozen)

- Detection: primary = **median subject-level balanced accuracy**;
  secondary = AUROC, sensitivity, specificity, event-detection rate
  (an event counts as detected when ≥ 3 consecutive positive updates
  — 300 ms — fall within a MOVE block; fixed before execution).
  Baselines: majority class per outer fold.
- Regression: primary = paired per-subject delta-MAE
  (combined − kinematic AR) in the 30 % condition; zero-shot primary =
  median per-subject Pearson r. Secondary: incremental R² (squared
  multiple correlation of residuals on neural predictions beyond the
  AR baseline), predicted-vs-observed innovation correlation, RMSE,
  per-subject performance distribution. Raw correlation with the
  smooth target **cannot** establish success by itself — the AR
  baseline comparison is mandatory for every claim.
- A model is discussed as superior only under the §12 criteria on
  identical held-out subjects with uncertainty and corrected nulls.

## 12. Planned ablations (frozen; all reported, none selected post hoc)

1. Window 1,000 ms vs 500 ms (single prespecified comparison);
2. bipolar vs referential spatial representation;
3. lag-stack subsets {0–0.2 s} vs the full {0–0.5 s} stack;
4. band groups: low {theta…high beta} vs high {low gamma…HFA};
5. Track B model decomposition: neural-only vs combined vs
   kinematic-only on identical folds;
6. DMD, contrastive learning, and foundation models: **not part of
   this contract** (deferral rationale recorded in the owner brief;
   a future DMD experiment must be event-aligned under a new contract).

## 13. Uncertainty, nulls, multiplicity (frozen)

- **Subject-cluster bootstrap:** 2,000 resamples, seed 42, 95 %
  percentile CIs; subjects resampled with all their windows; applied
  to every headline statistic (medians re-computed per resample).
  Percentile intervals undercover at pilot n and describe this cohort.
- **Detection null:** within-held-out-subject **block-kind label
  permutation** — the block-kind assignment of the held-out
  recording's blocks is permuted (balance- and autocorrelation-
  preserving; the trained decoder is untouched), 1,000 draws, seed 42,
  conservative p = (1 + #{null ≥ observed}) / (1 + n). Majority-class
  baseline reported alongside.
- **Regression nulls (v1.0.1-corrected conventions, reused):**
  circular shifts of the standardized target within each recording
  (neural predictions unchanged, persistence recomputed from the
  shifted stream; effective circular displacement
  `min(shift, span − shift)` ≥ 2 s; blocks shorter than twice the
  minimum displacement excluded and counted) and blockwise 2 s segment
  shifts as the secondary null; ≥ 1,000 shifts each; independent row
  shuffling never used (asserted in code).
- **Multiplicity:** the four primary endpoints form one family with a
  Holm correction; each secondary family is Holm-corrected within
  itself. All p-values are descriptive/exploratory at this cohort size
  and carry the permutation caveat.
- **Single adjustment + final-derived verdicts (v2.1.0, enforced):**
  each endpoint contributes its RAW null p-value to the primary family
  exactly once — the detection model-family Holm over
  {Ridge, XGBoost} is descriptive only and never an input. Every
  displayed verdict Boolean = criterion + bootstrap CI + FINAL
  Holm-adjusted p < 0.05, so per-track verdicts can never contradict
  `global_null`. Secondary families implemented and Holm'd within
  themselves: detection (AUROC / sensitivity / specificity /
  event-detection rate, from the same block-permutation draws),
  regression (incremental R² / innovation r / RMSE / per-subject
  spread, from the same circular-shift draws), and the ablation
  contrasts (paired per-subject, two-sided sign-flip, Holm across the
  family). Track D uses ONE event p-value family (log-ITI, amplitude-z
  and velocity-z associations + event-prediction shift null) adjusted
  once; the two block-level outcomes (ITI CV, within-block amplitude
  decrement) have no prespecified per-tap association test, carry no
  p-value, and are reported descriptively. The mean of
  within-recording z-scores is never an endpoint (≈ 0 by
  construction).
- **Machinery stopping rule enforced (v2.1.0):** a track with fewer
  than 50 % of loaded recordings passing its gates is reported
  not-executable and its endpoints are EXCLUDED from the primary Holm
  family (counted, never silently failed).
- **Definitive-run gate (v2.1.0):** only `--definitive` driver runs —
  exact frozen null draws, ablations on, comparators on — are named
  `run_…` and may write the public results document; every other
  setting combination is a `smoke_…` run regardless of draw count and
  can never touch `docs/results_adaptive_transfer_real.md`.

## 14. Stopping rules and prohibited changes (frozen)

- A track with fewer than 50 % of eligible recordings passing its data
  gates is reported **not-executable** — no substitution of models,
  bands, targets, or thresholds.
- No track stops early for performance reasons; the complete frozen
  matrix runs and is reported regardless of interim results.
- No result-driven stopping, no post-hoc target/band/model/channel/
  lag/threshold/spatial changes, no calibration-fraction selection on
  final-test performance, no presentation of classification as vigor
  decoding, no pooled-window independence claims, no unchanged v1-DMD
  rerun, no deep learning before Tracks A–C are complete and the
  protocol amended, no physical-unit cross-subject claims, no merged
  zero/few-shot claim. Post-hoc work, if any, is labeled `post_hoc`,
  reported separately, and never eligible for any §12 claim.

## 15. Reporting and privacy (frozen)

`docs/results_adaptive_transfer_real.md` plus an aggregate
machine-readable summary, cohort-flow artifact, per-track baseline
comparisons, calibration curves, subject-cluster CIs, corrected
permutation results, spatial-coverage summary, and an explicit
success/failure verdict for every frozen criterion — distinguishing
detection from regression, zero-shot from few-shot, neural-only from
kinematic-only from combined, association from prediction, exploratory
from confirmatory, statistical evidence from practical usefulness, in
language readable by a technical recruiter or engineer. No subject
results, coordinates, filenames, or identifiable metadata are ever
committed. Raw and derived data stay in gitignored local directories.
Provenance per run: git SHA, lock hash, contract version + hash,
input manifests (verified against the approved dataset, never
committed), deterministic seeds, stage label.

## 16. Versioning and changelog (append-only)

| Version | Date | Change | Hash at freeze |
|---|---|---|---|
| 2.0.0 | 2026-11-19 | Initial freeze (owner decisions on spatial, target, endpoints, calibration floors recorded above; Phase-0 feasibility audit the same day). No v2 real-data execution has occurred. | `2d6becb33d5834c2ff4e67f817856da9342ca55bd4736211c0629d938adac02b` |
| 2.0.1 | 2026-11-19 | **Pre-execution refinement** (before any v2 real-data run): the few-shot subject-specific mechanism changes from a from-scratch calibration refit to a **calibration-selected blend** — synthetic verification fixtures measured that a from-scratch refit (~a few hundred rows against ~440 features) can only select maximal shrinkage and that the transferred zero-shot residual forecast alone already reduces held-out MAE, so the calibration segment now chooses among {pure AR, transferred residual, transferred + anchored correction, full correction} by chronological blocked CV inside the calibration segment. No target, band, model family, grid, floor, fold, null, or criterion changed. | `e5f47ee16c7424a8c4bf425eb3c123170b0326fae1b31fd0d8f99628f8353537` |
| 2.1.0 | 2026-11-19 | **Pre-execution integrity refinement** (review findings; still before any v2 real-data run). Owner decision recorded: **offline retrospective** — spans processed independently end to end, zero-phase filtering within a span documented, no online claim. (1) Calibration/test targets and features are built SPAN-LOCALLY from clipped spans with `TARGET_EDGE_EXCLUSION_S = 2.0` target-edge exclusions — no future test sample can alter a calibration artifact (pinned by adversarial mutation tests). (2) The frozen 1 s REST/MOVE boundary guard is now ENFORCED in machinery with recorded earliest/latest samples per row. (3) Multiplicity repaired: the primary family receives each endpoint's RAW p exactly once (no double Holm; the detection model-family Holm is descriptive), every displayed verdict derives from the FINAL adjusted result, secondary families (detection metrics, regression metrics, ablation contrasts) are implemented and Holm'd within themselves, Track D uses one event p-value family, event association requires its permutation null, event prediction requires its bootstrap CI. (4) AR/persistence histories reset at MOVE-block boundaries; the persistence null scores the persistence predictions (fixed). (5) The < 50 % machinery stopping rule is enforced (not-executable tracks' endpoints are excluded from the family, counted). (6) Track D drops mean-of-z endpoints (≈ 0 by construction) in favor of per-tap z association claims. (7) Driver gains an explicit `--definitive` gate (exact draws + ablations + comparators; non-definitive runs are `smoke_…` and can never write the public results document), collision-proof run directories, and `--no-xgboost` reaches Track A. (8) Committable summaries carry counts only; ordinal-position lists are local-only. No band, model family, grid, floor, fold, or criterion value changed. | `027503a3d0f8359f89d56b16cb47e05adf73df670924216820f57b7fa433bb20` |

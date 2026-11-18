# Follow-on neural-decoding study — sanitized aggregate report

**Written for:** reviewers and future readers of this repository
(aggregates only; no subject codes, no per-subject values, no
redistributed data).

**Stage: exploratory.** Nothing here is a confirmatory or clinical
claim. This study **predicts concurrent finger-tapping behavior from
intracranial ECoG in a single session** — it does **not** predict DBS
response, treatment outcome, or clinical benefit, and no such claim is
made or implied. The recorded first authorized run and all earlier
provenance are preserved untouched (separate versioned output
directories; the frozen five-feature reference below was re-computed on
identical folds and reproduces that run exactly).

> **Corrective status (2026-11-17).** A post-run review found the
> v1.0.0 integration only partially covered the frozen contract: MOVE
> baseline MAEs were never emitted (so the §12 transfer criterion's
> error half was never evaluated), the circular-shift null could draw
> nominal shifts whose **effective** circular displacement was a single
> 100 ms update bin (its reported p = 0.001 is **withdrawn**), the
> `delta_combined` block was never built, the Holm family ran over 8 of
> its 10 members, and the DMD feature/diagnostic implementation had a
> conjugate-projection defect and a basis mismatch. The v1.0.0 sections
> below are retained with their defective claims marked
> **[superseded]** or **[withdrawn]**; the corrective **v1.0.1 rerun**
> (identical inputs — SHA-256 manifests verified equal) recomputed every
> affected analysis under contract v1.0.1
> (`protocol_hash 5e0e2566…e92a7`) and its section at the end of this
> document carries the definitive frozen-contract verdicts.

## v1.0.0 run (2026-11-16) — headline, with corrective annotations

1. **Subject-level arms (REST / DELTA):** no expanded feature block ×
   model combination met the frozen promotion rule (improve BOTH MAE
   and RMSE vs BOTH naive baselines AND the frozen five-feature
   reference on identical outer folds) in any block — this half stands.
   The formal "global null" declaration, however, required the
   state-change test on the `delta_combined` block, which **was never
   built under v1.0.0** [superseded — evaluated under v1.0.1].
2. **MOVE arm:** the decoders' held-out performance (median
   across-subject window-level Pearson r ≈ 0.06) is dwarfed by the
   naive temporal references — the one-update persistence baseline
   alone reaches r = 0.92 — so the correlation half of the transfer
   criterion strongly disfavored transfer. But the criterion's MAE
   comparison was **never computed** (baselines emitted correlations
   only), and the shift-null p = 0.001 is withdrawn (effective-
   displacement defect). The v1.0.0 sentence "methodology transfer NOT
   successful" was directionally supported but formally unevaluable
   [superseded — evaluated in code under v1.0.1].
3. **DMD representation:** DMD did not improve correlation over
   conventional band power (nodes r = 0.010 vs power r = 0.063);
   DMD and combined showed marginally LOWER median MAE (0.7199/0.7198
   vs 0.7211) — an untested difference, with no paired uncertainty or
   autocorrelation-preserving representation-difference null run. The
   v1.0.0 phrasing "does not outperform conventional band power" is
   defensible only on correlation; the blanket "worse" reading is
   corrected here, and the reconstruction diagnostic behind the
   "physiology is far from low-rank linear" claim was defective
   [superseded — recomputed under v1.0.1].

## Cohort and recording flow

- 31 recordings / 31 subjects entered the subject-level arms
  (31/31 passed the frozen tap-rate QC; one subject-level feature frame
  per recording; no exclusions).
- MOVE arm: 29 of 31 recordings yielded usable windows; 2 recordings
  were excluded explicitly (zero windows survived block-boundary and
  QC filtering — recorded with machine-readable reasons in the run's
  QC block; nothing imputed).
- 38,078 neural windows built (500 ms causal windows, 100 ms updates,
  frozen lag stack); 1,290 additional candidate windows dropped by the
  frozen causal rules (block-boundary/edge or QC reasons — counts in
  the run's `qc` record).

## Signal and target QC counts

- Target QC (frozen tap floors): 31/31 recordings pass.
- Envelope target (`glove_vigor_envelope_z`): ok for 29/29 MOVE-arm
  recordings.
- Window QC: 1,290 candidate windows dropped
  (`window_outside_block` / `insufficient_ok_channels` — the causal
  edge rules), plus the 2 excluded recordings above.
- Subject-level feature extraction: 71 feature columns per recording;
  failed channels/blocks are counted upstream and never imputed
  (extraction QC carried in the gitignored per-recording outputs).

## Contracts

Exact frozen definitions: `docs/followon_protocol.md` and the
machine-readable `curve_icon_dbso.protocol` (v1.0.0 at this run's
provenance, hash below; the corrective v1.0.1 revision of 2026-11-17
completes implementation coverage — see its §14 changelog entry).
REST = resting features → subject-level tapping phenotypes; MOVE =
time-resolved ECoG → concurrent glove vigor envelope
(500 ms/100 ms windows, lag stack ≤ 500 ms); DELTA = movement-minus-rest
modulation → subject-level phenotypes (log-ratio power, signed
dynamics differences).

## Models and baselines

- Subject-level arms: dummy mean/median baselines; Ridge (primary,
  frozen 7-point alpha grid), linear SVR (C ∈ {0.1, 1, 10} ×
  ε ∈ {0.1, 0.3}·training-MAD), Bayesian Ridge, fixed Matérn-3/2 GP
  (bounds fixed — nothing tuned), fixed shallow XGBoost (optional,
  DD-003). All hyperparameters selected by inner subject-grouped CV
  (GroupKFold(3)) on outer-training subjects only; chosen values
  recorded per fold.
- MOVE arm: lagged Ridge/Wiener (primary), linear SVR, shallow XGBoost;
  baselines = fold-global mean, held-out recording's own block-mean
  (naive-target reference, labeled as such), persistence (one-update
  lag).
- Five-feature reference: the frozen `curve_icon_dbso.kinematics`
  implementation with the original corrected model set, on the
  identical outer folds (reproduces the recorded run: e.g. GB
  MAE 0.523 / RMSE 0.650; LinearRegression 0.765 / 1.163).

## Nested evaluation design

Outer: subject-grouped — LOSO ≤ 12 subjects, else GroupKFold(5); here
31 subjects → 5 folds, every subject's rows/windows in exactly one
fold (group integrity asserted everywhere). Inner: GroupKFold(3)
(inner-LOSO < 6 subjects) on outer-training data only; criterion
inner-CV mean MAE; ties → smaller frozen grid index. Metrics:
per-held-out-subject first, then median/IQR across subjects; pooled
window values only as flagged secondary descriptives.

```
outer folds (subject-grouped, identical across blocks/models)
 ├─ fold k: train subjects ──► inner GroupKFold(3) ──► pick frozen-grid point
 │                        └─► fit (impute+scale+model) on train only
 └─ held-out subject(s) ──► aligned OOF predictions ──► per-subject metrics
                                                          └─► median/IQR
```

## Subject-level results (primary target; MAE/RMSE, identical folds)

| Block | Baseline mean | Baseline median | Ridge (primary) | Best model in block |
|---|---|---|---|---|
| clinical | 0.559 / 0.680 | 0.484 / 0.684 | 0.530 / 0.650 | Ridge 0.530 / 0.650 |
| rest_spectral | 0.559 / 0.680 | 0.484 / 0.684 | 0.578 / 0.698 | no single winner: GP best MAE 0.577 / 0.710; Ridge best RMSE 0.578 / 0.698 |
| rest_dynamics | 0.559 / 0.680 | 0.484 / 0.684 | 0.567 / 0.684 | BayesianRidge 0.562 / 0.681 |
| delta_spectral | 0.559 / 0.680 | 0.484 / 0.684 | 0.561 / 0.681 | BayesianRidge 0.560 / 0.680 |
| delta_dynamics | 0.559 / 0.680 | 0.484 / 0.684 | 0.672 / 0.863 | BayesianRidge 0.559 / 0.680 |
| combined | 0.559 / 0.680 | 0.484 / 0.684 | 0.571 / 0.698 | GP 0.561 / 0.687 |

(Correction, 2026-11-17: the original table listed
"BayesianRidge 0.559 / 0.680" for `rest_spectral` — a copy of the
dummy-mean baseline values; the actual aggregate is as above, and
Bayesian Ridge's real rest_spectral row is 0.694 / 1.059. The
correction does not alter the zero-promotion conclusion: 0.694/1.059 is
strictly worse than both baselines on MAE.)

No model improves BOTH error metrics versus BOTH baselines in any
block → **zero promotions** (the `beats_reference_*` flags that are
individually true reflect only the weak five-feature LinearRegression
reference, not the rule). R² negative or ≈ 0 everywhere; several
correlations are negative. Full tables (including negative results for
every model × block) in the gitignored run directory.

## Uncertainty and null procedures

- Subject-cluster bootstrap (2,000 resamples, seed 42): combined-block
  Ridge MAE 0.571 [0.432, 0.711], RMSE 0.698 [0.558, 0.824] —
  intervals spanning the baselines; percentile intervals under-cover at
  this n and describe the cohort only.
- Corrected subject-level label-permutation null through the COMPLETE
  nested pipeline (inner selection included; each permutation scored
  against its own permuted targets; 1,000 permutations, seed 42),
  combined block. **Multiplicity correction (2026-11-17):** the v1.0.0
  run executed 4 of the 5 prespecified family models (XGBoost was run
  in the arm but omitted from the permutation family), so the reported
  "smallest adjusted ≈ 0.83" covered only 8 of the 10 family members.
  With the two absent XGBoost members entered conservatively at p = 1,
  the smallest Holm-adjusted value over the full ten-test family is
  **1.0**. Smallest raw p remains 0.104 (SVR MAE) among the executed
  eight. The complete family — XGBoost executed, raw and adjusted
  values in a machine-readable artifact — is part of the v1.0.1
  corrective rerun. All p-values descriptive.
- MOVE autocorrelation-preserving null [**withdrawn**]: the v1.0.0
  sampler drew nominal shifts in [2 s, span) whose **effective**
  circular displacement min(shift, span − shift) could be a single
  100 ms update bin — so the statement that every shift was at least
  two seconds was false, and the reported p = 0.001 (observed median
  r = 0.063 vs null median 0.013, 1,000 shifts) is withdrawn pending
  the corrected null (primary AND blockwise), recomputed under v1.0.1.
  Even after correction, an above-null result would show only that the
  observed alignment exceeded one prespecified shifted-target null —
  shared task timing, nonstationarity, and residual artifacts remain
  possible explanations, and it would not establish incremental
  predictive value over the naive temporal references.

## MOVE decoding and methodology transfer

| Decoder | median r (IQR) | median MAE | pooled MAE (secondary, non-independent) |
|---|---|---|---|
| lagged Ridge (primary) | 0.063 (0.122) | 0.721 | 0.715 |
| linear SVR | 0.064 (0.100) | 0.722 | 0.720 |
| shallow XGBoost | 0.050 (0.107) | 0.726 | 0.719 |
| persistence baseline | **0.919** | not emitted under v1.0.0 | — |
| block-mean baseline (naive) | **0.335** | not emitted under v1.0.0 | — |
| fold-global mean | 0.000 | not emitted under v1.0.0 | — |

Subject-cluster bootstrap of the primary's median r (2,000, seed 42):
0.063 [0.036, 0.109] — independently reproduced from the per-subject
table, but under v1.0.0 saved only in prose; the v1.0.1 run stores the
draws, seed, statistic, and endpoints as a machine-readable artifact.

**Transfer verdict [superseded].** The v1.0.0 text declared "transfer
NOT successful — criterion (b) failed by a wide margin", but criterion
(b) (median MAE better than persistence and block-mean baselines) was
**never computed**: no baseline MAE was emitted, only correlations. The
correlation comparison (persistence r = 0.919 vs every decoder ≤ 0.064)
strongly disfavors transfer, and the corrected v1.0.1 run evaluates
criterion (b) in code from emitted baseline MAEs — its section below
carries the formal Boolean. The withdrawn p = 0.001 claim is likewise
restated there from the corrected nulls.

## DMD versus conventional spectral features

| Representation | median r | median MAE |
|---|---|---|
| multiband power + Ridge | 0.063 | 0.721 |
| DMD spatial nodes + Ridge | 0.010 | 0.720 |
| combined (power + DMD) | 0.047 | 0.720 |

DMD node features (frozen rank 8) did not improve correlation over
band power on held-out subjects (0.010 vs 0.063), and combining them
dilutes correlation (0.047). The MAE ordering is reversed and
negligibly small (DMD 0.7199 and combined 0.7198 vs power 0.7211) —
no paired uncertainty or autocorrelation-preserving
representation-difference null was run, so this table supports only
the correlation statement, not an error-metric superiority claim in
either direction. Per-window compute ≈ 0.28 ms.

**Reconstruction diagnostic [superseded].** The v1.0.0 text read the
mean rank-8 relative reconstruction error ≈ 3.2e3 as "physiological
windows are far from exactly low-rank linear systems at rank 8". That
number came from a defective path: when any eigenvalue magnitude
exceeded 1.05, the diagnostic replayed a single one-step prediction
across the whole window instead of the modal evolution, and the
stabilized amplitudes used a plain transpose instead of the conjugate
(the basis was also mixed between raw Φ and orthonormal Q). The ≈3.2e3
value therefore primarily reflects numerical instability and the
fallback path — it is **not evidence about the physiology**, and the
claim is withdrawn. The v1.0.1 run repairs the projection, keeps
features and amplitudes in one basis, and reports an always-finite
one-step error plus a multi-step error only for stable spectra.

## Leave-one-subject influence

Combined-block Ridge: dropping any single subject moves aggregate MAE
by at most ±0.025 (range −0.024…+0.019) — no dominant subject; the
null is not driven by one outlier. Coefficient signs are fold-stable
(≥ 0.8 sign agreement for most REST-spectral features in standardized
space) — the models are stable but uninformative.

## Interpretation matrix (frozen §12) — v1.0.0 status

| Outcome | v1.0.0 status |
|---|---|
| Resting-trait evidence | **None** — no block meets the promotion rule; nothing survives the (incomplete) Holm family. Stands. |
| Movement-state transfer | **Not fully evaluable under v1.0.0** — criterion (a) met (r > 0, CI excluding 0); criterion (b) never computed (no baseline MAE emitted); correlation comparisons strongly disfavor transfer. Evaluated in code under v1.0.1. |
| State-change evidence (DELTA > REST) | **Not fully tested under v1.0.0** — the separate `delta_spectral`/`delta_dynamics` blocks did not promote, but the frozen criterion runs on the `delta_combined` block, which was never built. Evaluated under v1.0.1. |
| Global null | **Cannot be declared under v1.0.0** — the matrix needs all three outcomes (transfer Boolean, state-change test) that v1.0.0 never computed. Declared — either way — under v1.0.1. |

## Limitations, multiplicity, external validation

- n = 31 subjects; percentile bootstrap intervals under-cover; every
  p-value is descriptive/hypothesis-generating.
- Multiplicity: the prespecified Holm family (primary target × 5
  models × {MAE, RMSE} per arm) was defined before the run; the v1.0.0
  execution covered 8 of its 10 members (see the correction above).
  No post-hoc selection of any block/model occurred; no SHAP anywhere.
- **Recorded omission:** the frozen secondary targets
  (`tap_interval_cv`, `tap_vigor_index`, `tap_sequence_slope`) were
  extracted but never modeled under v1.0.0 or the v1.0.1 corrective
  rerun. They are deferred, not analyzed: "full prespecified depth"
  applies to the primary-target inference only, and any future
  secondary-target analysis is a separate, labeled piece of work with
  its own multiplicity control — never an addition to the completed
  primary inference.
- The MOVE negative result is specific to this envelope definition and
  protocol; persistence's correlation dominance shows the behavioral
  signal is mostly smoothness — a harder target (e.g. vigor
  residualized on its own lag, or higher-rate kinematics) could behave
  differently.
- Two MOVE recordings were excluded for having no usable windows under
  the frozen causal rules — their QC records are retained verbatim.
- Threats to validity: single session, single task, uncalibrated glove
  (targets standardized per recording), possible residual
  state/anesthesia confounds, band power at 1200 Hz fs limits HFO
  interpretation, and the frozen rank-8 DMD may be too restrictive for
  8-channel grids.
- **External-validation sizing (illustrative, not a minimum):** the
  ≈800-subject figure is a simple two-sided correlation-power
  calculation for detecting r = 0.1 at 80 % power — an illustrative
  scale marker, not a minimum sample size for this nested predictive
  pipeline. The "order 10²" MOVE statement was likewise not derived
  from an effect estimate for the prespecified incremental-MAE
  contrast. A replication should be sized on the paired subject-level
  MAE difference versus persistence and its uncertainty, expected
  usable-window duration and exclusion rate, and an
  external-validation margin — with the decisive comparison
  pre-registered against naive temporal baselines, not against zero.

## v1.0.1 corrective rerun (2026-11-18) — definitive frozen-contract verdicts

Rerun under contract v1.0.1 (`protocol_hash 5e0e2566…e92a7`) from
**identical inputs** — the per-file SHA-256 manifests of both v1.0.1
runs are byte-identical to the v1.0.0 manifest (31 recordings; manifest
file digest `22a61ef632ef9d5c…`). Determinism proof: every unchanged
computation reproduces v1.0.0 exactly (all six shared subject-level
blocks, all three MOVE decoders' CV tables). Only the corrected /
previously-missing analyses changed. Run directories:
`outputs/followon_subject_modeling/v1.0.1/`,
`outputs/followon_move_modeling/v1.0.1/` (gitignored; v1.0.0 dirs
preserved untouched).

### Subject-level arms (REST / DELTA)

- Unchanged blocks: identical to v1.0.0 (zero promotions anywhere).
- **`delta_combined` block (new)**: no model promotes — best model GP
  (MAE 0.557 / RMSE 0.678), Ridge 0.919 / 1.653; none beats both naive
  baselines and the reference on both metrics. Honest nuance: SVR,
  Bayesian Ridge, and GP each *beat the REST-combined block on both
  error metrics*, but the frozen state-change criterion ALSO requires
  the promotion rule versus baselines, which none meets → **state-change
  evidence: none** (evaluated in code, per-model components stored in
  `run_summary.json`).
- **Complete ten-test Holm family** (`holm_family_combined.json`, XGBoost
  executed): smallest raw p = 0.1039 (SVR MAE); **smallest Holm-adjusted
  p = 1.0** — the v1.0.0 ".831" figure covered 8 of 10 members and is
  superseded. **Resting-trait evidence: none** (Ridge does not promote;
  Holm p = 1.0 on both metrics).

### MOVE arm — transfer formally evaluated

| Decoder / baseline | median r | median MAE (IQR) |
|---|---|---|
| lagged Ridge (primary) | 0.063 (0.122) | 0.721 (0.113) |
| linear SVR | 0.064 (0.100) | 0.722 (0.107) |
| shallow XGBoost | 0.050 (0.107) | 0.726 (0.101) |
| **persistence baseline** | 0.919 | **0.282** (0.125) |
| **block-mean baseline (naive)** | 0.335 | **0.629** (0.089) |
| fold-global mean | 0.000 | 0.719 (0.100) |

The transfer criterion's missing half is now computed and it is not
close: the decoders' median MAE (0.721) is **2.6× the persistence
baseline** (0.282) and worse than the block-mean baseline (0.629) — the
envelope's own one-step autocorrelation predicts far better than any
neural decoder. Criterion (a) met (median r = 0.063 > 0; subject-cluster
bootstrap [0.036, 0.109] excludes 0 — saved as
`bootstrap_median_r.json` with draws, seed, endpoints, resample count);
criterion (b) failed on both required baselines →
**movement-state transfer: NOT successful — evaluated in code** (all
components and the Boolean stored in `run_summary.json`).

**Corrected nulls** (effective displacement min(shift, span − shift)
≥ 2 s; blocks shorter than twice the minimum excluded and counted — 1
block per draw excluded, 128 shifted; 1,000 draws each, decoder
retrained per draw): circular null p = 0.0020, blockwise (2 s segments)
null p = 0.0020. The observed median r (0.063) exceeds 999 of 1,000
draws under both corrected nulls (null median ≈ 0.013 — essentially
unchanged from the withdrawn v1.0.0 run, whose sampling defect affected
validity, not the distribution's center). Per the standing language:
the observed alignment exceeded the prespecified shifted-target nulls,
but its magnitude was small, it sits far below the naive temporal
references, and it does not establish incremental predictive value —
shared task timing, nonstationarity, and residual artifacts remain
possible explanations.

### DMD representation (repaired features and diagnostics)

| Representation | median r | median MAE |
|---|---|---|
| multiband power + Ridge | 0.063 | 0.721 |
| DMD spatial nodes + Ridge | 0.011 | 0.720 |
| combined (power + DMD) | 0.052 | 0.720 |

Same qualitative picture as v1.0.0 under the repaired conjugate
projection and single-basis magnitudes: DMD did not improve correlation
over band power (0.011 vs 0.063; combined 0.052); the MAE ordering
remains marginally reversed and untested (no paired
representation-difference null was run — only the correlation statement
is supported). **Reconstruction diagnostics (repaired):** the
v1.0.0 unstable-branch fallback NEVER fired on real data — zero windows
had any eigenvalue above the 1.05 stability threshold — so the ≈3.2e3
multi-step number was genuine modal growth, not the replay path: it
reflects exponential compounding of near-unitary fitted eigenvalues
over the ~600-step window and is **not evidence about the linearity of
the physiology**. The interpretable diagnostic is the new one-step
relative error: mean 0.152 (median 0.122) — the rank-8 linear system
predicts one step ahead with ~15 % relative error on average. Per-window
compute ≈ 0.28 ms.

### Interpretation matrix (frozen §12) — COMPLETE, all evaluated in code

| Outcome | Verdict (v1.0.1) |
|---|---|
| Resting-trait evidence | **None** — no promotion; Holm p = 1.0 (complete ten-test family) |
| Movement-state transfer | **NOT successful** — criterion (a) met; criterion (b) failed vs both baselines (in code) |
| State-change evidence | **None** — delta_combined comparators beat REST-combined on errors, but none meets the promotion rule |
| **Global null** | **YES** — all three criteria negative, under the complete frozen matrix (descriptive, exploratory) |

The v1.0.0 headline direction survives the corrective rerun everywhere
it was evaluable; what changed is that every formal verdict is now
computed, machine-readable, and complete.

## Provenance

- Contract: v1.0.0 at this run,
  `protocol_hash ed07c397792073a265019b25491687bf27ae5d2836b24b9a8d9f6625d04565d1`;
  corrective v1.0.1,
  `protocol_hash 5e0e2566f9c5df24d8ac751cceaf96d27a15a781dfb70e90b6d2a0f78e1e92a7`
  (changelog: `docs/followon_protocol.md` §14 — no frozen quantity
  changed).
- Input manifest (31 recording files): SHA-256
  `8b3d6a517132e042b39f03e416a4f5b279d0655c93f10b52d2583f4dcc226aa8`
  (per-file digests in the gitignored `input_manifest.json`; the
  v1.0.1 rerun writes its own manifest and is verified identical).
- Runs: git SHA `493b9a6` (tracked source), lock hash `1f81065a…b64c64`,
  seed 42, stage exploratory; per-subject predictions, fold
  assignments, raw features, fitted models, and any MLflow stores stay
  in gitignored local directories (`outputs/followon_subject_modeling/v1.0.0/`,
  `outputs/followon_move_modeling/v1.0.0_run2/`; the v1.0.1 rerun wrote
  fresh versioned directories — `…/v1.0.1/` for both arms — and never
  overwrites these).
- v1.0.1 identical-inputs proof: per-file SHA-256 manifests of both
  v1.0.1 runs are byte-identical to the v1.0.0 manifest (31 files);
  manifest file digest `22a61ef632ef9d5c…`; machine-readable artifacts
  new under v1.0.1: `holm_family_combined.json` (ten tests, raw +
  Holm), `interpretation_matrix` in the subject `run_summary.json`,
  `blockwise_shift_null.json`, `bootstrap_median_r.json` (draws + seed
  + endpoints), `transfer_criterion_evaluated` (components + Boolean)
  in the MOVE `run_summary.json`, and repaired
  `dmd_diagnostics.json` (one-step / stable-gated multi-step).
- The contrastive addendum (Prompt 7) is implemented and synthetically
  verified but its real-data run and contract-version bump are DEFERRED
  (documented in `docs/followon_protocol.md` §21).

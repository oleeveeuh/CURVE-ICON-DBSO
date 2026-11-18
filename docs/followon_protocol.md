# Follow-on protocol: REST / MOVE / DELTA (frozen analysis contract)

**Stage:** protocol-frozen **exploratory**. Machine-readable contract:
`src/curve_icon_dbso/protocol.py` — `protocol_version 1.0.1`
(corrective revision of 2026-11-17; original freeze 2026-11-16),
current hash
`5e0e2566f9c5df24d8ac751cceaf96d27a15a781dfb70e90b6d2a0f78e1e92a7`;
the v1.0.0 freeze hash
`ed07c397792073a265019b25491687bf27ae5d2836b24b9a8d9f6625d04565d1`
remains recorded history (§14) and is the hash carried by every v1.0.0
run's provenance (recompute the current hash any time with
`python -c "from curve_icon_dbso.protocol import protocol_hash; print(protocol_hash())"`).

This document and the module it mirrors were frozen **before any
follow-on real-data result was inspected** (owner instruction,
[neural_decoding_implementation_prompts.md](neural_decoding_implementation_prompts.md)
Prompt 0, 2026-11-16; motivation and literature basis in
[literature_review.md](literature_review.md)). A pinned-hash test
(`tests/test_protocol_contract.py`) fails on any silent edit of a frozen
quantity — changes require a new contract version and a changelog entry
here (§14).

**Relationship to the recorded first result.** The first authorized
real-data run (the prespecified five-feature kinematic analysis,
[methodology.md](methodology.md)) is immutable history. Nothing in this
protocol re-fits, re-scores, or overwrites it. It appears below only as
a frozen *reference* that follow-on models are compared against on
identical outer folds. Real data stay in gitignored local directories
(`data/`, `outputs/`, `artifacts/`) per
[data_access.md](data_access.md); per-subject outputs are never
committed; tests use synthetic fixtures only.

## 1. Arms

| Arm | Question | Output level | Targets |
|---|---|---|---|
| REST | Does the resting neural state predict tapping phenotypes? | subject | `tap_rate_mean` (primary), secondary kinematics |
| MOVE | Does movement ECoG decode concurrent glove movement in unseen subjects? | window, subject-held-out | `glove_vigor_envelope_z` |
| DELTA | Does movement-minus-rest modulation predict tapping phenotypes better than the resting state alone? | subject | `tap_rate_mean` (primary), secondary kinematics |

The scientific comparison is not which algorithm wins, but whether
neural information is a **resting trait** (REST), a **concurrent state
signal** (MOVE), or a **state-change signal** (DELTA) — see the
interpretation matrix (§12).

## 2. Targets (frozen)

**Primary subject-level target: `tap_rate_mean`** — the already-frozen
`curve_icon_dbso.kinematics` tap-rate extraction, unchanged: per MOVE
block × glove channel, Welch (1 s Hamming, 50 % overlap) dominant peak
in 0.5–4.0 Hz contributing only if it carries ≥ 0.20 of in-band power;
target = median over contributing channel-blocks; QC needs ≥ 2
contributing channel-blocks. This contract does not modify that
implementation or its recorded result.

**Secondary subject-level targets** (frozen definitions; implemented by
the glove-phenotyping phase exactly as written; detection band 0.5–4 Hz
via order-4 zero-phase Butterworth, minimum peak separation 0.15 s,
≥ 5 cycles per contributing channel-block, ≥ 2 contributing
channel-blocks per recording):

| Target | Definition |
|---|---|
| `tap_interval_cv` | SD/mean of successive inter-cycle intervals per contributing channel-block; median across channel-blocks |
| `tap_vigor_index` | RMS(diff(filtered)) / (2π · f_tap · RMS(filtered)) per contributing channel-block — dimensionless, amplitude-independent vigor; median across channel-blocks |
| `tap_sequence_slope` | OLS slope of per-cycle amplitude vs cycle index, normalized by the block's mean amplitude (dimensionless per-cycle decay); median across channel-blocks |

`tap_amplitude` is recorded for QC **only**: glove sensors are
uncalibrated across subjects, so amplitude is within-subject-only and
excluded from all cross-subject modeling.

**MOVE-arm continuous target: `glove_vigor_envelope_z`** — per QC-OK
glove channel: 0.5–4 Hz filtered, Hilbert |envelope|; average across OK
channels; 100 ms moving-mean smooth; mean within each 100 ms update
bin; per-recording z-score across MOVE samples. The standardization is
part of the frozen target definition and uses only that recording's
glove data (no neural data, no fold membership, no identity).

## 3. Neural feature families and exact frequency boundaries

Standard bands, half-open `[low, high)` over Welch PSD bin centers
(the topmost active edge is inclusive; adjacent bands share edges
without double counting):

| Key | Name | Range (Hz) |
|---|---|---|
| `theta` | theta | 4–8 |
| `alpha` | alpha | 8–12 |
| `beta_low` | low beta | 13–20 |
| `beta_high` | high beta | 20–35 |
| `gamma_low` | low gamma | 35–60 |
| `gamma_high` | high gamma | 60–100 (standing choice) |

**Acquisition-driven resolution (deterministic, cohort-uniform).**
`protocol.resolve_spectral_bands(min cohort fs)` activates a band only
when its top edge clears `fs/2 − 10 Hz`; callers pass the *minimum*
sampling rate over all recordings so feature space is identical for
every subject (a below-floor recording is a QC exclusion, never a
partial feature vector). The recorded 90–200 Hz high-gamma alternative
is **never auto-selected**: adopting it requires a new contract version
plus cohort fs ≥ 420 Hz. The HFO block (legacy 80–500, a separate
feature block never summed with band powers) is feasible only when
cohort fs ≥ 1020 Hz and additionally gated by the artifact QC rule.

**Transforms (frozen).** Band power; relative band power;
log10 relative band power (primary spectral feature); aperiodic
exponent + offset (OLS of log10 PSD vs log10 f over the active range —
dependency-free 1/f estimate); aperiodic-adjusted band power (mean
in-band PSD ÷ aperiodic fit predicted at band center).

**Dynamics (frozen).** Hjorth mobility and complexity; line length per
second; spectral entropy (active range); beta bursts: envelope
threshold at the recording's own REST-block 75th percentile (reference
data only — no target, no fold information), occupancy, mean duration
(minimum 50 ms), median excess ratio.

**DELTA rules.** Power-type features: log10(MOVE) − log10(REST)
(dimensionless log ratio). Dynamics features: signed native-unit
difference. MOVE/REST features are each averaged over their blocks of
the same recording before the difference.

## 4. Artifact / QC rules (frozen)

Machine-readable per-channel/window reasons; failed channels and
windows are counted and **never imputed, zero-filled, or silently
dropped**.

| Rule | Threshold |
|---|---|
| non-finite samples | immediate fail |
| flatline | window max − min ≤ 1e-12 |
| saturation | > 5 % of samples at the recording's max/min value |
| rapid transients | fraction of \|diff\| > 20 × MAD(diff, recording) exceeding 0.005 |
| implausible band power | window band-power z vs the recording's own REST-window distribution > 10 |
| line noise | notch at the detected line frequency ± 2 Hz + harmonics to Nyquist; detection = larger PSD maximum within ± 2 Hz of 50 or 60 Hz; within 3 dB → QC note `line_freq_ambiguous`, larger wins |
| target QC | the frozen tap floors (§2) |
| minimum usable data | REST arm ≥ 5 s; MOVE arm ≥ 30 s usable MOVE data |

## 5. Channel / anatomical aggregation (frozen)

Bipolar rereferencing only when an electrode-adjacency mapping exists
for the cohort; documented fallback: common median reference across
QC-OK channels, per recording (target-independent). With a
channel→region mapping: per-region (PM, M1, S1, other/unknown) MEDIAN
over QC-OK channels. Without one: MEDIAN over all QC-OK channels plus
the OK-channel count carried as an explicit covariate. Mapping
availability is UNKNOWN until localization metadata is inspected — the
fallback is the standing default; nothing is guessed. Feature blocks
are cohort-uniform: active for the whole cohort or excluded entirely
(reported), never partially computed.

## 6. Windowing — MOVE arm (frozen)

500 ms windows; 100 ms update; lag stack {0, 0.1, 0.2, 0.3, 0.4, 0.5} s
of *preceding* neural context (offsets back from the target sample; the
decoder sees only neural windows ending at or before the target time —
never the future). Windows never cross REST/MOVE block boundaries or
QC-invalid intervals.

## 7. Validation (frozen) — the outer evaluation unit is always subject

- **Outer:** LOSO for ≤ 12 subjects, else `GroupKFold(5)` — identical to
  the implemented `curve_icon_dbso.evaluation` splitter; identical outer
  folds across every model, feature block, and arm. No subject's windows
  appear in more than one outer fold (MOVE windows are subject-keyed and
  structurally cannot straddle folds).
- **Inner:** `GroupKFold(3)` over outer-training subjects (inner-LOSO
  when outer-training has < 6 subjects). All imputation, scaling,
  reduction, burst thresholds, and hyperparameter selection are fitted
  on inner-training subjects only.
- **Selection criterion:** inner-CV mean MAE; ties resolve to the
  smaller frozen grid index.

## 8. Models and bounded hyperparameter grids (frozen)

Ceiling: ≤ 7 grid combinations per model. Selection never uses
outer-test performance.

**Subject-level arms (REST / DELTA).** Baselines: `DummyRegressor` mean
and median through the identical folds. Primary: **Ridge**,
alpha ∈ {1e-3 … 1000} (7-point log grid). Comparators: linear SVR
(C ∈ {0.1, 1, 10} × epsilon ∈ {0.1, 0.3} × training-fold target MAD);
Bayesian Ridge (library defaults); fixed Matérn-3/2 GP (bounds fixed —
nothing tuned; normalize_y, n_restarts 0, alpha 1e-6); one shallow
boosted-tree model (fixed XGBoost: 100 trees, depth 2, lr 0.05, n_jobs
1, seed 42 — optional dependency per DD-003, skipped cleanly when
absent, never required for the core analysis).

**MOVE arm.** Baselines: fold-global mean; held-out recording's own
per-block target mean (naive-target baseline, never a model claim);
persistence (one-update lag). Primary: **lagged Ridge / Wiener** over
the frozen lag stack (same alpha grid). Comparators: linear SVR (same C
grid); the same fixed shallow XGBoost.

A prespecified DMD representation ablation (§11) compares conventional
multiband power with deterministic dynamic-mode-decomposition spatial
node features (frozen rank 8, min(8, n_channels) dimensions) under
identical outer folds; DMD spatial edges are a clearly secondary
ablation (much larger dimensionality). DMD rank and features are fitted
within outer-training data only.

## 9. Metrics, baselines, promotion (frozen)

Subject-level metrics: MAE, RMSE, R², Pearson r, Spearman ρ on aligned
per-subject out-of-fold predictions (aligned write-back at original
indices, as in the corrected pipeline). MOVE metrics: per-held-out-
subject window-level Pearson r first, then median and IQR across
subjects; pooled-window values are secondary descriptives carrying the
explicit non-independence caveat — thousands of windows are not
thousands of independent participants.

**Promotion rule (both error metrics, never correlation alone).** A
subject-level model promotes over the field only if, on identical outer
folds, it improves **both MAE and RMSE** versus **both dummy baselines
and the frozen five-feature reference**; results carry subject-cluster
bootstrap CIs and the corrected permutation null, with Holm-adjusted
p-values across the prespecified family (primary target ×
{ridge, SVR, Bayesian ridge, GP, XGBoost} × {MAE, RMSE} within one arm)
reported alongside raw ones. A MOVE decoder is discussed as superior
only if it improves median across-subject MAE versus the persistence
and block-mean baselines on identical folds.

## 10. Uncertainty and null procedures (frozen)

- **Subject-cluster bootstrap:** 2,000 resamples, seed 42, 95 %
  percentile intervals; subjects (all their rows) resampled together;
  percentile intervals undercover at pilot n and describe this cohort.
- **Label-permutation null (corrected):** ≥ 1,000 subject-level
  permutations (floor for any reported tail probability; SCI-009 floor
  is 200) through the **complete** nested grouped-CV workflow; each
  permutation's null statistic is scored against that permutation's own
  permuted targets — never the original targets; conservative
  p = (1 + #at-least-as-favorable) / (1 + n); stratified by recording
  count; seed 42.
- **MOVE autocorrelation-preserving null:** circular shifts of the
  glove target within each recording (neural windows unchanged), each
  shift's **effective circular displacement**
  `min(shift, span − shift)` ≥ 2 s — v1.0.1 clarification: a nominal
  shift near `span` is circularly a tiny shift in the opposite
  direction, so the sampling range is `[min_bins, span − min_bins]` and
  blocks shorter than twice the minimum displacement are excluded and
  counted — ≥ 1,000 shifts; blockwise shifts (segments ≥ 2 s, same
  effective-displacement guarantee) as the secondary null; independent
  row shuffling is never used.
- **Leave-one-subject influence** (refit deltas) for the primary model,
  descriptive only.
- All p-values on this cohort are descriptive/hypothesis-generating and
  are reported with the `permutation_caveat` for the actual subject
  count.

## 11. Feature-block ablations (frozen)

Subject-level arms — every block × every model on identical outer
folds, all reported (no outer-test selection):

1. `clinical` — age, dbs_volt, dbs_freq;
2. `rest_spectral` — log relative band powers, aperiodic exponent/
   offset, aperiodic-adjusted band powers, HFO block when cohort-uniform
   rules permit;
3. `rest_dynamics` — Hjorth ×2, line length, spectral entropy, beta
   bursts ×3;
4. `delta_spectral` — log-ratio band powers, aperiodic exponent change,
   aperiodic-adjusted band-power changes;
5. `delta_dynamics` — signed dynamics differences;
6. `combined` — REST spectral + dynamics + clinical;
7. `delta_combined` — delta_spectral + delta_dynamics + clinical (§12's
   state-change criterion is evaluated on THIS block against block 6;
   implemented in the v1.0.1 corrective revision — the v1.0.0
   integration run never built it).

MOVE arm representation ablation: multiband power vs DMD spatial nodes
vs combined, plus a lag ablation over the frozen lag stack.

## 12. Methodology-transfer criteria and interpretation matrix (frozen)

| Outcome | Criterion (all descriptive/exploratory) |
|---|---|
| Movement-state transfer succeeded | lagged Ridge: median across-subject within-subject Pearson r > 0 with bootstrap 95 % CI excluding 0 **and** median across-subject MAE better than persistence and block-mean baselines |
| Resting-trait evidence | Ridge on the combined REST block meets the full promotion rule **and** Holm-adjusted permutation p < 0.05 on MAE or RMSE |
| State-change evidence | DELTA combined block beats the REST combined block on **both** MAE and RMSE on identical folds **and** meets the promotion rule versus baselines |
| Global null | none of the above |

No clinical utility, DBS-outcome prediction, or causal claims under any
outcome.

## 13. Confirmatory / exploratory / post-hoc rules (frozen)

- **This cohort is exploratory.** Its targets and basic structure were
  already inspected in the recorded first authorized run; the freeze
  makes the follow-on protocol *prespecified*, not confirmatory. Every
  follow-on result on the DABI cohort is exploratory regardless of
  freezing, and is reported with the small-sample caveats.
- **Confirmatory** is reserved for an external, independent cohort
  evaluated under an unchanged contract version, pre-registered before
  any inspection of that cohort.
- **Post-hoc** = any target, band, model, kernel, channel, lag,
  threshold, or analysis choice made after inspecting follow-on
  outputs, or any analysis outside this document. Post-hoc work is
  labeled `post_hoc` in outputs, reported separately, and is never
  eligible for promotion or transfer claims.
- Every run records `protocol_version` and `protocol_hash`; results
  from different contract versions are never mixed in one primary
  table.

## 14. Versioning and changelog

Frozen quantities may change only via a new semver with a changelog
entry here; the pinned-hash test enforces that the module, this
document, and the recorded hash move together.

| Version | Date | Change | Hash at freeze |
|---|---|---|---|
| 1.0.0 | 2026-11-16 | Initial freeze (Prompt 0, owner instruction) | `ed07c397792073a265019b25491687bf27ae5d2836b24b9a8d9f6625d04565d1` |
| 1.0.1 | 2026-11-17 | **Corrective** (post-v1.0.0-review): completes implementation coverage of already-frozen criteria — adds the `delta_combined` ablation block (§11 "DELTA analogously", never built under v1.0.0); pins the shift null to **effective** circular displacement `min(shift, span − shift)` ≥ 2 s (the v1.0.0 sampler could draw a nominal shift that is circularly a one-bin shift); requires the complete ten-test Holm family (optional-model members absent from an environment are entered conservatively at p = 1 and marked `not_executed`); MOVE baselines must emit per-subject MAE alongside correlation (the frozen transfer criterion was never fully evaluable under v1.0.0 outputs). No target, band, model, grid, floor, or fold rule changed. v1.0.0 runs and their provenance remain immutable history. | `5e0e2566f9c5df24d8ac751cceaf96d27a15a781dfb70e90b6d2a0f78e1e92a7` |

## 15. Unresolved data-layout issues (recorded, not guessed)

Resolved from acquisition metadata at conversion time — never guessed;
until then the frozen defaults above apply:

- per-recording ECoG and glove sampling rates (govern §3's band
  resolution and the HFO gate);
- glove channel count, units, cross-subject calibration (assumed
  absent — amplitude stays within-subject-only);
- channel→anatomical-region label availability (fallback aggregation,
  §5);
- electrode adjacency for bipolar rereferencing (fallback common
  median reference);
- number and duration of REST/MOVE blocks per recording (governs the
  minimum-usable-data gates and MOVE window counts);
- whether raw tables share the `restmove_segs` sample grid per modality
  (the `.npz` interchange contract requires a common grid, resolved at
  conversion);
- clinical workbook fields beyond age/dbs_volt/dbs_freq (the clinical
  block stays frozen at three covariates until documented in a version
  bump).

## 16. Signal-processing implementation notes (added 2026-11-16, Prompt 1)

The signal layer is implemented and verified on synthetic fixtures only
(`curve_icon_dbso.signal_qc`, `curve_icon_dbso.neural_features`,
`curve_icon_dbso.pac`); no real-data feature was computed in this phase
and no neural-feature/target correlation was inspected. Exact units and
definitions, as implemented:

| Quantity | Units / definition |
|---|---|
| `{band}_power` | Welch PSD integral over the band — (sensor units)² |
| `{band}_relative_power` | band power ÷ total power over [4 Hz, top active edge] — dimensionless |
| `{band}_log_relative_power` | log10 of the above — dimensionless |
| `{band}_power_aperiodic_adjusted` | mean in-band PSD ÷ aperiodic-predicted PSD at the band center — dimensionless periodic surplus |
| `aperiodic_exponent` | χ in PSD ∝ 1/f^χ = −(OLS slope of log10 PSD vs log10 f over [4, top]) — dimensionless; 1/f background → χ ≈ 1 |
| `aperiodic_offset` | intercept of the same fit — log10(sensor units²/Hz) at f = 1 Hz |
| `hjorth_mobility` | sqrt(var(x′)/var(x)) with x′ ≈ diff(x)·fs — radians/second; a pure sine of f Hz gives 2πf |
| `hjorth_complexity` | mobility(x′)/mobility(x) — dimensionless (≈ 1 for a sinusoid) |
| `line_length_per_s` | Σ\|Δx\| / duration — sensor units / s |
| `spectral_entropy` | normalized Shannon entropy of the PSD over the active range — [0, 1] |
| `beta_burst_occupancy` | fraction of samples above the recording's REST 75th-percentile envelope threshold — dimensionless; REST blocks are evaluated against their own threshold (≈ 0.25 by construction; deviations measure burst concentration) |
| `beta_burst_mean_duration_ms` / `…_median_amplitude_ratio` | mean above-threshold run length (ms; runs ≥ 50 ms) / median envelope÷threshold during bursts — dimensionless; `beta_burst_n` distinguishes "no bursts" (n = 0) from measured values |
| `{band}_log_ratio_power` (DELTA) | move log-relative power − rest log-relative power — dimensionless (a difference of logs, never a log of a log) |
| `*_delta` (DELTA dynamics) | MOVE − REST signed difference in native units |
| PAC MVL (optional) | mean vector length of mean-normalized high-band amplitude over low-band phase — dimensionless [0, 1]; gated on ≥ 10 low-band cycles and surrogate z ≥ 2 against ≥ 200 seeded circular shifts (shifts ≥ 1 s); disabled unless explicitly enabled |

Implementation readings of the frozen rules (thresholds unchanged):

- **Saturation/rail detection** is applied per channel-block: samples at
  that block's own max/min value. This is strictly more sensitive than
  cohort-wide rails and cannot mask a pinned channel.
- **PSD estimator**: the contract-approved Welch (1 s Hamming, 50 %
  overlap). Multitaper is deliberately not added.
- **Comparator note**: the frozen SVR spec string ("LinearSVR-equivalent")
  means sklearn `SVR(kernel='linear')` — the epsilon-SVR with a linear
  kernel.
- QC summaries carry counts and machine-readable reason strings only;
  per-window/per-channel records stay in gitignored local outputs.
  Cohort-level normalization is deferred to the modeling phase's
  fold-local sklearn Pipeline (fit on training subjects, applied to
  held-out subjects); the burst threshold and all QC references are
  within-recording and target-independent by construction.

## 17. Glove phenotyping implementation notes (added 2026-11-16, Prompt 2)

Implemented in `curve_icon_dbso.glove_phenotyping`; verified on
synthetic waveforms with known ground truth (known rate, interval
jitter, amplitude decay, missing cycles, sensor inversion, drift,
noise). The frozen `tap_rate_mean` implementation
(`curve_icon_dbso.kinematics`) is passed through verbatim — never
re-fit, re-scored, or changed. Exact units and definitions, as
implemented:

| Quantity | Units / definition |
|---|---|
| `tap_rate_mean` (primary) | frozen kinematics median dominant 0.5–4 Hz peak — Hz; passthrough, unchanged |
| `tap_interval_cv` (secondary) | SD/mean of successive inter-cycle intervals (s) — dimensionless; per contributing channel-block with ≥ 5 intervals; recording value = median across blocks; ≥ 2 contributing blocks required |
| `tap_vigor_index` (secondary) | RMS(diff(filtered)·fs) / (2π · f_tap · RMS(filtered)) — dimensionless, amplitude-independent (a pure sinusoid scores ≈ 1.0 at any amplitude); the diff is scaled to per-second velocity units |
| `tap_sequence_slope` (secondary) | OLS slope of per-cycle peak amplitude vs cycle index ÷ mean amplitude — dimensionless per-cycle fractional change (negative = decrement) |
| `tap_amplitude` (within-subject-only) | median RMS of the filtered channel-blocks — sensor units; recorded for QC only, never modeled cross-subject |
| `glove_vigor_envelope_z` (MOVE target) | per-recording z-scored vigor envelope, 100 ms update grid — dimensionless; block ids mark MOVE-block membership; windows never cross block boundaries |

Implementation notes:

- Contribution status reuses the primary's frozen prominence rule
  (single source of truth — the secondary targets cannot drift from the
  primary's channel-block inclusion); cycle detection uses the frozen
  0.15 s minimum separation and the inversion rule (orientation = larger
  peak count, ties upright).
- The detection band-pass uses the numerically stable second-order-
  section form of the same frozen order-4 Butterworth (the raw (b,a)
  factorization of a narrow normalized band-pass is ill-conditioned
  enough to break amplitude invariance at the 1e-2 level).
- Every function takes glove arrays only — no ECoG argument exists, so
  target extraction structurally cannot read neural data (future or
  otherwise), participant identity, or evaluation-fold membership.
  Targets may summarize a whole block (they are prediction targets);
  causality binds the neural windows that decode them (§6).
- `format_glove_qc_report` produces the sanitized aggregate QC template
  (inclusion counts, per-target availability, reliability) from
  per-recording dicts that carry no subject identity; it structurally
  cannot print per-subject values.

## 18. Subject-level modeling implementation notes (added 2026-11-16, Prompt 3)

Implemented in `curve_icon_dbso.subject_modeling`; verified on
deterministic synthetic scenarios (linear recovery, nonlinear failure,
null, outlier dominance) plus a deliberate fingerprint-leakage contrast
(row-wise leave-one-row-out evaluation of a memorizing model looks
spuriously perfect; the frozen subject-grouped pipeline collapses to the
naive baseline — and the module refuses subject-straddling folds
outright). As implemented:

- **Blocks** resolve to exact frozen column names (`select_block`);
  `combined` = rest_spectral + rest_dynamics + clinical and
  `delta_combined` = delta_spectral + delta_dynamics + clinical (the
  §12 state-change comparison runs on the latter; added in v1.0.1). A
  missing column raises with the full list — nothing is imputed into
  existence.
- **Nested CV**: identical outer folds computed once per arm and reused
  across every block × model; inner selection (GroupKFold(3), inner-LOSO
  under 6 outer-training subjects) uses inner-training subjects only;
  chosen hyperparameters are recorded per fold. SVR epsilon =
  factor × training-fold target MAD (fold-local; documented fallbacks
  for degenerate MADs). The GP's Matérn-3/2 kernel and noise bounds are
  frozen (`bounds='fixed'`) — nothing is tuned.
- **Statistics**: per-fold MAE/RMSE always; R²/Pearson/Spearman only
  when a fold holds ≥ 2 subjects (else `None`, never fabricated);
  aggregates on the per-subject aligned OOF table.
- **Uncertainty**: bootstrap reuses the corrected
  `bootstrap.subject_bootstrap_ci`; the permutation null reruns the
  COMPLETE nested pipeline per permutation (inner selection included)
  and scores each null statistic against that permutation's own
  permuted targets — the corrected convention pinned since the
  2026-11-16 review.
- **Influence**: leave-one-subject aggregate MAE/RMSE deltas (descriptive;
  a negative delta marks the dominant subject).
- **Coefficient stability**: linear-model coefficients per outer fold in
  the fold-local STANDARDIZED space — signs are comparable across folds
  only after scaling (stated in the function contract).
- **Prediction intervals**: Bayesian Ridge and GP return per-row
  predictive std via `return_std` through the fitted pipeline — labeled
  exploratory wherever emitted.
- **Promotion**: BOTH MAE and RMSE must improve versus BOTH naive
  baselines AND the frozen five-feature reference (computed with the
  frozen `kinematics` implementation on the identical folds) — point
  estimates on identical outer folds; correlation never promotes.
  No SHAP; no outer-test selection anywhere.

**Local-only real-data command (implemented, not executed here):**
`scripts/run_followon_subject_modeling.py` — runs the full frozen chain
against the owner's local authorized copy, writes gitignored outputs to
a VERSIONED run directory (never overwrites earlier runs, including the
recorded first authorized run), prints the non-blocking data-use
reminder, and records provenance (git SHA, lock hash, protocol version +
hash, stage, data-use note). Execution is a deliberate owner action;
nothing is gated on it.

## 19. MOVE time-resolved study implementation notes (added 2026-11-16, Prompt 4)

Implemented in `curve_icon_dbso.move_modeling`; verified on synthetic
fixtures with exact neural-to-glove coupling. As implemented:

- **Window indexing (the off-by-one contract)**: the target bin's time
  is its CENTER `(bin + 0.5)·0.1 s`; a lagged window ends at
  `target_time − lag` (sample EXCLUSIVE — half-open `[start, end)`), so
  no window ever contains the target instant or later samples. Windows
  that would extend before the block start or past the block end are
  dropped with machine-readable reasons (`window_outside_block`),
  as are windows overlapping declared `invalid_intervals`.
- **Features per window per lag**: log-relative band power per active
  band + aperiodic exponent, Welch (0.5 s segments) per QC-OK channel,
  frozen median aggregation across channels; window QC (flatline /
  saturation / transients) drops windows with reasons.
- **Decoders**: lagged Ridge (frozen 7-point alpha grid, inner-selected),
  linear SVR (frozen C grid), fixed shallow XGBoost (optional). Inner
  selection: subject-grouped GroupKFold(3) (inner-LOSO < 6) on
  outer-training windows only; identical outer folds across models,
  blocks, and lag subsets.
- **Baselines**: fold-global mean; the held-out recording's own
  per-block target mean (naive-target reference, labeled "never a model
  claim"); persistence (one-update lag). Each baseline reports
  per-held-out-subject Pearson r AND MAE — median/IQR across subjects —
  so the frozen §12 transfer criterion (median MAE vs persistence and
  block-mean) is evaluated in code, with its components and Boolean
  stored in `run_summary.json` (v1.0.1; v1.0.0 emitted correlations
  only, leaving the criterion's MAE half unevaluable).
- **Reporting**: per-subject window-level Pearson r and MAE first;
  median/IQR across subjects as the headline; pooled-window MAE only as
  `pooled_mae_secondary_non_independent`. The median-r subject-cluster
  bootstrap is saved as a machine-readable artifact
  (`bootstrap_median_r.json`: statistic, percentile endpoints, seed,
  resample count, and the full draw list) — never prose-only.
- **Nulls**: circular shifts of the target within each recording block
  (effective circular displacement `min(shift, span − shift)` ≥ 2 s;
  blocks too short to guarantee it are excluded and counted — v1.0.1);
  blockwise (2 s segment multiples, same guarantee) as the secondary
  null — BOTH are run and saved by the real-data command; independent
  row shuffling never used (asserted).
- **Deferred by design**: a REST-versus-MOVE detection endpoint is NOT
  prespecified in the frozen contract and is deliberately not
  implemented (adding it after seeing results would be post-hoc work).
- **Plots**: sanitized PNGs (validated categorical palette slots in
  fixed order, one axis per panel, subject positions on axes — no
  identifiers) for the per-subject distribution, observed-vs-null, lag
  ablation, and band-block ablation.

**Local-only real-data command (implemented, not executed here):**
`scripts/run_followon_move_modeling.py` — same versioned-output and
provenance discipline as the subject-level command; synthetic end-to-end
verification in `tests/test_run_followon_move_script.py`.

## 20. DMD representation implementation notes (added 2026-11-16, Prompt 5)

Implemented as an OPTIONAL module (`curve_icon_dbso.dmd_representation`);
verified on deterministic synthetic fixtures only. **No novelty or
superiority is claimed from any in-sample result** — the only success
criterion is prespecified improvement on held-out subjects with
uncertainty and an autocorrelation-preserving null, evaluated through
the frozen comparison machinery (identical outer folds, same primary
linear decoder, per-subject-first metrics).

- **Rank**: the single frozen rule `min(8, n_ok_channels)` — never
  tuned, never selected in inner folds.
- **Determinism**: exact SVD/eig (no randomized paths); modes and
  eigenvalues sorted by a deterministic key (−|λ|, Re, Im) so feature
  columns have a stable order; bitwise-identical features on identical
  windows (asserted).
- **Spatial-node features (real)**: per kept mode — log10|λ|, frequency
  (cycles/sample), log10 amplitude, and the node magnitude
  (median across QC-OK channels, or per-region medians in the frozen
  `REGION_LABELS` order when a region map exists). Differing channel
  counts enter ONLY through this region representation or the median
  fallback — patient-specific channel positions are never compared as
  aligned. Feature dimensionality: 4r without a region map;
  3r + r·|REGION_LABELS| with one.
- **Spatial-edge features**: per channel pair, median across modes of
  the node-magnitude products — dimensionality ∝ n_channels²/2, gated
  behind explicit `with_edges` (clearly secondary ablation).
- **Stabilized amplitudes**: modal amplitudes for FEATURES are computed
  against the orthonormalized mode basis (QR of Φ) via the CONJUGATE
  transpose `Qᴴx₀` — for complex modes the plain transpose is not the
  adjoint and the projection is wrong (fixed in v1.0.1). The raw
  least-squares amplitudes are ill-conditioned for near-degenerate
  modes (noise amplification demonstrated empirically at ~40× in
  log-space) and are retained only for the exact-dynamics
  reconstruction diagnostic. Node/edge magnitudes use the SAME `Q`
  basis as the amplitudes — multiplying raw-Φ magnitudes by Q-basis
  amplitudes (pre-v1.0.1) compared two different bases and is
  discontinued.
- **QC/cost reporting**: failed channels are excluded before the SVD
  (never imputed); per-window compute seconds and TWO reconstruction
  diagnostics are recorded, never as model features: the one-step
  relative error (always finite and interpretable) and the multi-step
  trajectory error — the latter computed ONLY when every kept
  eigenvalue satisfies |λ| ≤ 1.05, because an unstable modal evolution
  diverges and its trajectory error measures divergence, not the
  linearity of the data (pre-v1.0.1, the unstable branch replayed one
  one-step prediction across the window, which is why its ≈10³ "mean
  relative error" was uninterpretable). Neither diagnostic licenses a
  claim about how "linear" the physiology is.
- **Negative control (synthetic, documented)**: a weak coherent tone
  buried under high-variance low-rank interference — band power resolves
  the tone (pooled |r| ≈ 0.99) while the rank-truncated DMD cannot
  outperform it (a small residual correlation is intrinsic: the
  unmodeled coherent component perturbs the fitted operator through
  spatial overlap). Held-out comparison on the control: power > DMD.
- **Integration**: `--with-dmd` (and the secondary `--with-dmd-edges`)
  on `scripts/run_followon_move_modeling.py`; the builder gains an
  opt-in `capture_windows` flag so DMD sees exactly the builder's kept,
  QC-passed windows.

## 21. Integration status and deferred contrastive run (2026-11-16; corrected 2026-11-17)

**Superseded statement, kept for the record:** the v1.0.0 integration
was first described as "executed at full prespecified depth." That
statement is **withdrawn**: the post-run review (2026-11-17) found parts
of the frozen matrix unexecuted under v1.0.0 — no MOVE baseline MAEs
(transfer criterion only partially evaluable), a shift null whose
effective displacement could be a single update bin, no `delta_combined`
block, and a Holm family run over 8 of its 10 members. The v1.0.0 run
directories and `docs/results_followon_real.md`'s original claims remain
in place as immutable history, marked superseded where defective; the
corrective v1.0.1 rerun (identical inputs, verified by input manifests)
is the version whose sanitized aggregates carry the frozen-contract
verdicts.

The contrastive addendum (Prompt 7) is implemented (`curve_icon_dbso.contrastive_representation`, NumPy InfoNCE with the frozen caps: hidden 32, embedding 8, temperature 0.1, weight decay 1e-4, lr 0.05, batch 256, ≤ 200 epochs, seeds {42, 1, 2}, learning curve {50, 200} epochs, deferred rule floor 12 outer-training subjects) and synthetically verified, including a numeric-gradient check of the tied-encoder InfoNCE gradients. Its contract-version bump (1.1.0, additive minor) and real-data run are **DEFERRED**: the bump is withheld until the v1.0.1 corrective integration is recorded; windows are not independent subjects, and the addendum runs only after the 1.1.0 bump documents it. The secondary kinematic targets (`tap_interval_cv`, `tap_vigor_index`, `tap_sequence_slope`) were extracted but never modeled under v1.0.0 or the v1.0.1 corrective rerun — a recorded omission; any future secondary-target inference is a separate, labeled analysis with its own multiplicity control, never part of the completed primary inference.

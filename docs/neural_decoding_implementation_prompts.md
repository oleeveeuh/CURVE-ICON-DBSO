# Repository Prompts: Follow-on Neural Decoding Study

These prompts are intended to be run sequentially in the `CURVE-ICON-DBSO` repository. Each prompt preserves the corrected first authorized run as an immutable exploratory result and implements the follow-on study as a separate protocol. Do not commit, push, rewrite Git history, deploy to AWS, or place real participant data in tracked files unless the repository owner separately requests that action.

## Prompt 0 — Audit, protocol freeze, and design contract

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO.

Design and implement the protocol contract for the follow-on ECoG/kinematic study. First read docs/literature_review.md, docs/methodology.md, docs/limitations.md, docs/data_access.md, the active kinematic extraction/evaluation code, current tests, and the complete git diff. Preserve all existing user changes, especially the permutation-null correction. Do not modify or overwrite the recorded first authorized real-data result.

Create a new follow-on methodology document and a machine-readable versioned analysis contract. Define three separate arms:
1. REST: resting neural features predicting subject-level tapping phenotypes;
2. MOVE: time-resolved movement ECoG predicting concurrent glove movement;
3. DELTA: movement-minus-rest neural modulation predicting subject-level tapping phenotypes.

Freeze before any new real-data result is inspected:
- primary and secondary targets;
- neural feature families and exact frequency boundaries;
- artifact/QC rules;
- channel/anatomical aggregation rules;
- window sizes, update rate, and neural lags;
- models and bounded hyperparameter grids;
- outer and inner subject-grouped validation;
- metrics, baselines, promotion rules, uncertainty, and null procedures;
- planned feature-block ablations;
- criteria for successful methodology transfer;
- rules distinguishing confirmatory, exploratory, and post-hoc work.

Default decisions unless the existing data make them technically impossible:
- preserve tap_rate_mean as the frozen primary subject-level target;
- secondary kinematic targets: interval variability, movement velocity or amplitude, and sequence effect;
- primary expanded-feature estimator: Ridge;
- comparators: linear SVR, Bayesian Ridge, fixed Matérn-3/2 Gaussian-process regression, and one shallow boosted-tree model;
- time-resolved baseline: lagged Ridge/Wiener regression;
- time-resolved comparators: linear SVR and shallow XGBoost;
- standard neural bands: theta 4–8, alpha 8–12, low beta 13–20, high beta 20–35, low gamma 35–60, high gamma 60–100 or 90–200 subject to acquisition constraints, and HFO only where Nyquist/artifact rules permit;
- outer evaluation unit is always subject; no subject's windows may appear in more than one outer fold.

Add schema/contract tests that fail if targets, bands, folds, or model grids change silently. Use synthetic fixtures only in tests. End with a concise inventory of files changed, design decisions, unresolved data-layout issues, and verification performed.
```

## Prompt 1 — Signal QC, rereferencing, and feature extraction

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO and implement the signal-processing layer defined by the frozen follow-on contract. Read the current diff first and preserve unrelated work. Do not run model selection or inspect neural-feature correlations with real targets in this phase.

Implement pure, tested functions for:
- validated bipolar rereferencing or the contract's documented fallback when electrode adjacency is unavailable;
- line-noise and harmonic handling;
- explicit clipping/rejection of flatlines, saturation, rapid transients, and implausible band power;
- per-window and per-channel QC with machine-readable failure reasons;
- multitaper or contract-approved Welch PSD;
- relative/log power and aperiodic-adjusted periodic power;
- theta, alpha, low/high beta, low/high gamma, and valid HFO summaries;
- beta-burst occupancy, duration, and amplitude using thresholds learned from training/reference data without target information;
- Hjorth mobility/complexity, line length, and spectral entropy;
- optional PAC behind strict reliability and surrogate-QC gates;
- REST, MOVE, and MOVE-minus-REST feature blocks;
- region-aware robust aggregation when PM/M1/S1 or equivalent localization exists.

Do not silently impute failed signal windows or zero-fill failed channels. Carry QC counts and reasons into aggregate outputs without participant identifiers. Ensure any normalization that estimates cohort parameters can be fitted on training data and applied to held-out data. Add deterministic synthetic tests for known sinusoids, 1/f backgrounds, bursts, artifacts, channel failures, band boundaries, and state differences. Add property tests for finite outputs and invariance where scientifically expected. Update documentation with exact units and definitions.

Run focused tests, lint, and the offline suite if feasible. Do not access AWS. Do not commit or push. Report any acquisition-specific choice that cannot be resolved from repository evidence rather than guessing.
```

## Prompt 2 — Expanded glove phenotyping

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO. Implement the prespecified secondary glove outcomes without changing the frozen tap_rate_mean implementation or its historical result.

From MOVE blocks and individual glove channels, implement robust, unit-documented estimates of:
- tap or movement rate;
- inter-tap interval variability;
- movement velocity or a defensible amplitude-independent vigor envelope;
- amplitude where cross-subject calibration permits it, otherwise mark it within-subject-only;
- sequence effect/decrement across a block;
- valid-cycle counts and target reliability/QC.

Use peak/event detection settings frozen in the new contract. Do not optimize thresholds against ECoG features or model performance. Separate primary and secondary endpoints in every output. Include synthetic waveforms with known rate, jitter, amplitude decay, missing cycles, sensor inversion, drift, and noise. Ensure target extraction never reads future neural data, participant identity, or evaluation-fold membership.

Produce a sanitized aggregate QC report template showing inclusion counts, target availability, and reliability without subject codes or participant-level values. Run focused tests and lint; preserve existing user modifications; do not commit, push, deploy, or copy real outputs into tracked paths.
```

## Prompt 3 — Subject-level REST and DELTA modeling

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO. Implement the frozen subject-level REST and DELTA modeling protocols using synthetic fixtures first.

Build sklearn-compatible pipelines for the prespecified models: mean/median baselines, Ridge, linear SVR, Bayesian Ridge, fixed Matérn-3/2 Gaussian-process regression, and one shallow boosted-tree comparator. All imputation, scaling, feature reduction, and hyperparameter selection must be fitted only on inner-training subjects. Use nested subject-grouped validation with identical outer folds for every model and feature block. Keep the clinical/stimulation-only, REST spectral, REST dynamics, DELTA spectral, DELTA dynamics, and combined blocks distinct.

Implement:
- aligned out-of-fold subject predictions;
- per-fold and aggregate MAE, RMSE, R², Pearson and Spearman statistics;
- subject-cluster bootstrap intervals;
- corrected target-permutation testing through the complete nested pipeline;
- leave-one-subject influence analysis;
- foldwise coefficient/sign stability for linear models;
- prediction intervals for Bayesian/Gaussian-process models, clearly labeled exploratory;
- explicit comparisons against both naive baselines and the frozen five-feature result;
- a promotion rule requiring improvement in both MAE and RMSE, not a favorable correlation alone.

Never select a feature block, model, kernel, channel, or target using outer-test performance. Do not add SHAP by default. Add leakage tests that deliberately plant subject-specific fingerprints and fail under row-wise splitting. Add deterministic synthetic tests for linear, nonlinear, null, and outlier-driven scenarios.

After the implementation and tests pass, provide a separate local-only command for the authorized real-data run. If executing it is within the current instruction, write only ignored participant-level outputs and tracked aggregate documentation only when publication terms allow it. Preserve the original run rather than overwriting it. Do not use AWS, commit, or push.
```

## Prompt 4 — Time-resolved MOVE decoder

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO. Implement the MOVE methodology-transfer study defined in the follow-on contract. Read and preserve the current diff and existing real-data artifacts.

Create a windowed dataset builder that aligns ECoG windows with continuous glove targets while retaining subject, recording, block, and timestamp metadata locally. Default to contract-defined 500 ms–1 s windows, 100–250 ms updates, and up to 500 ms of preceding neural context. Prevent windows from crossing block boundaries or known invalid intervals. Participant identifiers and raw values must remain in ignored local outputs.

Implement three prespecified decoders:
1. lagged Ridge/Wiener regression as the primary temporal baseline;
2. linear SVR;
3. one shallow XGBoost model.

Evaluate continuous glove velocity/tapping vigor and, if prespecified, a separate REST-versus-MOVE sanity endpoint. Use outer subject-held-out folds and inner subject-grouped model selection. Every window from a subject must remain in one outer fold. Report metrics per held-out subject first and then summarize across subjects; do not treat windows as independent observations. Include block-mean/time-only and naive target baselines.

Implement autocorrelation-preserving nulls using contract-defined circular shifts or blockwise shifts within recording, with minimum shifts that break local neural–behavior alignment. Do not use independent row shuffling. Add tests proving that ordinary random window splitting produces spuriously optimistic results on a synthetic subject-fingerprint dataset while the implemented grouped split does not. Add alignment tests with known neural-to-glove lags and off-by-one traps.

Produce sanitized aggregate results and plots: held-out performance distribution, observed versus null performance, lag ablation, feature-block ablation, and per-subject metric distribution without identifiers. Do not commit real predictions, access AWS, or push changes.
```

## Prompt 5 — Dynamic mode decomposition representation

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO. Add the prespecified dynamic-mode-decomposition representation experiment as an optional module, following docs/literature_review.md and the frozen analysis contract.

Implement deterministic multichannel DMD for contract-defined ECoG windows and derive real-valued spatial-node features. Spatial-edge features may be implemented but must remain a separate ablation because their dimensionality is much larger. Select/truncate rank only within inner training folds or use the single frozen rank rule from the contract. Handle differing channel counts through the documented anatomical-region representation; do not compare patient-specific channel positions as though they were aligned.

Compare under identical outer subject folds:
- conventional multiband power with the primary linear decoder;
- DMD node features with the same decoder;
- combined power + DMD node features;
- optional DMD edge features as a clearly secondary ablation.

Include reconstruction, conjugate-pair, deterministic-output, rank, dimensionality, and synthetic oscillatory-system tests. Add a negative-control dataset where DMD should not outperform power. All preprocessing and dimensionality reduction must be fitted within folds. Report computational cost and missing/QC behavior as well as predictive performance.

Do not claim novelty or superiority from an in-sample result. The only success criterion is prespecified improvement on held-out subjects with uncertainty and an autocorrelation-preserving null. Do not commit participant-level artifacts, use AWS, commit, or push.
```

## Prompt 6 — Integrated run, inference, and research report

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO. Integrate and verify the complete follow-on neural-decoding study after Prompts 0–5 are implemented. Preserve the frozen first authorized run and all existing provenance.

First run formatting, lint, focused scientific tests, leakage tests, the offline test suite, and container tests where the environment permits. Record environmental failures separately from code failures. Then run the authorized real-data pipeline locally only, using versioned output directories so no earlier results are overwritten.

Generate a sanitized aggregate research report containing:
- cohort and recording flow without subject identifiers;
- signal and target QC counts;
- exact REST, MOVE, and DELTA contracts;
- model and baseline definitions;
- nested subject-grouped evaluation diagram;
- subject-level uncertainty and corrected null procedures;
- complete model and feature-block results, including negative results;
- comparison of frozen five-feature, expanded REST, DELTA, and concurrent MOVE decoding;
- DMD versus conventional spectral features;
- leave-one-subject influence results;
- limitations, multiplicity, exploratory status, and external-validation requirements;
- a result-interpretation matrix distinguishing resting-trait evidence, movement-state transfer, state-change evidence, and global null findings.

Do not promote a model merely for p < 0.05, correlation, or a single metric. Require the frozen practical promotion rule and report all prespecified comparisons. Do not claim clinical utility, DBS-outcome prediction, or causal effects. Do not include SHAP unless separately requested and labeled descriptive.

Update the methodology append-only run log, limitations, architecture, README summary, and literature-review cross-references only after the run is complete. Store participant-level predictions, detailed fold assignments, raw features, fitted real-data models, and MLflow runs in ignored local paths. Add provenance hashes for source, lock file, analysis contract, feature schema, and aggregate input manifest. Do not deploy the real model, use AWS, commit, or push.

End with: files changed; test results; exact aggregate conclusions; unresolved threats to validity; whether methodology transfer succeeded under the prespecified definition; and the minimum external cohort needed for confirmation.
```

## Prompt 7 — Optional contrastive representation learning

```text
Work in /Users/olivialiau/CURVE-ICON-DBSO. Implement this phase only after the simpler time-resolved decoders and DMD comparison are complete. Treat it as exploratory and optional.

Add a CEBRA-style or equivalent contrastive representation for window-level ECoG. Do not apply deep learning to the 31-row subject table. Fit every representation from scratch using outer-training subjects only, or document an independently sourced pretrained model and its license. Held-out-subject signals and labels must not influence normalization, representation learning, early stopping, or hyperparameter choice.

Compare a frozen low-capacity contrastive model followed by linear Ridge/SVR against conventional multiband and DMD representations on the identical folds. Cap architecture size and the hyperparameter grid in the analysis contract before running real data. Include shuffled-label and shuffled-temporal-alignment controls, learning curves by training duration, seed variability, and compute/runtime reporting.

If the effective data volume or subject count is insufficient, implement the interface and synthetic tests but document the experiment as deferred rather than forcing a result. Do not describe additional windows as additional independent subjects. Do not commit real embeddings or participant-level outputs, access AWS, commit, or push.
```

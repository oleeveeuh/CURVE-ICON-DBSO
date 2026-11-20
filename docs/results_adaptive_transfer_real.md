# Adaptive-transfer v2 results (authorized real-data run)

**Stage:** exploratory (contract 2.1.0, hash `027503a3d0f8359f89d56b16cb47e05adf73df670924216820f57b7fa433bb20`). Run directory: `outputs/transfer_v2/run_2.1.0_20261120T193101Z_6db3abe3` (gitignored; per-subject values never committed).

## Cohort flow

- recordings loaded: 31; eligible: 31
- signal-quality-flagged retained: 1
- Track D ineligible: 0

## Frozen verdicts (explicit, per criterion)

Every verdict = criterion + bootstrap CI + FINAL Holm-adjusted p (primary family for the four primary endpoints; one event family for the event claims).

| Criterion | Verdict | Effect | Interval | Holm p |
|---|---|---|---|---|
| Track A/C-zero detection | SUCCESS | 0.537 | [0.5227425732562718, 0.5466448655152161] | 0.002997002997002997 |
| Track C zero-shot regression (median r) | SUCCESS | 0.073 | [0.03402308960006181, 0.10063441052397037] | 0.002997002997002997 |
| Track B primary (delta-MAE, 30%) | failure | 0.000 | [0.0, 0.0] | 1.0 |
| Track C few-shot gain (30%) | SUCCESS | 0.910 | [0.8898881596386982, 0.9369691409058543] | 0.001999000499750125 |
| Track D association | failure | 0.003 | [-0.026579333887184902, 0.03824261118925803] | 0.07992007992007992 |
| Track D prediction | failure | 0.056 | [0.03709429041127972, 0.08103311795290671] | 1.0 |

**Global null: NO.**

## Detection (Track A; zero-shot by construction)

- Ridge logistic: median subject BAC 0.537 (IQR 0.519–0.552); bootstrap CI 0.523–0.547; majority-class baseline: BAC 0.500; exceeded
- comparator XGBoost (fixed): median BAC 0.559 (sensitivity 0.887, specificity 0.284)
- block-permutation null: p_raw = 0.000999000999000999, Holm (model family, descriptive) = 0.001998001998001998 (1,000 draws, seed 42)

## Regression (Tracks B/C)

- zero-shot median subject r (neural-only): 0.07284357338861541 (persistence 0.8988865251217911); circular-shift null p_raw = 0.000999000999000999
- few-shot 30%: median delta-MAE (0.0) = MAE(combined) − MAE(AR); circular-shift null p_raw = 1.0; blockwise p_raw = 1.0

### Calibration curve

| Track | Fraction | Subjects | Calibration minutes (median) | few-shot performance | zero-shot (same interval) | gain CI |
|---|---|---|---|---|---|---|
| regression | 0.1 | 30 | 0.204 | 0.973 | 0.044 | [0.8731805121187173, 0.9365501795873836] |
| regression | 0.2 | 30 | 0.407 | 0.974 | 0.044 | [0.8854746783987859, 0.9367648972631001] |
| regression | 0.3 | 31 | 0.611 | 0.973 | 0.044 | [0.8898881596386982, 0.9369691409058543] |
| detection | 0.1 | 28 | 0.204 | 0.517 | 0.533 | [-0.02880213467876097, 0.0024429482214145537] |
| detection | 0.2 | 29 | 0.407 | 0.521 | 0.533 | [-0.021592205168312395, 0.00398366644272774] |
| detection | 0.3 | 31 | 0.611 | 0.524 | 0.534 | [-0.015393978600795055, 0.009912912328500556] |

## Track D (event-level)

- association (log_inter_tap_interval vs anchor_cell_hfa_log_power): Fisher-z r 0.002539287586612748, CI [-0.026579333887184902, 0.03824261118925803] (31 subjects; association claim, not prediction)
- secondary association (peak_velocity_z vs anchor_cell_hfa_log_power): Fisher-z r 0.022495892830717942, CI [0.0030590193590930895, 0.044339130836261134]
- secondary association (tap_amplitude_z vs anchor_cell_hfa_log_power): Fisher-z r 0.029298932633303763, CI [-0.006966040180754009, 0.07463163118811168]
- event p-value family (Holm within family): {"event_prediction_shift_null": 1.0, "log_inter_tap_interval_association": 0.07992007992007992, "peak_velocity_z_association": 0.003996003996003996, "tap_amplitude_z_association": 0.003996003996003996}
- prediction: median delta-MAE 0.055870363797572264, bootstrap CI [0.03709429041127972, 0.08103311795290671], null p_raw 1.0

## Ablations (prespecified; all reported, none selected)

Paired per-subject contrasts against the corresponding primary condition (two-sided sign-flip null, Holm within the contrast family); aggregates only.

| Ablation contrast | n | Median contrast | Direction | Raw p | Holm p |
|---|---|---|---|---|---|
| detection_bipolar_vs_referential_bac | 31 | -0.005 | positive favors variant | 0.233 | 1.000 |
| detection_window_1s_vs_500ms_bac | 31 | -0.003 | positive favors variant | 0.697 | 1.000 |
| regression_band_group_high_vs_full_delta_mae | 31 | 0.000 | negative favors variant | 1.000 | 1.000 |
| regression_band_group_low_vs_full_delta_mae | 31 | 0.000 | negative favors variant | 1.000 | 1.000 |
| regression_lag_subset_00_02s_vs_full_delta_mae | 31 | 0.000 | negative favors variant | 1.000 | 1.000 |
| regression_window_1s_vs_500ms_delta_mae | 31 | 0.000 | negative favors variant | 1.000 | 1.000 |

Condition medians (aggregate): detection_bipolar median BAC 0.530; detection_window_1s median BAC 0.533; regression_band_group_high median ΔMAE 0.000; regression_band_group_low median ΔMAE 0.000; regression_lag_subset_00_02s median ΔMAE 0.000; regression_window_1s median ΔMAE 0.000


## Distinctions preserved

- **association vs prediction:** Track D association (subject-replicated Spearman) and prediction (held-out error) are reported separately
- **detection vs regression:** Track A classifies REST vs MOVE; Tracks B/C decode a continuous standardized movement envelope — never mixed
- **exploratory vs confirmatory:** this cohort is exploratory by construction; confirmatory status is reserved for an external cohort under an unchanged contract
- **neural vs kinematic vs combined:** every regression claim compares the neural model against the kinematic AR baseline and the combined forecast
- **statistical vs practical:** small above-null effects may not be practically useful; every verdict carries its effect size and interval
- **zero shot vs few shot:** zero-shot uses no held-out-subject labeled data; few-shot uses only the chronological calibration segment; separate claims, separate endpoints

## Limitations

- exploratory single-center cohort; every p-value is descriptive; percentile bootstrap intervals undercover at this n
- glove physical units are UNKNOWN across subjects; magnitude claims are within-person only
- spatial mapping is grid-relative anchored (no MNI coordinates exist); second-strip contacts are EXCLUDED from the primary grid (their anchor assignment is undocumented) and are used in no analysis
- zero-shot regression claims are correlation-only by the frozen scale rule (target standardization may never be fitted on a final test interval)
- OFFLINE RETROSPECTIVE analysis (owner decision 2026-11-19): calibration and test spans are processed independently end to end (no sample crosses a span boundary), and zero-phase filtering is used WITHIN a span; this contract makes no online or causal-inference claim

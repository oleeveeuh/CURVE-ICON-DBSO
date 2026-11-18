"""Subject-level resampling: cluster bootstrap confidence intervals and a
label-permutation null.

Both procedures operate on **subjects, never rows**: with one subject
contributing more than one recording, row-level resampling would treat
repeated measures of one person as independent evidence. Every draw here
moves all rows of a sampled subject together
(:func:`generate_cluster_resamples`, :func:`permute_subject_targets`).

Uncertainty quantification (SCI-008, SCI-009) — and its limits:

* Bootstrap: percentile confidence intervals for MAE and RMSE from
  2,000 cluster resamples with a fixed seed. At cohort sizes like n = 8
  the percentile method *undercovers* (intervals are too narrow); the
  intervals describe this cohort and must not be read as population
  statements.
* Permutation: at least 200 subject-level target permutations through the
  complete grouped-CV workflow — at least 1,000 whenever a tail
  probability will actually be reported. Each permutation's null
  statistic is scored against that permutation's **own permuted targets**
  (held-out rows included), never against the original targets. The
  p-value convention is the conservative
  ``p = (1 + #{null at least as favorable as observed}) / (1 + n_permutations)``
  (never returns zero, no anti-conservative small-count behavior).
  Results are **descriptive**, not confirmatory hypothesis tests, at
  n = 8 subjects; even near n ≈ 31 (the kinematic cohort) they remain
  hypothesis-generating. See :func:`permutation_caveat`.

Non-negotiable inputs: predictions must be complete and finite — this
module refuses to silently drop or impute (raised
:class:`~curve_icon_dbso.errors.DataValidationError`), consistent with the
package-wide rule that missingness is handled upstream (fold-local
imputation) and recorded, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import DataValidationError

# Protocol defaults (SCI-008 / SCI-009). Tests and the synthetic demo may
# use smaller counts for speed; real evaluation runs must use >= 2,000
# bootstrap resamples and >= 200 permutations — and >= 1,000 permutations
# whenever a tail probability will be reported (pre-publication review,
# 2026-11-16; the corrected real-data rerun uses 5,000).
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
PERMUTATION_N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42
CI_LEVEL = 0.95

# Metrics supported for bootstrap CIs and permutation p-values, with the
# direction that counts as "more favorable" for the p-value convention.
SUPPORTED_STATISTICS = ("MAE", "RMSE")
METRIC_DIRECTIONS = {
    "MAE": "lower",   # smaller is better
    "RMSE": "lower",
    "R2": "higher",   # larger is better
    "Pearson_r": "higher",
}


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate and percentile CI for one metric of one model."""

    metric: str
    point: float
    ci_low: float
    ci_high: float
    level: float
    n_resamples: int
    seed: int
    n_subjects: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "metric": self.metric,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "n_subjects": self.n_subjects,
        }


@dataclass(frozen=True)
class PermutationMetricResult:
    """Observed statistic, its permutation null, and the conservative p-value."""

    metric: str
    direction: str
    observed: float
    p_value: float
    null_mean: float
    null_median: float
    n_permutations: int
    seed: int
    null: np.ndarray

    def summary(self) -> dict[str, float | int]:
        """Machine-readable summary without the full null array."""
        return {
            "metric": self.metric,
            "direction": self.direction,
            "observed": self.observed,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_median": self.null_median,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
        }


def _validate_observations(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Shared input validation: aligned, finite, >= 2 subjects.

    Returns the cleaned arrays plus a mapping of subject -> row indices
    (order of first appearance). Missing predictions are rejected, not
    dropped: silently resampling around holes would bias every interval.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    if not (y_true.shape == y_pred.shape == (groups.shape[0],)):
        raise DataValidationError(
            f"y_true, y_pred and groups must be equally long; got "
            f"{y_true.shape[0]}, {y_pred.shape[0]}, {groups.shape[0]}"
        )
    if y_true.size == 0:
        raise DataValidationError("Bootstrap/permutation needs at least one observation")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.isfinite(values).all():
            bad = int(np.count_nonzero(~np.isfinite(values)))
            raise DataValidationError(
                f"{name} contains {bad} non-finite value(s). Missing predictions must "
                "be handled upstream (they are a QC matter), not silently resampled."
            )
    subjects = list(dict.fromkeys(groups.tolist()))
    if len(subjects) < 2:
        raise DataValidationError(
            "Need at least 2 unique subjects for subject-level resampling; got "
            f"{len(subjects)}"
        )
    row_indices = {
        subject: np.flatnonzero(groups == subject) for subject in subjects
    }
    return y_true, y_pred, groups, row_indices


def generate_cluster_resamples(
    groups: np.ndarray, n_resamples: int, seed: int
) -> list[np.ndarray]:
    """Draw ``n_resamples`` subject-level bootstrap resamples.

    Each returned array holds the ROW indices of one resample: subjects are
    drawn with replacement and every row of a drawn subject is carried
    along, so a subject drawn twice contributes two independent copies of
    all its rows (the standard cluster bootstrap).

    Raises:
        DataValidationError: On ``n_resamples < 1`` or fewer than 2 subjects.
    """
    groups = np.asarray(groups)
    subjects = list(dict.fromkeys(groups.tolist()))
    if len(subjects) < 2:
        raise DataValidationError(
            f"Need at least 2 unique subjects for cluster resampling; got {len(subjects)}"
        )
    if n_resamples < 1:
        raise DataValidationError(f"n_resamples must be >= 1, got {n_resamples}")
    row_indices = {subject: np.flatnonzero(groups == subject) for subject in subjects}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(n_resamples, len(subjects)))
    return [
        np.concatenate([row_indices[subjects[j]] for j in draw])
        for draw in draws
    ]


def subject_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    statistics: tuple[str, ...] = ("MAE", "RMSE"),
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> dict[str, BootstrapCI]:
    """Subject-cluster bootstrap confidence intervals (SCI-008).

    Resamples subjects (with all their rows) 2,000 times by default with a
    fixed seed, recomputes each statistic on every resample, and returns
    percentile intervals at ``level`` (default 95%). The point estimate is
    the full-sample value.

    Duplicated sampled clusters contribute all their rows again (standard
    cluster bootstrap). Missing/non-finite predictions raise instead of
    being dropped. Fewer than 2 subjects raise.

    At n = 8 subjects these intervals are wide and the percentile method
    undercovers — treat them as cohort descriptions, not population CIs.
    """
    from .evaluation import compute_regression_metrics

    for stat in statistics:
        if stat not in SUPPORTED_STATISTICS:
            raise DataValidationError(
                f"Unsupported statistic '{stat}'; supported: {SUPPORTED_STATISTICS}"
            )
    y_true, y_pred, _, row_indices = _validate_observations(y_true, y_pred, groups)
    if not (0.0 < level < 1.0):
        raise DataValidationError(f"CI level must be in (0, 1), got {level}")

    point_metrics = compute_regression_metrics(y_true, y_pred)
    resamples = generate_cluster_resamples(groups, n_resamples, seed)

    distributions: dict[str, list[float]] = {stat: [] for stat in statistics}
    for row_idx in resamples:
        metrics = compute_regression_metrics(y_true[row_idx], y_pred[row_idx])
        for stat in statistics:
            distributions[stat].append(float(metrics[stat]))

    alpha = (1.0 - level) / 2.0
    result: dict[str, BootstrapCI] = {}
    for stat in statistics:
        values = np.asarray(distributions[stat], dtype=float)
        low, high = np.quantile(values, [alpha, 1.0 - alpha])
        result[stat] = BootstrapCI(
            metric=stat,
            point=float(point_metrics[stat]),
            ci_low=float(low),
            ci_high=float(high),
            level=level,
            n_resamples=n_resamples,
            seed=seed,
            n_subjects=len(row_indices),
        )
    return result


def permute_subject_targets(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return ``y`` with targets permuted at the SUBJECT level.

    Subjects (all their rows) are shuffled as blocks, so every permutation
    keeps the design's repeated-measure structure intact while breaking any
    target/subject association.

    Unequal recording counts: a 2-recording subject's target block cannot
    be assigned to a 1-recording subject without fabricating rows, so the
    permutation is **stratified by recording count** — target blocks are
    exchanged only among subjects with the same number of rows, and
    subjects with no same-size partner keep their own targets. This makes
    the null slightly conservative (fewer effective exchanges) rather than
    structurally wrong; with one recording per subject (the kinematic
    cohort's design) it is the full permutation. Deterministic given
    ``rng``.
    """
    y = np.asarray(y, dtype=float)
    _, _, _, row_indices = _validate_observations(y, y, groups)
    subjects = list(row_indices)
    strata: dict[int, list[str]] = {}
    for subject in subjects:
        strata.setdefault(len(row_indices[subject]), []).append(subject)
    permuted = y.copy()
    for members in strata.values():
        if len(members) < 2:
            continue  # no same-size partner: keep own targets (conservative)
        order = rng.permutation(len(members))
        for position, source in enumerate(order):
            destination = row_indices[members[position]]
            origin = row_indices[members[source]]
            permuted[destination] = y[origin]
    return permuted


def permutation_p_value(
    observed: float, null: np.ndarray, direction: str, n_permutations: int
) -> float:
    """Conservative empirical p-value (documented SCI-009 convention).

    ``p = (1 + #{null values at least as favorable as observed}) / (1 + n)``.
    The +1 correction (Phipson & Smyth) keeps p >= 1/(1+n) — never zero —
    and is anti-conservative-safe at the small permutation counts feasible
    with tiny cohorts. "Favorable" follows :data:`METRIC_DIRECTIONS`.
    """
    null = np.asarray(null, dtype=float)
    if null.size != n_permutations:
        raise DataValidationError(
            f"null distribution has {null.size} values, expected {n_permutations}"
        )
    if direction == "lower":
        count = int(np.count_nonzero(null <= observed))
    elif direction == "higher":
        count = int(np.count_nonzero(null >= observed))
    else:
        raise DataValidationError(f"Unknown direction '{direction}'")
    return float((1 + count) / (1 + n_permutations))


def holm_adjusted_pvalues(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni step-down adjustment (family-wise error control).

    An exploratory screen inspects several models and metrics; one nominal
    ``p < .05`` among many inspected tests is the classic multiplicity
    trap. The documented remedy: define the family actually inspected and
    report step-down adjusted p-values alongside the raw ones.

    Deterministic: ties are broken by first appearance in ``pvalues``
    (Python sorts are stable). Adjusted values are non-decreasing in the
    sorted raw values, never below their raw value, and capped at 1.0.
    """
    if not pvalues:
        return {}
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for position, (name, raw) in enumerate(ordered):
        running_max = max(running_max, (m - position) * raw)
        adjusted[name] = min(1.0, float(running_max))
    return adjusted


def run_permutation_null(
    X,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict[str, object],
    metrics: tuple[str, ...] = ("MAE", "RMSE", "R2"),
    n_permutations: int = PERMUTATION_N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
    splitter=None,
    preprocessing=None,
) -> dict[str, dict[str, PermutationMetricResult]]:
    """Label-permutation null through the complete grouped-CV workflow (SCI-009).

    For each of ``n_permutations`` (>= 200 for real evaluation runs, and
    >= 1,000 when a tail probability will be reported) the targets are
    permuted at the subject level and the **complete** grouped
    cross-validation — same folds, fold-local preprocessing, aligned
    out-of-fold predictions — is rerun for every model. Each permutation's
    null statistic is computed against that permutation's **permuted**
    targets, held-out rows included (mirroring sklearn's
    ``permutation_test_score``) — never against the original targets.
    The observed metrics (from an unpermuted run of the same machinery)
    are compared against each null distribution with the conservative
    :func:`permutation_p_value` convention.

    Folds are computed once and reused across permutations: permutations
    shuffle targets, never the group structure, so the split is unchanged.

    Returns ``{model_name: {metric: PermutationMetricResult}}``. Determinism:
    fixed seed, single generator, fixed permutation order.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from .evaluation import compute_regression_metrics, run_group_cv

    for metric in metrics:
        if metric not in METRIC_DIRECTIONS:
            raise DataValidationError(
                f"Unsupported metric '{metric}'; supported: {tuple(METRIC_DIRECTIONS)}"
            )
    if n_permutations < 1:
        raise DataValidationError(f"n_permutations must be >= 1, got {n_permutations}")
    if not models:
        raise DataValidationError("run_permutation_null needs at least one model")

    y = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups)
    _validate_observations(y, y, groups_arr)

    observed_results = run_group_cv(
        X, y, groups_arr, models, preprocessing=preprocessing, splitter=splitter
    )

    # Folds depend only on X/groups; permutations change y only. The same
    # default-splitter rule as run_group_cv (LOSO for small cohorts).
    from .evaluation import _default_splitter, make_preprocessing

    splitter = splitter or _default_splitter(len(np.unique(groups_arr)))
    preprocessing = preprocessing if preprocessing is not None else make_preprocessing()
    splits = list(splitter.split(X, y, groups_arr))

    rng = np.random.default_rng(seed)
    nulls: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metrics} for name in models
    }

    for _ in range(n_permutations):
        y_perm = permute_subject_targets(y, groups_arr, rng)
        for model_name, model in models.items():
            # Same loop structure as run_group_cv: preallocated OOF array,
            # fold-local pipeline fit, write-back at original indices.
            n = len(X)
            oof = np.full(n, np.nan)

            for train_idx, test_idx in splits:
                steps = [(name, clone(step)) for name, step in preprocessing]
                pipe = Pipeline([*steps, ("model", clone(model))])
                pipe.fit(X.iloc[train_idx], y_perm[train_idx])
                oof[test_idx] = pipe.predict(X.iloc[test_idx])
            # The null is the CV score OF the permuted problem: score the
            # out-of-fold predictions against this permutation's own
            # (permuted) targets. Scoring against the original y instead
            # would measure residual label agreement, not the null
            # distribution, and invalidate every p-value (regression test:
            # tests/test_permutation.py::test_permutation_null_scores_...
            # against_permuted_targets).
            perm_metrics = compute_regression_metrics(y_perm, oof)
            for metric in metrics:
                nulls[model_name][metric].append(float(perm_metrics[metric]))

    results: dict[str, dict[str, PermutationMetricResult]] = {}
    for result in observed_results:
        per_metric: dict[str, PermutationMetricResult] = {}
        for metric in metrics:
            null = np.asarray(nulls[result.model_name][metric], dtype=float)
            per_metric[metric] = PermutationMetricResult(
                metric=metric,
                direction=METRIC_DIRECTIONS[metric],
                observed=float(result.metrics[metric]),
                p_value=permutation_p_value(
                    float(result.metrics[metric]), null, METRIC_DIRECTIONS[metric],
                    n_permutations,
                ),
                null_mean=float(null.mean()),
                null_median=float(np.median(null)),
                n_permutations=n_permutations,
                seed=seed,
                null=null,
            )
        results[result.model_name] = per_metric

    return results


def permutation_caveat(n_subjects: int) -> str:
    """Interpretation caveat for permutation p-values (SCI-010).

    Descriptive only at pilot sizes; still hypothesis-generating near the
    kinematic cohort's ~31 recordings.
    """
    if n_subjects < 20:
        return (
            f"WARNING: permutation p-values from n={n_subjects} subjects are DESCRIPTIVE "
            "only — they summarize how unusual the observed metric is against this "
            "cohort's own label-permutation null, and cannot support a confirmatory "
            "claim. At this size a single subject can dominate the result."
        )
    return (
        f"CAUTION: at n={n_subjects} subjects, permutation results remain "
        "hypothesis-generating rather than confirmatory; interpretation must be "
        "qualified accordingly (small-cohort study)."
    )

#!/usr/bin/env python3
"""Generate the README figures.

Three figures, each stating its data provenance, cohort size, and
evaluation method on the figure itself:

1. ``figures/pipeline_architecture.png`` — what the code actually does.
2. ``figures/cohort_modality_flow.png`` — cohort and modality structure.
3. ``figures/synthetic_oof_verification.png`` — correctly aligned OOF
   predictions from the SYNTHETIC demo run (labeled as verification,
   never as study results).

Deterministic: the underlying evaluation is the seeded synthetic demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from curve_icon_dbso import features as features_mod  # noqa: E402
from curve_icon_dbso import reporting  # noqa: E402
from curve_icon_dbso.config import DemoConfig  # noqa: E402
from curve_icon_dbso.evaluation import (  # noqa: E402
    baseline_models,
    default_models,
    run_group_cv,
)
from curve_icon_dbso.synthetic import generate_synthetic_cohort  # noqa: E402

# --- Palette (validated reference palette, light surface) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
C_RF = "#2a78d6"   # categorical slot 1
C_GB = "#eb6834"   # categorical slot 2
C_LR = "#1baf7a"   # categorical slot 3 (sub-3:1 on light -> always direct-labeled)
C_BASE = "#898781"

MODEL_COLORS = {"Random Forest": C_RF, "Gradient Boosting": C_GB, "Linear Regression": C_LR}

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": BASELINE_AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def run_synthetic_evaluation(config: DemoConfig | None = None):
    """Run the seeded synthetic demo evaluation in-memory (no files written)."""
    config = config or DemoConfig()
    cohort = generate_synthetic_cohort(n_subjects=config.n_subjects, seed=config.seed)
    enriched = features_mod.add_clinical_composites(cohort)
    x, y, groups = features_mod.build_model_matrix(enriched)
    models = {**baseline_models(), **default_models(config.models)}
    results = run_group_cv(x, y.to_numpy(), groups.to_numpy(), models)
    return results, int(groups.nunique()), int(len(cohort))


# ============================================================================
# Figure 1: pipeline architecture
# ============================================================================

def box(ax, x, y, w, h, title, lines, accent=C_RF, title_size=9.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor="white", edgecolor=GRID, linewidth=1.2))
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.012), w, 0.012, boxstyle="round,pad=0.0,rounding_size=0.006",
        facecolor=accent, edgecolor="none"))
    ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=INK)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + h - 0.10, body, ha="center", va="top",
            fontsize=8.2, color=INK_2, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=BASELINE_AXIS))


def figure_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    ax.text(0.02, 0.97, "Pipeline architecture — what the code actually computes",
            fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.915,
            "Exploratory research prototype · not for clinical use · every output is provenance-labeled",
            fontsize=9, color=INK_2, va="top")

    row_y, box_h = 0.66, 0.22
    box(ax, 0.02, row_y, 0.26, box_h, "Inputs (either, explicitly labeled)",
        ["• deterministic SYNTHETIC cohort", "  (8 subjects · 9 recordings · seed 42)",
         "• authorized user's local data", "  (never committed — docs/data_access.md)"])

    box(ax, 0.37, row_y, 0.28, box_h, "Feature construction",
        ["• UPDRS motor composites +", "  asymmetry indices (validated columns)",
         "• HFO: Welch band power + peak", "  freq, per-channel QC flags",
         "• prespecified feature set (7) —", "  no data-driven selection"])

    box(ax, 0.74, row_y, 0.24, box_h, "Design matrix",
        ["X (7 prespecified features)", "y — UPDRS improvement (%)",
         "groups — subject ID", "(subjects are the evaluation", "units, not rows)"])

    row2_y, box_h2 = 0.30, 0.24
    box(ax, 0.02, row2_y, 0.30, box_h2, "Leave-one-subject-out CV",
        ["one fold per subject: all recordings", "of a subject held out together",
         "sklearn Pipeline per fold:", "median-impute → standardize → model",
         "(preprocessing fit on TRAIN fold only)"], accent=C_GB)

    box(ax, 0.41, row2_y, 0.26, box_h2, "Aligned OOF predictions",
        ["oof[test_idx] = predict(X_test)", "preallocated array, original",
         "row indices — regression-tested", "with nonsequential folds"])

    box(ax, 0.76, row2_y, 0.22, box_h2, "Metrics + baselines",
        ["MAE · RMSE · R² · Pearson r", "vs DummyRegressor mean/median",
         "per-subject OOF table", "small-sample warning (n=8)"], accent=C_GB)

    out_y = 0.02
    box(ax, 0.28, out_y, 0.44, 0.16, "Provenance-labeled outputs",
        ["metrics.csv · per-subject OOF · summary.json",
         "data_provenance: synthetic | authorized-local",
         "figures: labeled real / historical / synthetic"], accent=MUTED)

    arrow(ax, 0.155, row_y, 0.155, row2_y + box_h2)          # inputs -> CV
    arrow(ax, 0.32, row_y + 0.11, 0.37, row_y + 0.11)        # inputs -> features
    arrow(ax, 0.65, row_y + 0.11, 0.74, row_y + 0.11)        # features -> design
    arrow(ax, 0.86, row_y, 0.86, row2_y + box_h2)            # design -> (feeds CV too)
    arrow(ax, 0.51, row2_y, 0.41, row2_y)                    # CV -> OOF
    arrow(ax, 0.67, row2_y + 0.12, 0.76, row2_y + 0.12)      # OOF -> metrics
    arrow(ax, 0.60, row2_y, 0.50, out_y + 0.16)              # metrics -> outputs

    ax.text(0.02, 0.245,
            "Removed from the historical pipeline: full-dataset feature selection, "
            "full-dataset scaling/imputation,\nrandom HFO placeholder fallback, "
            "misaligned saved predictions, PDP 'confidence bands', mixed-effects claims.",
            fontsize=8, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Figure 2: cohort / modality flow
# ============================================================================

def figure_cohort_flow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    ax.text(0.02, 0.97, "Cohort and modality flow", fontsize=13,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.915,
            "Historical pilot structure as recorded in tracked artifacts — counts unverified against the source data",
            fontsize=9, color=INK_2, va="top")

    y0, bh = 0.60, 0.24
    box(ax, 0.02, y0, 0.28, bh, "Restricted DABI/USC data",
        ["UPDRS assessments (pre/post)", "ECoG recordings + metadata",
         "Cortical thickness (MRI)", "HFO features (channel level)"],
        accent=MUTED)
    box(ax, 0.36, y0, 0.26, bh, "Modality merge on subject",
        ["inner join, complete cases", "excludes subjects missing",
         "any modality"], accent=C_RF)
    box(ax, 0.68, y0, 0.30, bh, "Analysis cohort",
        ["8 unique subjects · 9 recordings",
         "one subject contributes 2 recordings", "(REST + MOVE states)"], accent=C_RF)

    y1 = 0.16
    box(ax, 0.36, y1, 0.26, 0.20, "Evaluation",
        ["leave-one-subject-out CV", "7 prespecified features",
         "fold-local preprocessing"], accent=C_GB)
    box(ax, 0.68, y1, 0.30, 0.20, "Reported, always labeled",
        ["subject + recording counts", "MAE/RMSE/R²/r vs baselines",
         "per-subject OOF predictions"], accent=MUTED)

    arrow(ax, 0.30, y0 + 0.12, 0.36, y0 + 0.12)
    arrow(ax, 0.62, y0 + 0.12, 0.68, y0 + 0.12)
    arrow(ax, 0.83, y0, 0.83, y1 + 0.20)
    arrow(ax, 0.62, y1 + 0.10, 0.68, y1 + 0.10)

    ax.text(0.02, 0.40,
            "Privacy: no participant data, subject identifiers, or per-subject\n"
            "outputs are committed. Data stays with authorized users\n"
            "(docs/data_access.md).",
            fontsize=8.4, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Figure 3: synthetic OOF verification
# ============================================================================

def figure_oof_verification(results, n_subjects: int, n_recordings: int, out: Path) -> None:
    order = ["Mean baseline", "Linear Regression", "Random Forest", "Gradient Boosting"]
    by_name = {r.model_name: r for r in results}

    y_true = None
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.1), dpi=200, sharex=True, sharey=True)
    for ax, name in zip(axes, order, strict=True):
        result = by_name[name]
        y_true = result.per_subject["y_true"].to_numpy()
        y_pred = result.per_subject["y_pred"].to_numpy()

        is_baseline = "baseline" in name.lower()
        color = C_BASE if is_baseline else MODEL_COLORS[name]
        ax.scatter(y_true, y_pred, s=70, color=color, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        lo = min(y_true.min(), y_pred.min()) - 2
        hi = max(y_true.max(), y_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.6,
                color=BASELINE_AXIS, zorder=2)

        m = result.metrics
        ax.set_title(name, fontsize=10.5, fontweight="bold",
                     color=INK, loc="left")
        ax.text(0.03, 0.97, f"MAE {m['MAE']:.2f}\nRMSE {m['RMSE']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
        ax.set_xlabel("True improvement (%)", fontsize=9)

    axes[0].set_ylabel("Out-of-fold prediction (%)", fontsize=9)
    fig.suptitle(
        "SYNTHETIC DATA — pipeline verification only, not study results",
        fontsize=12, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, 0.955,
             f"{n_subjects} synthetic subjects · {n_recordings} synthetic recordings · "
             "leave-one-subject-out · preprocessing fit per training fold · seed 42",
             ha="center", fontsize=9, color=INK_2)
    fig.text(0.5, -0.02,
             "Each point is one held-out synthetic subject (mean over its recordings). "
             "Dashed line: perfect prediction.\nBaseline predicts the training-fold mean "
             "for every held-out subject. Values are generated, not measured.",
             ha="center", fontsize=8.2, color=MUTED)

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    results, n_subjects, n_recordings = run_synthetic_evaluation()
    print(reporting.format_summary(results, n_subjects, n_recordings,
                                   reporting.PROVENANCE_SYNTHETIC))

    figure_architecture(figures_dir / "pipeline_architecture.png")
    figure_cohort_flow(figures_dir / "cohort_modality_flow.png")
    figure_oof_verification(results, n_subjects, n_recordings,
                            figures_dir / "synthetic_oof_verification.png")
    print(f"\nFigures written to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

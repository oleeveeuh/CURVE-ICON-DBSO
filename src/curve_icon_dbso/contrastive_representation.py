"""OPTIONAL contrastive (CEBRA-style) representation for window-level
ECoG — exploratory addendum to the frozen follow-on protocol (Prompt 7),
implemented ONLY after the time-resolved decoders (Prompt 4) and the DMD
comparison (Prompt 5) are complete.

Hard discipline, all enforced structurally:

* **Window-level only.** The encoder operates on window feature vectors;
  there is NO interface to the subject-level table, so deep learning
  cannot touch the subject-level arms.
* **From scratch, outer-training subjects only.** Every representation
  is fitted on outer-training windows; held-out-subject signals and
  labels cannot influence normalization (train-fitted statistics),
  representation learning (fit sees train rows only), early stopping
  (fixed epoch budget — no validation-based stopping), or
  hyperparameter choice (the encoder hyperparameters are FROZEN, not
  tuned; the downstream Ridge uses the frozen 7-point alpha grid with
  the existing inner selection).
* **No new dependency.** Neither torch nor cebra is present in this
  environment; the frozen low-capacity encoder is a deterministic NumPy
  InfoNCE implementation (equivalent per the prompt's "CEBRA-style or
  equivalent"). Architecture and grid are capped — the planned contract
  addendum (1.1.0, additive minor; the bump lands only after the v1.0.0
  real-data integration completes so the two arms of one integration
  never mix contract versions) records: hidden 32 (ReLU), embedding 8,
  temperature 0.1, weight decay 1e-4, deterministic mini-batch SGD
  (lr 0.05, batch 256), max epochs 200, seeds {42, 1, 2},
  learning-curve durations {50, 200} epochs, and the deferred rule
  below. The downstream decoder stays on the frozen Ridge grid.
* **Controls**: shuffled-label (temporal-neighbor pairing destroyed by a
  deterministic permutation) and shuffled-temporal-alignment (positives
  shifted ≥ the MOVE minimum shift). Seed variability
  {42, 1, 2} and learning curves by training duration are reported with
  runtime.
* **Deferred rule**: windows are NOT independent subjects. If the
  smallest outer-training subject budget is below the frozen floor
  (12), the experiment is DEFERRED — a ``deferred: True`` result is
  returned instead of forced numbers.
* **Gradient correctness**: the InfoNCE gradients here are the standard
  tied-encoder form — with P = softmax(z_a z_pᵀ / τ), the anchor-side
  embedding gradient is (P − I) z_p / (τ·B) and the positive-side
  gradient is (P − I)ᵀ z_a / (τ·B); both backpropagate through the
  shared two-layer encoder, with weight decay on both layers. A
  hand-checked numeric-gradient test pins this (see
  ``tests/test_contrastive_representation.py``).

Verification: deterministic synthetic fixtures only (temporal-structure
learning, control degradation, leakage isolation, determinism, gradient
check, seed variability, runtime). No committed real embeddings, no AWS,
no commits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import protocol
from .errors import DataValidationError

# Frozen addendum caps (planned contract 1.1.0 additive section; enforced
# here so the module can never silently exceed them).
CONTRASTIVE_HIDDEN: int = 32
CONTRASTIVE_EMBEDDING: int = 8
CONTRASTIVE_TEMPERATURE: float = 0.1
CONTRASTIVE_WEIGHT_DECAY: float = 1e-4
CONTRASTIVE_LEARNING_RATE: float = 0.05
CONTRASTIVE_BATCH_SIZE: int = 256
CONTRASTIVE_MAX_EPOCHS: int = 200
CONTRASTIVE_SEEDS: tuple[int, ...] = (42, 1, 2)
CONTRASTIVE_LEARNING_CURVE_EPOCHS: tuple[int, ...] = (50, 200)
#: Minimum outer-training subjects for a from-scratch fit (below this the
#: experiment is DEFERRED, not forced).
CONTRASTIVE_MIN_TRAIN_SUBJECTS: int = 12


def temporal_pairs(groups: np.ndarray, recording: np.ndarray,
                   target_bin: np.ndarray) -> np.ndarray:
    """Positive-pair index array (2 × n_pairs): consecutive update bins
    within the same recording block — the temporal-neighbor definition.
    Deterministic; uses positions only, never identities."""
    pairs: list[tuple[int, int]] = []
    frame_index = pd.DataFrame({"group": groups, "rec": recording,
                                "bin": target_bin})
    for _, rows in frame_index.groupby(["group", "rec"], sort=True):
        ordered = rows.sort_values("bin")
        indices = ordered.index.to_numpy()
        bins = ordered["bin"].to_numpy()
        for position in range(indices.size - 1):
            if bins[position + 1] == bins[position] + 1:
                pairs.append((int(indices[position]),
                              int(indices[position + 1])))
    if not pairs:
        raise DataValidationError(
            "no temporal-neighbor pairs found; the contrastive control "
            "cannot be defined on this dataset"
        )
    return np.asarray(pairs, dtype=int).T  # (2 × n_pairs) as documented


@dataclass
class ContrastiveResult:
    """One fitted encoder's evaluation on identical held-out folds."""

    seed: int
    epochs: int
    control: str
    per_subject: pd.DataFrame
    summary: dict
    train_loss_curve: list[float] = field(default_factory=list)
    fit_seconds: float = 0.0


class TemporalContrastiveEncoder:
    """Frozen low-capacity InfoNCE encoder (deterministic NumPy SGD).

    Architecture (frozen caps): input → hidden 32 (ReLU) → embedding 8,
    temperature 0.1, weight decay 1e-4, lr 0.05, batch 256, fixed epoch
    budget (no early stopping). ``fit`` sees outer-training windows ONLY;
    normalization statistics are train-fitted and reused at transform, so
    held-out data cannot influence anything.
    """

    def __init__(self, hidden: int = CONTRASTIVE_HIDDEN,
                 embedding: int = CONTRASTIVE_EMBEDDING,
                 temperature: float = CONTRASTIVE_TEMPERATURE,
                 weight_decay: float = CONTRASTIVE_WEIGHT_DECAY,
                 learning_rate: float = CONTRASTIVE_LEARNING_RATE,
                 batch_size: int = CONTRASTIVE_BATCH_SIZE,
                 epochs: int = CONTRASTIVE_MAX_EPOCHS,
                 seed: int = 42,
                 control: str = "none") -> None:
        if control not in ("none", "shuffled_label", "shifted_alignment"):
            raise DataValidationError(
                f"unknown contrastive control '{control}'"
            )
        self.hidden = int(hidden)
        self.embedding = int(embedding)
        self.temperature = float(temperature)
        self.weight_decay = float(weight_decay)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.control = control
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w1_: np.ndarray | None = None
        self.w2_: np.ndarray | None = None
        self.loss_curve_: list[float] = []
        self.fit_seconds_: float = 0.0

    def _forward(self, X_norm: np.ndarray) -> np.ndarray:
        assert self.w1_ is not None and self.w2_ is not None
        hidden = np.maximum(X_norm @ self.w1_, 0.0)
        return hidden @ self.w2_

    def fit(self, X: np.ndarray, pairs: np.ndarray) -> TemporalContrastiveEncoder:
        """Fit on outer-training windows ONLY (``pairs`` indexes rows of
        ``X``). Deterministic given the seed."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.shape[1] < 8:
            raise DataValidationError(
                "contrastive fit needs at least 8 temporal-neighbor pairs"
            )
        if pairs.max() >= n:
            raise DataValidationError("pair indices outside the window set")
        if not np.isfinite(X).all():
            raise DataValidationError("contrastive fit needs finite features")
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        X_norm = (X - self.mean_) / self.std_
        rng = np.random.default_rng(self.seed)
        self.w1_ = rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden))
        self.w2_ = rng.normal(0, np.sqrt(2.0 / self.hidden),
                              (self.hidden, self.embedding))
        anchor, positive = pairs[0], pairs[1]
        if self.control == "shuffled_label":
            positive = rng.permutation(positive)
        elif self.control == "shifted_alignment":
            shift = max(int(round(
                protocol.MOVE_CIRCULAR_SHIFT_MIN_S / protocol.MOVE_UPDATE_S
            )), 1)
            positive = np.roll(positive, shift)
        self.loss_curve_ = []
        started = time.perf_counter()
        for _epoch in range(self.epochs):
            order = rng.permutation(anchor.size)
            epoch_losses: list[float] = []
            for start in range(0, anchor.size, self.batch_size):
                batch = order[start:start + self.batch_size]
                ia, ip = anchor[batch], positive[batch]
                Xa, Xp = X_norm[ia], X_norm[ip]
                Ha = np.maximum(Xa @ self.w1_, 0.0)
                Hp = np.maximum(Xp @ self.w1_, 0.0)
                Za, Zp = Ha @ self.w2_, Hp @ self.w2_
                Za_n = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12)
                Zp_n = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12)
                logits = Za_n @ Zp_n.T / self.temperature
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probabilities = exp_logits / exp_logits.sum(axis=1,
                                                            keepdims=True)
                loss = float(np.mean(
                    -np.log(np.diag(probabilities) + 1e-12)
                ))
                epoch_losses.append(loss)
                # Tied-encoder InfoNCE gradients WITH the L2-normalization
                # Jacobian: with c_a = (P − I)·Ẑp/(B·τ) collecting the
                # softmax coefficients on the normalized anchor embeddings,
                # dL/dz = (c − (c·ẑ)ẑ)/||z|| (numeric-gradient test pins
                # this); the 1/||z|| appears ONLY here.
                n_a = np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12
                n_p = np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12
                c_a = (probabilities - np.eye(batch.size)) @ Zp_n / (
                    batch.size * self.temperature
                )
                g_a = (c_a - np.sum(c_a * Za_n, axis=1, keepdims=True)
                       * Za_n) / n_a
                # The delta term: q_ij = p_ij − δ_ij subtracts the
                # anchor's own normalized embedding on the positive side.
                c_p = (probabilities.T @ Za_n
                       - Za_n) / (batch.size * self.temperature)
                g_p = (c_p - np.sum(c_p * Zp_n, axis=1, keepdims=True)
                       * Zp_n) / n_p
                grad_Za, grad_Zp = g_a, g_p
                grad_W2 = Ha.T @ grad_Za + Hp.T @ grad_Zp
                grad_W2 += self.weight_decay * self.w2_
                grad_Ha = grad_Za @ self.w2_.T * (Ha > 0)
                grad_Hp = grad_Zp @ self.w2_.T * (Hp > 0)
                grad_W1 = Xa.T @ grad_Ha + Xp.T @ grad_Hp
                grad_W1 += self.weight_decay * self.w1_
                self.w2_ = self.w2_ - self.learning_rate * grad_W2
                self.w1_ = self.w1_ - self.learning_rate * grad_W1
            self.loss_curve_.append(float(np.mean(epoch_losses)))
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed rows with the FROZEN train-fitted statistics and weights
        (held-out data cannot change anything)."""
        if self.w1_ is None or self.mean_ is None:
            raise DataValidationError("transform before fit")
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        embedding = self._forward(X_norm)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
        return embedding / norms


def _per_subject_pearson(y: np.ndarray, pred: np.ndarray,
                         groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for subject in sorted(set(groups.tolist())):
        mask = groups == subject
        a, b = y[mask], pred[mask]
        r = None
        if a.size >= 2 and np.std(a) > 0 and np.std(b) > 0:
            r = float(np.corrcoef(a, b)[0, 1])
        rows.append({"subject_position": subject,
                     "n_windows": int(mask.sum()),
                     "pearson_r": r,
                     "mae": float(np.mean(np.abs(b - a)))})
    return pd.DataFrame(rows)


def _summarize(per_subject: pd.DataFrame) -> dict:
    valid = per_subject["pearson_r"].dropna().to_numpy()
    return {
        "median_pearson_r": float(np.median(valid)) if valid.size else None,
        "median_mae": float(np.median(per_subject["mae"])),
        "n_subjects": int(per_subject.shape[0]),
    }


def run_contrastive_comparison(
    dataset,
    model_name: str = "lagged_ridge_primary",
    seeds: tuple[int, ...] = CONTRASTIVE_SEEDS,
    curve_epochs: tuple[int, ...] = CONTRASTIVE_LEARNING_CURVE_EPOCHS,
    defer_below_train_subjects: int = CONTRASTIVE_MIN_TRAIN_SUBJECTS,
) -> dict:
    """Frozen comparison on IDENTICAL outer subject folds: contrastive
    embedding + frozen Ridge vs conventional multiband power — per
    Prompt 7, plus both controls, the seed table, and the learning
    curve. Per outer fold the encoder is fitted on OUTER-TRAINING
    windows only, then embeds both sides; the downstream Ridge runs the
    frozen alpha grid through the existing inner selection on embedded
    training data. Deferred rule: below the frozen outer-training
    subject floor the experiment is DEFERRED, not forced (windows are
    never counted as subjects).
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from .evaluation import _default_splitter, make_preprocessing
    from .move_modeling import _build_move_estimator, _select_move_params

    X = dataset.block_subset("multiband_power")
    y = dataset.target
    groups = dataset.groups
    n_subjects = len(np.unique(groups))
    splits = list(_default_splitter(n_subjects).split(X, y, groups))
    smallest_train = min(len(np.unique(groups[train])) for train, _ in splits)
    if smallest_train < defer_below_train_subjects:
        return {
            "deferred": True,
            "reason": (
                "outer-training subject budget below the frozen "
                "from-scratch floor; windows are not independent subjects"
            ),
            "n_subjects": n_subjects,
            "smallest_outer_train_subjects": int(smallest_train),
        }

    recording = dataset.metadata["recording_position"].to_numpy()
    target_bin = dataset.metadata["target_bin"].to_numpy()
    pairs = temporal_pairs(groups, recording, target_bin)

    def decode_fold(train_idx: np.ndarray, test_idx: np.ndarray,
                    encoder_seed: int, epochs: int,
                    control: str) -> tuple[np.ndarray, float, list[float]]:
        # Pairs reference GLOBAL row indices; the fold encoder sees the
        # re-indexed training subset, so pairs are filtered to
        # both-endpoints-in-training and remapped to local positions.
        train_position = {int(g): i for i, g in enumerate(train_idx)}
        local_pairs = np.asarray([
            (train_position[int(a)], train_position[int(b)])
            for a, b in pairs.T
            if int(a) in train_position and int(b) in train_position
        ], dtype=int).T
        encoder = TemporalContrastiveEncoder(epochs=epochs,
                                             seed=encoder_seed,
                                             control=control)
        encoder.fit(X.iloc[train_idx].to_numpy(), local_pairs)
        embedded_train = encoder.transform(X.iloc[train_idx].to_numpy())
        embedded_test = encoder.transform(X.iloc[test_idx].to_numpy())
        embedded_train_frame = pd.DataFrame(embedded_train)
        params = _select_move_params(model_name, embedded_train_frame,
                                     y[train_idx], groups[train_idx],
                                     seed=encoder_seed)
        steps = [(name, clone(step)) for name, step in make_preprocessing()]
        pipe = Pipeline([*steps, ("model", _build_move_estimator(
            model_name, params, seed=encoder_seed))])
        pipe.fit(embedded_train_frame, y[train_idx])
        pred = pipe.predict(pd.DataFrame(embedded_test))
        return pred, encoder.fit_seconds_, encoder.loss_curve_

    max_epochs = max(curve_epochs)
    results: dict = {"deferred": False, "runs": {}, "seed_table": [],
                     "learning_curve": [], "controls": {}, "runtime": {}}
    for encoder_seed in seeds:
        oof = np.full(y.size, np.nan)
        total_fit = 0.0
        for train_idx, test_idx in splits:
            pred, fit_seconds, _ = decode_fold(
                train_idx, test_idx, encoder_seed, max_epochs, "none"
            )
            oof[test_idx] = pred
            total_fit += fit_seconds
        per_subject = _per_subject_pearson(y, oof, groups)
        summary = _summarize(per_subject)
        results["runs"][f"seed{encoder_seed}"] = {
            "per_subject": per_subject, "summary": summary,
        }
        results["seed_table"].append({"seed": encoder_seed, **summary})
        results["runtime"][f"seed{encoder_seed}_fit_seconds"] = total_fit
    for control in ("shuffled_label", "shifted_alignment"):
        oof = np.full(y.size, np.nan)
        for train_idx, test_idx in splits:
            pred, _, _ = decode_fold(train_idx, test_idx, seeds[0],
                                     max_epochs, control)
            oof[test_idx] = pred
        results["controls"][control] = _summarize(
            _per_subject_pearson(y, oof, groups)
        )
    for epochs in curve_epochs:
        oof = np.full(y.size, np.nan)
        for train_idx, test_idx in splits:
            pred, _, _ = decode_fold(train_idx, test_idx, seeds[0], epochs,
                                     "none")
            oof[test_idx] = pred
        results["learning_curve"].append({
            "epochs": epochs,
            **_summarize(_per_subject_pearson(y, oof, groups)),
        })
    return results

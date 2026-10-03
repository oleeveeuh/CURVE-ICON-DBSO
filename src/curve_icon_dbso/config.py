"""Configuration handling.

Analysis parameters live in a small dataclass so that every run is
explicitly parameterized: no hardcoded personal paths, no hidden
defaults that change behavior between runs. Paths to private data are
supplied by the user at run time (CLI args or a YAML config file); the
repository itself contains none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import DataValidationError


@dataclass(frozen=True)
class HFOConfig:
    """HFO extraction parameters.

    Attributes:
        low_hz: Lower edge of the HFO band in Hz.
        high_hz: Upper edge of the HFO band in Hz. Must be strictly below
            the Nyquist frequency of the recording.
    """

    low_hz: float = 80.0
    high_hz: float = 500.0

    def __post_init__(self) -> None:
        if not (0.0 < self.low_hz < self.high_hz):
            raise DataValidationError(
                f"HFO band is invalid: need 0 < low < high, got "
                f"low={self.low_hz}, high={self.high_hz}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Fixed, documented model hyperparameters.

    Hyperparameters are fixed a priori and are NOT tuned against the
    evaluation subjects. With eight subjects there is no way to tune
    hyperparameters without leaking information; see docs/methodology.md.
    """

    rf_n_estimators: int = 200
    rf_max_depth: int = 3
    gbr_n_estimators: int = 50
    gbr_max_depth: int = 2
    gbr_learning_rate: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the deterministic synthetic end-to-end demo."""

    seed: int = 42
    n_subjects: int = 8
    output_dir: Path = field(default_factory=lambda: Path("outputs/synthetic_demo"))
    hfo: HFOConfig = field(default_factory=HFOConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if self.n_subjects < 2:
            raise DataValidationError(
                f"n_subjects must be >= 2 for group-wise evaluation, got {self.n_subjects}"
            )


def load_config(path: str | Path) -> DemoConfig:
    """Load a :class:`DemoConfig` from a YAML file.

    Unknown keys are rejected so that typos fail loudly instead of being
    silently ignored.

    Raises:
        DataValidationError: If the file cannot be parsed or contains
            unknown keys.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - trivial
        raise DataValidationError(f"Could not parse YAML config {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise DataValidationError(f"Config root must be a mapping, got {type(raw).__name__}")

    known = {"seed", "n_subjects", "output_dir", "hfo", "models"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise DataValidationError(
            f"Unknown config keys in {path}: {unknown}. Allowed keys: {sorted(known)}"
        )

    hfo_raw = raw.get("hfo", {})
    models_raw = raw.get("models", {})
    for section, section_keys in (("hfo", hfo_raw), ("models", models_raw)):
        if not isinstance(section_keys, dict):
            raise DataValidationError(f"Config section '{section}' must be a mapping")
    allowed_hfo = {"low_hz", "high_hz"}
    unknown_hfo = sorted(set(hfo_raw) - allowed_hfo)
    if unknown_hfo:
        raise DataValidationError(f"Unknown keys in 'hfo' section: {unknown_hfo}")
    allowed_models = {
        "rf_n_estimators",
        "rf_max_depth",
        "gbr_n_estimators",
        "gbr_max_depth",
        "gbr_learning_rate",
        "seed",
    }
    unknown_models = sorted(set(models_raw) - allowed_models)
    if unknown_models:
        raise DataValidationError(f"Unknown keys in 'models' section: {unknown_models}")

    return DemoConfig(
        seed=int(raw.get("seed", 42)),
        n_subjects=int(raw.get("n_subjects", 8)),
        output_dir=Path(raw.get("output_dir", "outputs/synthetic_demo")),
        hfo=HFOConfig(**hfo_raw),
        models=ModelConfig(**models_raw),
    )

"""Explicit exception types used across the package.

The historical implementation failed silently in several places (broad
``except`` blocks, zero-filled fallbacks, randomly generated placeholder
features). Every failure mode here is meant to be loud and actionable.
"""


class CurveIconError(Exception):
    """Base class for all package-specific errors."""


class DataValidationError(CurveIconError):
    """Raised when input tables are malformed or missing required columns."""


class SamplingRateError(CurveIconError):
    """Raised for invalid or physiologically impossible sampling rates."""


class MissingDataError(CurveIconError):
    """Raised when required data files or directories are absent.

    The package never substitutes synthetic or random values for missing
    real data. Callers who want a synthetic run must request it explicitly
    via :mod:`curve_icon_dbso.synthetic`.
    """


class ArtifactError(CurveIconError):
    """Raised when a model artifact package is missing, corrupt, or
    incompatible (hash mismatch, schema drift, library-version drift).

    The API fails readiness loudly on this error instead of serving a
    model it cannot vouch for (MOD-003). ``error_class`` is a stable,
    machine-readable category (``artifact_missing``, ``artifact_corrupt``,
    ``artifact_incompatible``, ``artifact_invalid``) for metadata-only
    logs and error responses — never the offending values themselves.
    """

    def __init__(self, message: str, *, error_class: str = "artifact_invalid") -> None:
        super().__init__(message)
        self.error_class = error_class

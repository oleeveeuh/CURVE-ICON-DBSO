"""Pydantic v2 request/response models, generated from the canonical schema.

The row model is **built** from :mod:`curve_icon_dbso.schema` at import
time (feature names, numeric limits, batch bounds) so the API cannot
drift from the training/manifest schema — DAT-001's "one definition".
Validation policy (DAT-003, API-004/005/006):

* exactly the seven schema features, strict JSON numbers — strings and
  booleans rejected, NaN/Infinity rejected, integers accepted;
* documented limits enforced per feature;
* ``extra="forbid"`` everywhere — unknown fields (including subject
  identifiers, which the schema has no field for) are 422s, not
  silently dropped data (API-006);
* batch size 1..256 enforced by the model itself;
* complete rows only — the service does not impute (see schema.py).

Error responses never echo payload values: the handlers in
``api.main`` forward only field paths and stable error codes, never the
pydantic ``input``/``msg`` fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from .. import schema as schema_mod

# Fail at import time if the limits table ever drifts from the feature list.
schema_mod.validate_limits()


def build_row_model() -> type[BaseModel]:
    """Build the strict feature-row model from the canonical schema."""
    fields: dict[str, tuple[type, Any]] = {}
    for name in schema_mod.FEATURES:
        low, high = schema_mod.FEATURE_LIMITS[name]
        fields[name] = (
            float,
            Field(
                strict=True,          # JSON numbers only: no "3.5" strings, no bools
                allow_inf_nan=False,  # NaN/Infinity rejected (DAT-003)
                ge=low,
                le=high,
                description=f"Finite number in [{low}, {high}] (schema "
                            f"v{schema_mod.SCHEMA_VERSION} limit)",
            ),
        )
    return create_model(
        "FeatureRow",
        __config__=ConfigDict(extra="forbid"),  # unknown fields / identifiers rejected
        __doc__=(
            "One complete feature row (schema v"
            f"{schema_mod.SCHEMA_VERSION}). Exactly the "
            f"{len(schema_mod.FEATURES)} prespecified features; no subject "
            "identifiers; no missing values — the service does not impute."
        ),
        **fields,
    )


FeatureRow = build_row_model()


class PredictRequest(BaseModel):
    """``POST /predict`` body: an ordered batch of 1..256 complete rows."""

    model_config = ConfigDict(extra="forbid")

    rows: list[FeatureRow] = Field(
        min_length=schema_mod.BATCH_MIN_ROWS,
        max_length=schema_mod.BATCH_MAX_ROWS,
        description=(
            f"Ordered batch of {schema_mod.BATCH_MIN_ROWS}.."
            f"{schema_mod.BATCH_MAX_ROWS} feature rows; predictions are "
            "returned in this order"
        ),
    )


class PredictResponse(BaseModel):
    """``POST /predict`` response — predictions in input order."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[float]
    n_rows: int = Field(ge=schema_mod.BATCH_MIN_ROWS, le=schema_mod.BATCH_MAX_ROWS)
    model_version: str
    schema_version: str
    schema_hash: str


class HealthResponse(BaseModel):
    """``GET /health`` — liveness plus readiness state (API-002).

    Always 200 while the process is alive; readiness failures surface as
    ``ready: false`` plus a stable ``error_class`` — no infrastructure
    or data details.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = "alive"
    service: str
    package_version: str
    ready: bool
    model_version: str | None = None
    schema_version: str | None = None
    schema_hash: str | None = None
    research_only: bool = True
    not_for_clinical_use: bool = True
    error_class: str | None = None


class ReadyResponse(BaseModel):
    """``GET /ready`` — readiness-only probe (503 until the artifact is
    loaded and verified)."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    model_version: str | None = None
    schema_version: str | None = None
    error_class: str | None = None


class ErrorResponse(BaseModel):
    """Machine-readable error body: stable class codes, safe messages.

    ``errors.loc`` carries field paths (and batch positions) only —
    never payload values (API-005).
    """

    model_config = ConfigDict(extra="forbid")

    error_class: str
    message: str
    errors: list[dict] | None = None

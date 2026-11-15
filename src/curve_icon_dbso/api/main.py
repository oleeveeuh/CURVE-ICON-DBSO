"""The Phase 2 inference service: FastAPI + Pydantic v2 (API-001..010).

Endpoints (ADR-002 scope — no ``/explain``):

* ``GET /health``  — liveness plus readiness state; 200 while the
  process is alive, with ``ready`` and versions in the body (API-002).
* ``GET /ready``   — readiness-only; 503 until the artifact is loaded
  and fully verified. Container HEALTHCHECK target (CON-003).
* ``GET /model-info`` — the approved manifest fields, including
  ``intended_use`` and ``beats_baseline`` (API-003).
* ``POST /predict`` — ordered batches of 1..256 schema-valid rows;
  predictions returned in input order (API-004).

Hard rules enforced here:

* The artifact is loaded and hash-verified ONCE at startup (API-007);
  a missing/corrupt/incompatible artifact fails readiness loudly and
  permanently (MOD-003) — requests get 503, never a guess.
* Request/response bodies, predictions and feature values are NEVER
  logged (API-008). Logs carry exactly: request id, route, status,
  duration, row count, schema hash, model version, error class (API-009).
* Validation errors report field paths and stable error codes — never
  payload values (API-005).

Run locally::

    uvicorn curve_icon_dbso.api.main:app --port 8000 --no-access-log

The artifact directory comes from (first match wins): the
``artifact_dir`` argument, the ``CURVE_ARTIFACT_DIR`` environment
variable, or ``artifacts/synthetic-demo`` relative to the working
directory. Train it first with ``scripts/train_synthetic_model.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from .. import artifact as artifact_mod
from .. import schema as schema_mod
from ..errors import ArtifactError
from .schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

SERVICE_NAME = "curve-icon-dbso-api"
ARTIFACT_DIR_ENV = "CURVE_ARTIFACT_DIR"
DEFAULT_ARTIFACT_DIR = "artifacts/synthetic-demo"

# Transport-level cap. The largest valid body (256 rows x 7 features) is
# ~60 KB; anything past 1 MB is rejected before parsing.
MAX_BODY_BYTES = 1_000_000

LOG = logging.getLogger("curve_icon_dbso.api")

# ---------------------------------------------------------------- logging ---

_LOG_FORMAT = "%(asctime)s %(message)s"


def _configure_logging() -> None:
    """Give the service logger a stdout JSON-line handler (idempotent).

    Structured, metadata-only (API-009): the message is the JSON object,
    the framework prefix adds only the timestamp.
    """
    if not LOG.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False


def _emit(entry: dict[str, Any]) -> None:
    """One structured log line; callers pass metadata fields only."""
    LOG.info(json.dumps(entry, sort_keys=True))


class MetadataLoggingMiddleware:
    """Pure-ASGI request logging: the ONLY place requests are logged.

    Reads nothing from the body or the response payload. ``row_count``
    and ``error_class`` arrive via ``request.state`` set by the
    endpoints/handlers; schema hash and model version come from the
    loaded artifact (or ``None``).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status_holder: dict[str, int] = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode("ascii"))
                )
            await send(message)

        app_state = getattr(scope.get("app"), "state", None)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            request_state = scope.get("state") or {}
            _emit({
                "request_id": request_id,
                "route": scope.get("path", ""),
                "status": status_holder["status"],
                "duration_ms": duration_ms,
                "row_count": request_state.get("row_count"),
                "schema_hash": getattr(app_state, "schema_hash", None),
                "model_version": getattr(app_state, "model_version", None),
                "error_class": request_state.get("error_class"),
            })


class PayloadSizeLimitMiddleware:
    """Reject bodies over :data:`MAX_BODY_BYTES` before parsing (413)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            content_length = Headers(scope=scope).get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
                scope.setdefault("state", {})["error_class"] = "payload_too_large"
                response = JSONResponse(
                    status_code=413,
                    content=ErrorResponse(
                        error_class="payload_too_large",
                        message="request body exceeds the size limit",
                    ).model_dump(),
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ------------------------------------------------------------ error body ---

# pydantic error type -> (stable error_class, safe message). Messages are
# curated constants: no payload values, no pydantic "input"/"msg" echoes.
_VALIDATION_ERRORS: dict[str, tuple[str, str]] = {
    "missing": ("missing_field", "a required field is missing"),
    "extra_forbidden": ("unknown_field", "unknown fields are not allowed"),
    "float_type": ("invalid_type", "feature values must be JSON numbers"),
    "float_parsing": ("invalid_type", "feature values must be JSON numbers"),
    "int_type": ("invalid_type", "feature values must be JSON numbers"),
    "string_type": ("invalid_type", "feature values must be JSON numbers"),
    "bool_type": ("invalid_type", "feature values must be JSON numbers"),
    "finite_number": (
        "non_finite_value",
        "feature values must be finite (NaN/Infinity are not allowed)",
    ),
    "greater_than_equal": ("out_of_range", "value is outside the schema limit"),
    "less_than_equal": ("out_of_range", "value is outside the schema limit"),
    "too_short": ("empty_batch", f"batch must contain at least {schema_mod.BATCH_MIN_ROWS} row"),
    "too_long": ("batch_too_large", f"batch must contain at most {schema_mod.BATCH_MAX_ROWS} rows"),
    "list_type": ("invalid_type", "'rows' must be a JSON array"),
    "model_type": ("invalid_row", "each row must be a JSON object"),
    # FastAPI wraps the body model, so a non-object body/row surfaces at the
    # API layer as model_attributes_type; resolved by location below.
    "model_attributes_type": ("invalid_structure", "JSON object expected"),
    "json_invalid": ("malformed_json", "request body is not valid JSON"),
}
_VALIDATION_DEFAULT = ("invalid_value", "value failed schema validation")


def _validation_entries(exc: RequestValidationError) -> list[dict]:
    """Sanitized error entries: field path + class + message, nothing else."""
    entries = []
    for error in exc.errors():
        error_class, message = _VALIDATION_ERRORS.get(error["type"], _VALIDATION_DEFAULT)
        loc = [part if isinstance(part, int) else str(part) for part in error["loc"]]
        if error["type"] == "model_attributes_type":
            # Same pydantic type for the whole body and for a single row;
            # the location decides which stable class the caller sees.
            if len(loc) > 1 and loc[1] == "rows":
                error_class, message = "invalid_row", "each row must be a JSON object"
            else:
                error_class, message = (
                    "invalid_body",
                    "request body must be a JSON object with a 'rows' array",
                )
        entries.append({
            "loc": loc,
            "error_class": error_class,
            "message": message,
        })
    return entries


def _validation_error_class(entries: list[dict]) -> str:
    classes = {entry["error_class"] for entry in entries}
    return classes.pop() if len(classes) == 1 else "validation_error"


# ------------------------------------------------------------ application ---

def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    """Build the service. The artifact is loaded once, in the lifespan."""
    _configure_logging()
    resolved = Path(artifact_dir or os.environ.get(ARTIFACT_DIR_ENV, DEFAULT_ARTIFACT_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load_artifact_into(app)
        yield
        # No shutdown-time resource release: the model lives with the process.

    application = FastAPI(
        title=SERVICE_NAME,
        version=__version__,
        description=(
            "Research demonstration inference service for the CURVE-ICON-DBSO "
            "SYNTHETIC demonstration model. Research only; NOT for clinical "
            "use. The model is trained on deterministic synthetic data and "
            "supports no claim about real outcomes. There is no /explain "
            "endpoint (ADR-002)."
        ),
        lifespan=lifespan,
    )

    # Sane defaults even before (or without) lifespan startup: the service
    # comes up unready, never guessing (TestClient-without-lifespan, tests
    # of readiness failures).
    application.state.artifact_dir = str(resolved)
    application.state.model = None
    application.state.manifest = None
    application.state.ready = False
    application.state.load_error_class = "not_started"

    application.add_middleware(PayloadSizeLimitMiddleware)
    application.add_middleware(MetadataLoggingMiddleware)

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        entries = _validation_entries(exc)
        request.state.error_class = _validation_error_class(entries)
        status = 400 if any(e["error_class"] == "malformed_json" for e in entries) else 422
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                error_class=request.state.error_class,
                message="request failed schema validation; see errors for field paths",
                errors=entries,
            ).model_dump(),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        request.state.error_class = "internal_error"
        _emit({"error_class": "internal_error", "exception_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_class="internal_error",
                message="unexpected server error",
            ).model_dump(),
        )

    @application.get("/", tags=["meta"])
    async def root() -> dict:
        return {
            "service": SERVICE_NAME,
            "research_only": True,
            "not_for_clinical_use": True,
            "endpoints": ["/health", "/ready", "/model-info", "/predict"],
        }

    @application.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        state = application.state
        manifest = state.manifest if state.ready else None
        return HealthResponse(
            service=SERVICE_NAME,
            package_version=__version__,
            ready=bool(state.ready),
            model_version=manifest["model_version"] if manifest else None,
            schema_version=manifest["schema_version"] if manifest else None,
            schema_hash=manifest["schema_hash"] if manifest else None,
            error_class=state.load_error_class,
        )

    @application.get("/ready", response_model=ReadyResponse, tags=["meta"])
    async def ready() -> JSONResponse:
        state = application.state
        if state.ready and state.manifest is not None:
            body = ReadyResponse(
                ready=True,
                model_version=state.manifest["model_version"],
                schema_version=state.manifest["schema_version"],
            )
            return JSONResponse(status_code=200, content=body.model_dump())
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(ready=False, error_class=state.load_error_class).model_dump(),
        )

    @application.get("/model-info", tags=["meta"])
    async def model_info(request: Request):
        state = application.state
        if not _require_ready(request):
            return _not_ready_response(request)
        return JSONResponse(content=state.manifest)

    @application.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        responses={503: {"model": ErrorResponse, "description": "model artifact not ready"}},
    )
    async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        if not _require_ready(request):
            return _not_ready_response(request)
        state = application.state
        request.state.row_count = len(payload.rows)
        rows = [row.model_dump() for row in payload.rows]
        frame = pd.DataFrame(rows, columns=schema_mod.FEATURES)
        predictions = np.asarray(state.model.predict(frame), dtype=float)
        if not np.isfinite(predictions).all():  # defensive; cannot serve NaN output
            request.state.error_class = "internal_error"
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error_class="internal_error", message="unexpected server error"
                ).model_dump(),
            )
        return PredictResponse(
            predictions=[float(value) for value in predictions],
            n_rows=len(payload.rows),
            model_version=state.manifest["model_version"],
            schema_version=state.manifest["schema_version"],
            schema_hash=state.manifest["schema_hash"],
        )

    return application


def _require_ready(request: Request) -> bool:
    state = request.app.state
    return bool(state.ready) and state.model is not None and state.manifest is not None


def _not_ready_response(request: Request) -> JSONResponse:
    """503 with the stable load-failure class; no infrastructure detail."""
    request.state.error_class = request.app.state.load_error_class or "not_ready"
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error_class=request.state.error_class,
            message="model artifact is not loaded; the service is not ready",
        ).model_dump(),
    )


def _load_artifact_into(app: FastAPI) -> None:
    """Load and verify the artifact once (API-007 / MOD-003).

    Failures are recorded as a stable error class and logged metadata-only;
    the service STAYS UP so liveness/readiness can report the failure.
    """
    app_state = app.state
    try:
        model, manifest = artifact_mod.load_artifact(app_state.artifact_dir)
    except ArtifactError as exc:
        app_state.load_error_class = exc.error_class
        _emit({"error_class": exc.error_class})
        return
    except Exception as exc:  # noqa: BLE001 - any startup failure must be visible
        app_state.load_error_class = "artifact_load_failed"
        _emit({"error_class": "artifact_load_failed", "exception_type": type(exc).__name__})
        return
    app_state.model = model
    app_state.manifest = manifest
    app_state.ready = True
    app_state.load_error_class = None
    app_state.schema_hash = manifest["schema_hash"]
    app_state.model_version = manifest["model_version"]
    _emit({
        "schema_hash": manifest["schema_hash"],
        "schema_version": manifest["schema_version"],
        "model_version": manifest["model_version"],
    })


app = create_app()

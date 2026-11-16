# syntax=docker/dockerfile:1
#
# Phase 2 synthetic-track inference image (CON-001..003, CON-005, CON-006).
#
# Design notes:
# * Multi-stage: the builder resolves the committed uv.lock and trains the
#   deterministic synthetic artifact; the runtime stage copies ONLY the
#   finished virtualenv and the artifact — no uv, no build tooling, no
#   build context files beyond the package source (excluded via
#   .dockerignore: data, outputs, experiment stores, caches, tests, docs).
# * The model is TRAINED INSIDE the build from locked source. No model
#   bytes ever enter through the build context, so the image cannot
#   contain a real-data artifact by construction.
# * Locked environment: every dependency version comes from uv.lock
#   (`uv sync --locked`), the same lock local setup and CI use (REP-003).
# * Non-root runtime user (10001:10001), one uvicorn worker (API-010).
# * HEALTHCHECK targets /ready, the deployment readiness probe (CON-003).
#
# Immutable tagging (CON-006) is done at build time, not in this file:
#   docker build \
#     --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
#     -t curve-icon-dbso-api:$(git rev-parse --short=12 HEAD) .
# Mutable tags (`latest`) are convenience aliases only; deploy by digest.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder ---
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layers first: pyproject/lock changes invalidate them, source
# changes do not.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev --extra api

COPY src/ ./src/
# --no-editable is REQUIRED here: the default editable install points at
# /app/src, which the runtime stage does not copy (only .venv + artifact),
# so the editable image fails with ModuleNotFoundError at startup.
RUN uv sync --locked --no-dev --extra api --no-editable

# Deterministic synthetic training (schema v1.0.0, seed 42, 8 subjects).
# Byte-reproducible given the locked environment; the artifact manifest
# inside the image records its own hashes.
RUN .venv/bin/python -m curve_icon_dbso.training --output-dir /app/artifact

# ---------------------------------------------------------------- runtime ---
# Debian 13 (trixie) runtime: Debian 12 (bookworm) went oldstable and
# carries a flood of 2025/2026 CVEs with no bookworm fix. The venv's
# manylinux wheels are glibc-compatible with trixie.
#
# FAILURE_DEMO=1 builds the Phase 5 rollback-exercise image (owner-gated):
# it REMOVES the trained artifact directory, so the fail-closed readiness
# path (MOD-003) returns 503 forever and the deployment fails health
# checks — which is exactly what the ECS deployment circuit breaker and
# the unhealthy-target alarm are supposed to catch. Nothing else changes.
FROM python:${PYTHON_VERSION}-slim-trixie AS runtime

ARG FAILURE_DEMO=0

ARG SOURCE_SHA=unknown
LABEL org.opencontainers.image.title="curve-icon-dbso-api" \
      org.opencontainers.image.description="Synthetic-track research demonstration inference API. Research only; NOT for clinical use." \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.opencontainers.image.source="https://github.com/oleeveeuh/CURVE-ICON-DBSO"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CURVE_ARTIFACT_DIR=/app/artifact

# Non-root runtime user (API-010): fixed system UID/GID, no login shell,
# no home directory.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --from=builder --chown=10001:10001 /app/artifact /app/artifact

# Hardening: security patches for base packages, then remove the base
# image's global packaging tooling (pip/setuptools/wheel/jaraco.context).
# The app runs ONLY from /app/.venv (PATH below), so none of it is
# reachable at runtime — removing it fixes the CVEs it carries instead of
# documenting exceptions, and shrinks the image.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /usr/local/lib/python3.11/site-packages \
    /usr/local/lib/python3.11/__pycache__ \
    /usr/local/bin/pip* \
    /var/lib/apt/lists/*

# Phase 5 rollback-exercise hook (see FAILURE_DEMO above).
RUN if [ "$FAILURE_DEMO" = "1" ]; then rm -rf /app/artifact && printf '%s\n' \
      "deliberately failing deployment: Phase 5 rollback exercise" \
      > /failure-demo-marker; fi

USER 10001:10001

EXPOSE 8000

# Readiness-gated health check (CON-003): the container is healthy only
# once the artifact is loaded, hash-verified and serving.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"]

# One worker by default (API-010); more only after load testing justifies it.
CMD ["uvicorn", "curve_icon_dbso.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

"""Container build safety (CON-002) and a Docker-gated end-to-end run.

Two layers:

1. Always-on tests (no Docker needed): the .dockerignore must cover every
   prohibited category, the *effective build context* — the repository
   tree after .dockerignore filtering, approximated with a conservative
   pattern matcher — must contain no data/private file types, and the
   Dockerfile must pin the non-root user, the single worker, the
   readiness health check and the locked-dependency sync.

2. Docker-gated integration (skipped unless a docker daemon is present):
   build the image, run it, hit /health, /model-info and /predict, and
   scan the image filesystem for prohibited content.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_EXCLUSIONS = {
    # (category, patterns that must be present)
    "data directories": ["data/", "outputs/"],
    "experiment stores": ["mlruns/", "mlartifacts/"],
    "private data file types": ["**/*.csv", "**/*.mat", "**/*.edf", "**/*.xlsx"],
    "model pickles from context": ["**/*.pkl"],
    "credentials": [".env", "**/*.pem", "**/*.key"],
    "infrastructure state": [".git/", "**/*.tfstate", ".terraform/"],
    "caches": ["__pycache__/", ".pytest_cache/", ".ruff_cache/"],
    "local environments": [".venv/", "venv/"],
    "runtime-irrepo content": ["tests/", "docs/", "scripts/", "legacy/"],
}

REQUIRED_BUILD_INPUTS = ["pyproject.toml", "uv.lock", "README.md", "src/curve_icon_dbso/schema.py",
                        "src/curve_icon_dbso/api/main.py"]

# Types that must never enter the BUILD CONTEXT (includes .pkl: model bytes
# are trained inside the build, never shipped in from outside).
CONTEXT_PROHIBITED_SUFFIXES = (".csv", ".mat", ".edf", ".xlsx", ".pkl", ".pem", ".key",
                               ".tfstate", ".sqlite", ".db", ".nii")
# Types that must never exist in the RUNTIME IMAGE. Note: no ".pkl" here —
# the image legitimately contains /app/artifact/model.pkl (the synthetic,
# hash-verified serving artifact is public-track material by design).
IMAGE_PROHIBITED_SUFFIXES = (".csv", ".mat", ".edf", ".xlsx", ".pem", ".key",
                             ".tfstate", ".sqlite", ".db", ".nii")


def _dockerignore_patterns() -> list[str]:
    text = (REPO_ROOT / ".dockerignore").read_text()
    patterns = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def test_dockerignore_covers_every_prohibited_category():
    patterns = _dockerignore_patterns()
    for category, needed in REQUIRED_EXCLUSIONS.items():
        for pattern in needed:
            assert pattern in patterns, (
                f".dockerignore is missing {pattern!r} (category: {category})"
            )


def _is_excluded(path: str, patterns: list[str]) -> bool:
    """Conservative .dockerignore approximation (documented simplification:
    Docker's matcher also treats `dir/` matches as excluding everything
    beneath, which this walks anyway since we check every file)."""
    normalized = path
    for pattern in patterns:
        pat = pattern.lstrip("/")
        if pat.endswith("/"):
            if normalized.startswith(pat) or normalized == pat.rstrip("/"):
                return True
        elif "*" in pat:
            if fnmatch(normalized, pat) or fnmatch(Path(normalized).name, pat):
                return True
        elif normalized == pat or normalized.endswith("/" + pat):
            return True
    return False


def _effective_context_files() -> list[str]:
    """Repository files that would enter the build context: walk the working
    tree (docker sees the filesystem, not git) with the ignore rules applied,
    pruning .git (never sent) and the excluded virtualenvs."""
    patterns = _dockerignore_patterns()
    prune_names = {".git", ".venv", "venv"}
    included: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in prune_names]
        base = Path(dirpath).relative_to(REPO_ROOT)
        for filename in filenames:
            rel = filename if str(base) == "." else (base / filename).as_posix()
            if not _is_excluded(rel, patterns):
                included.append(rel)
    return included


def test_effective_build_context_contains_no_prohibited_files():
    included = _effective_context_files()
    prohibited = [rel for rel in included if rel.endswith(CONTEXT_PROHIBITED_SUFFIXES)]
    assert not prohibited, f"prohibited files would enter the build context: {prohibited}"
    for path in included:
        lowered = path.lower()
        assert "mlruns" not in lowered and "mlartifacts" not in lowered, path


def test_effective_build_context_contains_the_required_build_inputs():
    included = set(_effective_context_files())
    for required in REQUIRED_BUILD_INPUTS:
        assert required in included, (
            f"required build input {required!r} is excluded by .dockerignore"
        )


def test_dockerfile_builds_non_root_single_worker_readiness_checked_image():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    stages = dockerfile.split("FROM ")
    runtime_stage = stages[-1]
    user_lines = [
        line.strip() for line in runtime_stage.splitlines()
        if line.strip().upper().startswith("USER ")
    ]
    assert user_lines, "runtime stage has no USER directive (must be non-root)"
    assert user_lines[-1].split(None, 1)[1].strip() not in {"root", "0:0"}, (
        "runtime USER must not be root"
    )
    assert "HEALTHCHECK" in runtime_stage
    assert "/ready" in runtime_stage  # readiness-gated health check
    assert '"--workers", "1"' in runtime_stage  # API-010: one worker by default
    assert "uv sync --locked" in dockerfile  # REP-003: locked env in the image
    assert "CURVE_ARTIFACT_DIR=/app/artifact" in runtime_stage
    # The model is trained inside the build — never copied from the context.
    assert "curve_icon_dbso.training" in dockerfile
    assert not any(
        "COPY" in line and ".pkl" in line for line in dockerfile.splitlines()
    )


# ------------------------------------------------- docker-gated integration ---

docker_available = shutil.which("docker") is not None
docker_tests_enabled = os.environ.get("CURVE_SKIP_DOCKER_TESTS") != "1"
requires_docker = pytest.mark.skipif(
    not docker_available or not docker_tests_enabled,
    reason="Docker integration is unavailable or owned by the dedicated CI job",
)

_IMAGE_SCAN_SCRIPT = """
import os

# Prohibited extensions ANYWHERE except the documented benign zones:
# - system/pip CA trust stores (.pem): part of the base image, required for
#   TLS, not project content;
# - site-packages of hash-locked third-party wheels (scipy/numpy/sklearn/
#   matplotlib/pip): upstream legitimately ships .pem trust bundles and
#   .mat/.csv test fixtures there. OUR code is re-scanned below and is
#   NEVER exempt, wherever it lives.
PROHIBITED = {'.csv', '.mat', '.edf', '.xlsx', '.pkl', '.pem', '.key', '.tfstate', '.sqlite', '.db', '.nii'}
EXCLUDED_PREFIXES = (
    '/etc/ssl/certs',
    '/usr/lib/ssl/certs',
    '/usr/lib/ssl/cert.pem',  # Debian's CA bundle (trust store, not project content)
    '/usr/local/lib/python3.11/site-packages',
    '/app/.venv/lib/python3.11/site-packages',
    # The hash-verified SYNTHETIC serving artifact trained inside the build:
    # public-track material by design (tests/test_artifact.py proves
    # provenance; this scan is about foreign/private data).
    '/app/artifact',
)
offenders = []
our_pkg_offenders = []
for root, dirs, files in os.walk('/', topdown=True):
    dirs[:] = [d for d in dirs if d not in ('proc', 'sys', 'dev')]
    for name in files:
        path = os.path.join(root, name)
        if os.path.splitext(name)[1].lower() not in PROHIBITED:
            continue
        if path.startswith(EXCLUDED_PREFIXES):
            if 'curve_icon_dbso' in path:
                our_pkg_offenders.append(path)
            continue
        offenders.append(path)
print(';'.join(offenders + our_pkg_offenders))
"""


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def built_image():
    """Build the image tagged by source SHA (CON-006); skips without docker."""
    if not docker_available:
        pytest.skip("docker daemon not available on this machine")
    sha = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    tag = f"curve-icon-dbso-api:{sha}"
    subprocess.run(
        ["docker", "build", "--build-arg", f"SOURCE_SHA={sha}", "-t", tag, str(REPO_ROOT)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    return tag


def _wait_until_ready(port: int, timeout_seconds: float = 30.0) -> float:
    import time

    started = time.perf_counter()
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/ready", timeout=1
            ) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except Exception:  # noqa: BLE001 - probe loop
            time.sleep(0.05)
    raise TimeoutError(f"container not ready within {timeout_seconds}s")


@pytest.fixture(scope="module")
def running_container(built_image):
    import uuid

    port = _free_port()
    name = f"curve-icon-it-{uuid.uuid4().hex[:8]}"
    process = subprocess.Popen(
        ["docker", "run", "--rm", "--name", name, "-p", f"{port}:8000", built_image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        cold_start = _wait_until_ready(port)
        yield port, cold_start
    finally:
        subprocess.run(["docker", "kill", name], capture_output=True)
        process.wait(timeout=30)


@requires_docker
class TestDockerIntegration:
    """Full loop: build, run, health, predict, image-content scan."""

    def test_container_answers_health_model_info_and_predict(self, running_container):
        port, _ = running_container
        from curve_icon_dbso.api.payloads import synthetic_payload_rows

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            health = json.load(r)
        assert health["status"] == "alive" and health["ready"] is True
        assert health["research_only"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/model-info", timeout=5) as r:
            info = json.load(r)
        assert info["intended_use"] == "research only; not for clinical use"
        assert isinstance(info["beats_baseline"], bool)

        rows = synthetic_payload_rows(4)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=json.dumps({"rows": rows}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as r:
            predictions = json.load(r)["predictions"]
        assert len(predictions) == 4
        assert all(isinstance(value, float) for value in predictions)

    def test_container_starts_within_configured_limit(self, running_container):
        _, cold_start = running_container
        limit = float(os.environ.get("CURVE_COLD_START_LIMIT_SECONDS", "2"))
        assert cold_start < limit, (
            f"container cold start took {cold_start:.2f}s (limit {limit:.2f}s)"
        )

    def test_image_contains_no_prohibited_content(self, built_image):
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python", built_image,
             "-c", _IMAGE_SCAN_SCRIPT],
            capture_output=True, text=True, check=True,
        )
        offenders = [line for line in result.stdout.strip().split(";") if line]
        assert not offenders, f"image contains prohibited files: {offenders}"

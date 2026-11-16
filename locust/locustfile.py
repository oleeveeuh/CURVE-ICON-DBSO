"""Locust load-test profile for the public synthetic track (OBS-003).

Shape (LoadTestShape.tick): ramp from 1 to 50 concurrent users (≈ RPS at
~1s waits) over 5 minutes, SUSTAIN 50 for 10 minutes, then stop — the
protocol from docs/mlops_extension_plan.md §7.10 (OBS-003).

Traffic is generated SYNTHETIC payloads only: every /predict body comes
from ``curve_icon_dbso.api.payloads.synthetic_payload_rows`` — the same
deterministic, schema-valid generator the contract tests use. No real
data, no recorded requests, no payload replay from files.

Run headless (see docs/runbook.md §load-test)::

    uv run --extra cloud locust -f locust/locustfile.py \
        --headless -H https://<distribution>.cloudfront.net \
        --csv locust/results --html locust/report.html

Acceptance targets (OBS-004, measured — never asserted as givens):
p95 < 200 ms, error rate < 0.1%, zero task restarts, stable memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

# The payload generator lives in the repository package: make it importable
# no matter where locust is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curve_icon_dbso.api.payloads import synthetic_payload_rows  # noqa: E402

RAMP_SECONDS = 300  # 5-minute ramp: 1 -> 50 users
SUSTAIN_SECONDS = 600  # 10-minute sustain at 50
MAX_USERS = 50


class SyntheticLoadShape(LoadTestShape):
    """1 -> 50 users over 5 minutes, 50 sustained for 10 minutes, stop."""

    def tick(self) -> tuple[int, float] | None:
        elapsed = self.get_run_time()
        if elapsed < RAMP_SECONDS:
            users = 1 + int((elapsed / RAMP_SECONDS) * (MAX_USERS - 1))
            return users, 2.0  # spawn fast enough to track the ramp
        if elapsed < RAMP_SECONDS + SUSTAIN_SECONDS:
            return MAX_USERS, 2.0
        return None  # stop the run


class SyntheticApiUser(HttpUser):
    """One concurrent user issuing metadata-safe synthetic traffic."""

    wait_time = between(0.8, 1.2)  # ~1 request/second/user -> users ≈ RPS
    weight = 1

    def on_start(self) -> None:
        # Fail fast with a clear message if the environment is not the
        # synthetic public track (provenance contract, plan §5.1).
        with self.client.get("/ready", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("service not ready")

    @task(1)
    def ready(self) -> None:
        self.client.get("/ready", name="GET /ready")

    @task(4)
    def predict_small_batch(self) -> None:
        # 1-8 rows per request, freshly generated from the deterministic
        # synthetic generator (no recorded payloads, no file replay).
        n_rows = 1 + (self._counter() % 8)
        payload = {"rows": synthetic_payload_rows(n_rows, seed=42)}
        self.client.post(
            "/predict",
            json=payload,
            name="POST /predict [1-8 rows]",
        )

    def _counter(self) -> int:
        count = getattr(self, "_req_counter", 0) + 1
        self._req_counter = count
        return count

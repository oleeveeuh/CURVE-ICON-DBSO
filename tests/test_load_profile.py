"""Load-test profile tests (OBS-003): the Locust shape follows the exact
ramp/sustain/stop protocol, and every payload it can generate is
schema-valid synthetic content. Skips cleanly without locust installed
(the CI default suite does not carry the cloud extra).

The locust-dependent checks run in a SUBPROCESS: importing locust
monkey-patches ssl/gevent process-wide, which would corrupt unrelated
tests in the same pytest process."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCUSTFILE = REPO_ROOT / "locust" / "locustfile.py"

locust_installed = importlib.util.find_spec("locust") is not None
requires_locust = pytest.mark.skipif(not locust_installed, reason="locust (cloud extra) not installed")

# The locustfile imports the package from src/ — replicate sys.path setup.
PROBE_PREFIX = f"import sys; sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
PROBE_IMPORT = f"import importlib.util; spec = importlib.util.spec_from_file_location('lf', {str(LOCUSTFILE)!r}); lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf); "


def _run_probe(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", PROBE_PREFIX + PROBE_IMPORT + code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


@requires_locust
def test_shape_follows_the_observability_protocol():
    """OBS-003: ramp 1->50 users over 5 minutes, sustain 50 for 10 minutes,
    then stop. Spawn rate keeps pace with the ramp."""
    probe = (
        "RAMP, SUSTAIN, MAXU = lf.RAMP_SECONDS, lf.SUSTAIN_SECONDS, lf.MAX_USERS\n"
        "shape = lf.SyntheticLoadShape()\n"
        "def tick_at(t):\n"
        "    shape.get_run_time = lambda: t\n"
        "    return shape.tick()\n"
        "u0 = tick_at(0); assert u0[0] == 1 and u0[1] > 0, u0\n"
        "u_mid = tick_at(RAMP / 2); assert u_mid[0] == 1 + (MAXU - 1) // 2, u_mid\n"
        "u_pre = tick_at(RAMP - 1); assert u_pre[0] == MAXU - 1, u_pre\n"
        "u_sus = tick_at(RAMP + 1); assert u_sus[0] == MAXU, u_sus\n"
        "u_end = tick_at(RAMP + SUSTAIN - 1); assert u_end[0] == MAXU, u_end\n"
        "assert tick_at(RAMP + SUSTAIN + 1) is None\n"
        "print('SHAPE OK')\n"
    )
    result = _run_probe(probe)
    assert "SHAPE OK" in result.stdout, result.stderr or result.stdout


@requires_locust
def test_every_generated_payload_is_schema_valid():
    """The load profile may only send schema-valid synthetic rows (OBS-003
    privacy rule: generated payloads, never recorded traffic)."""
    probe = (
        "from curve_icon_dbso.api.schemas import PredictRequest\n"
        "from curve_icon_dbso.schema import FEATURES\n"
        "for n in range(1, 9):\n"
        "    request = PredictRequest.model_validate({'rows': lf.synthetic_payload_rows(n, seed=42)})\n"
        "    assert 1 <= len(request.rows) <= 8\n"
        "    for row in request.rows:\n"
        "        assert set(row.model_dump()) == set(FEATURES)\n"
        "print('PAYLOADS OK')\n"
    )
    result = _run_probe(probe)
    assert "PAYLOADS OK" in result.stdout, result.stderr or result.stdout


def test_profile_uses_only_the_synthetic_generator():
    """Source-level pin: the profile's traffic body comes from
    synthetic_payload_rows and nothing else."""
    source = LOCUSTFILE.read_text()
    assert "synthetic_payload_rows" in source
    assert "json.load" not in source and "open(" not in source, (
        "the load profile must generate payloads, never replay recorded traffic from files"
    )

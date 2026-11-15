"""FastAPI inference service for the synthetic demonstration model (Phase 2).

Deliberately import-light: importing this package pulls in nothing, so
Phase 5 load testing can use :mod:`curve_icon_dbso.api.payloads` (pure
stdlib + numpy/pandas) without the ``api`` dependency group, and the
core package never requires pydantic/FastAPI. Import the pieces
explicitly:

    from curve_icon_dbso.api.payloads import synthetic_payload_rows
    from curve_icon_dbso.api.main import create_app, app

Scope (ADR-002): ``/health``, ``/ready``, ``/model-info`` and
``/predict`` only — there is NO ``/explain`` endpoint in Phase 2.
"""

__all__: list[str] = []

# ADR-003 — Dependency and XGBoost choices

**Status:** Approved by the owner on 2026-11-13
**Source:** `docs/mlops_extension_plan.md` §17, DD-003
**Gates affected:** Phase 1 environment lock; CI; container build time/size;
scientific scope

## Decision

Use `uv` to generate a committed cross-platform lock and keep XGBoost in an
optional analysis dependency group. Include the fixed-parameter XGBoost
comparison only when explicitly enabled for the learning exercise; it is not
required for the core corrected analysis.

## Alternative considered

Use `pip-tools` and a platform-specific `requirements.lock`; omit XGBoost to
minimize native dependencies and keep the scientific comparison unchanged.

## Decision effect

Affects contributor workflow, container build time/size, CI, and scientific
scope.

## Notes

XGBoost must not enter the default test suite or the core corrected analysis
(SCI-006); its comparison runs only through the optional dependency group with
fixed, documented parameters.

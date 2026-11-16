# ADR-002 — Explanation endpoint

**Status:** Approved by the owner on 2026-11-13
**Source:** `docs/mlops_extension_plan.md` §17, DD-002
**Gates affected:** Phase 2 API scope; `/explain` remains excluded until this
decision is revisited

## Decision

Defer `/explain` from the minimum viable API. Add it only after the
prediction service is stable, using a synthetic-only explainer and an
explicit limitations response.

## Alternative considered

Include `/explain` in Phase 2 using SHAP or permutation-based local
attribution bundled with the synthetic artifact.

## Decision effect

Changes API scope, dependencies, image size, latency, and the risk of
overstating scientific meaning.

## Notes

`/explain` is not an acceptance criterion for Phase 2. Any future explanation
work must still satisfy EXP-001…006, and explanations from non-synthetic
data should follow the owner-responsibility guidance (non-blocking notices
in `docs/approvals.md`; the former item-6 PI requirement was retired
2026-11-15 as NOT_APPLICABLE).

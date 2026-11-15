# ADR-006 — Model promotion rule

**Status:** Approved by the owner on 2026-11-14 (DD-006); real-data promotion
additionally requires PI sign-off per `docs/approvals.md`
**Source:** `docs/mlops_extension_plan.md` §17, DD-006
**Decided by:** owner (the plan names owner + PI; the PI gate applies to any
real-data promotion and is unchanged)

## Decision

The **strict conditional rule**:

- Promote the best fixed candidate **only if it beats BOTH naive
  baselines** (mean and median dummy regressors) under the prespecified
  primary metric under the identical grouped folds.
- Otherwise, serve a **clearly labeled demonstration model** with
  `beats_baseline: false` visible in the model manifest and **no
  performance claim** anywhere in public documentation.

Machine-testable form (synthetic artifacts, required regardless since
Phase 2): the training entry point computes the baseline comparison,
records `beats_baseline` in the manifest, and the API surfaces it via
`/model-info`. Promotion happens in the training script, never by hand.

Rule integrity:

- The rule is fixed **before** any real-data result is viewed (SCI-011);
  changing it after viewing real results resets the analysis to
  exploratory.
- No result is tuned toward the retracted R² = 0.713, and `beats_baseline`
  is never edited after a run.

## Alternatives considered

- **Predetermined family regardless of score** (deployment independent of
  the observed comparison): rejected — it hides the honest
  beats/doesn't-beat outcome that this project explicitly preserves
  (SCI-014).

## Decision effect

- Phase 2 already implements this rule (manifest `beats_baseline` field);
  this ADR makes it the locked promotion policy for the synthetic track.
- Any future real-data promotion additionally requires the PI approval
  recorded in `docs/approvals.md`; no approval item is satisfied by this
  decision.

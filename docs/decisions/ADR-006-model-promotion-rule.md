# ADR-006 — Model promotion rule

**Status:** Approved by the owner on 2026-11-14 (DD-006); amended
2026-11-15/16 — the former PI sign-off clause is retired (independent
personal project); for any non-synthetic promotion, the
owner-responsibility guidance in `docs/approvals.md` applies first
(recommended, never a gate)
**Source:** `docs/mlops_extension_plan.md` §17, DD-006
**Decided by:** owner (the plan's owner+PI wording predates the
2026-11-15 retirement of the institutional framework)

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
- For any future non-synthetic promotion, review the
  owner-responsibility guidance first (register in
  `docs/data_access.md`, notices in `docs/approvals.md` — recommended,
  never a gate); no compliance item is satisfied by this decision.

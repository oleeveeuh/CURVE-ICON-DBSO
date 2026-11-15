# Approval checklist (redacted)

**Status: all items pending — no institutional approval is recorded.**
**Last reviewed:** 2026-11-13 ·
**Governing document:** `docs/mlops_extension_plan.md` (§6 Stakeholders and approvals)

This is the only approval record permitted in the repository, and it is
deliberately redacted. Decision documents, protocols, DUA/IRB correspondence,
and other evidence live in private storage **outside** this repository; this
file records only status, decision date, decision owner, and operational
restrictions. The presence of an item here never grants or implies approval.

## Rules

- Every item starts `PENDING`. A status may only change away from `PENDING`
  when a decision date **and** an out-of-repository evidence location are both
  recorded.
- Allowed statuses: `PENDING`, `APPROVED`, `APPROVED WITH RESTRICTIONS`,
  `DENIED`, `EXPIRED`, `SUPERSEDED`.
- Items past their review date revert to `PENDING` until re-confirmed.
- The **institutional real-data track stays disabled** until every item under
  "Institutional approvals" is recorded as `APPROVED` (or
  `APPROVED WITH RESTRICTIONS`) and unexpired. No synthetic/public work is
  blocked by these items.
- No account IDs, subject codes, per-subject values, or evidence documents may
  be added to this file — only the redacted fields shown per item.

## Institutional approvals (gate the real-data track)

### 1. PI scope approval

- Covers: engineering prototype scope, kinematic target, and synthetic public
  track (extension plan §6, row 1).
- Owner: PI and project owner
- Status: PENDING
- Decision date: —
- Evidence location (outside repository): —
- Restrictions: —
- Expiration / review date: —
- Gates: any new real-data Phase 1 work

### 2. IRB / human-subjects determination

- Covers: determination whether the new (kinematic) analysis and any re-analysis
  require IRB review or are exempt.
- Owner: PI and IRB office
- Status: PENDING
- Decision date: —
- Evidence location (outside repository): —
- Restrictions: —
- Expiration / review date: —
- Gates: any new real-data model run

### 3. DUA permissions

- Covers: permission for analysis, cloud processing, copies, model training,
  and derived-artifact retention under the applicable Data Use Agreement.
- Owner: PI and data provider (USC DABI / ICON Lab)
- Status: PENDING
- Decision date: —
- Evidence location (outside repository): —
- Restrictions: —
- Expiration / review date: —
- Gates: institutional cloud use; retention of real artifacts

### 4. Institutional AWS and security architecture

- Covers: institution-approved AWS account and security review for any
  institutional-track deployment.
- Owner: institutional security / ISO
- Status: PENDING
- Decision date: —
- Evidence location (outside repository): —
- Restrictions: —
- Expiration / review date: —
- Gates: institutional deployment

## Publication approvals

### 5. Public release of cohort size and aggregate metrics

- Covers: publishing cohort size and aggregate MAE/RMSE/R² in public
  documentation or the repository README.
- Owner: PI
- Status: PENDING
- Decision date: —
- Evidence location (outside repository): —
- Restrictions: —
- Expiration / review date: —
- Gates: public documentation

### 6. Real-data model explanations

- Covers: computing or publishing explanations (attribution/SHAP-style or
  otherwise) derived from real data.
- Owner: PI and, if applicable, IRB/DUA owner
- Status: PENDING
- Decision date: —
- Evidence location (outside repository): —
- Restrictions: —
- Expiration / review date: —
- Gates: any real-data explanation work (see ADR-002 for the synthetic-only
  public alternative)

## Design decisions are not institutional approvals

Owner-level design decisions DD-001 through DD-003 were approved by the
project owner on 2026-11-13 and are recorded in
[`docs/decisions/`](decisions/README.md) (ADR-001…003). They settle
engineering scope only; they do **not** satisfy any item above, and none of
them authorizes real-data analysis or institutional deployment.
DD-004–DD-006 remain open (ADR-004…006).

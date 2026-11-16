# Decision log / ADR index

One architecture decision record per design decision from
`docs/mlops_extension_plan.md` §17. Statuses: `Approved`, `Open`, `Proposed`,
`Superseded`. Approved decisions record the approver and date; open decisions
name the gate they block. New ADRs get the next sequential number.

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](ADR-001-public-repository-layout.md) | Public repository layout — one repository | Approved 2026-11-13 (owner) |
| [ADR-002](ADR-002-explanation-endpoint.md) | Defer `/explain` from the minimum viable API | Approved 2026-11-13 (owner) |
| [ADR-003](ADR-003-dependency-and-xgboost.md) | `uv` lock; XGBoost optional analysis dependency | Approved 2026-11-13 (owner) |
| [ADR-004](ADR-004-public-ingress-and-tls.md) | Public ingress and TLS — CloudFront default domain, ephemeral | Approved 2026-11-14 (owner) |
| [ADR-005](ADR-005-terraform-organization.md) | Terraform organization — single root, `dev` env, Phase 4 bootstrap | Approved 2026-11-14 (owner) |
| [ADR-006](ADR-006-model-promotion-rule.md) | Model promotion rule — strict conditional; non-synthetic promotion additionally requires a compliance-verified public dataset | Approved 2026-11-14 (owner); former PI clause retired 2026-11-15 |

Design decisions settle engineering scope only. They never verify a
dataset or authorize use of third-party data — see
[`docs/approvals.md`](../approvals.md) for the compliance gate and
[`docs/data_access.md`](../data_access.md) for the register.

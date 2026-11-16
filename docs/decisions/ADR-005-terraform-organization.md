# ADR-005 — Terraform organization

**Status:** Approved by the owner on 2026-11-14 (DD-005)
**Source:** `docs/mlops_extension_plan.md` §17, DD-005
**Decided by:** owner

## Decision

A **single small root module** at `infra/`, split into readable resource
files, with per-environment variable files under `infra/environments/`
(one file today: `dev.tfvars`). Remote state is bootstrapped from a
separate one-time stack at `infra/bootstrap/` (versioned private S3 bucket
+ DynamoDB lock table) that the owner runs manually once before the first
apply; its own state stays local and its resources are intentionally
retained across environment destroys.

Approved parameters recorded with this decision:

- Terraform environment value: `dev` (resource names and tags).
- GitHub protected environment gating apply/deploy: `aws-synthetic`.
- Region: `us-east-2` — the new-AWS-experience project's single assigned
  Region (owner decision 2026-11-14, before first apply). This supersedes the
  earlier us-west-1 choice: the managed SCP/RCP policies of the new AWS
  experience forbid creating regional resources outside the project Region.
- Remote-state decision: the bootstrap stack is **created in Phase 4**
  (no state bucket or lock table existed previously).
- Reusable modules are **not** extracted now; they are deferred until a
  second environment proves the need.
- GitHub OIDC federation (`enable_github_oidc`) is **disabled on this
  account**: the new-AWS-experience project SCP denies
  `iam:CreateOpenIDConnectProvider` (observed at first apply, 2026-11-14).
  CI delivery (CIC-003/CIC-004) therefore cannot run here; deployments are
  performed locally with the owner's short-lived `aws login` session. The
  workflows and policy JSONs remain in the repository, correct for a
  standard account, with the switch documented in `variables.tf`.

## Alternatives considered

- **Reusable modules from the outset** (network / service / artifact /
  observability): better multi-environment reuse; rejected for now —
  extra indirection with exactly one real environment and no second
  consumer of the interfaces yet.

## Decision effect

- Fastest review path: every resource of the public synthetic environment
  is visible as plain `infra/*.tf` files.
- A second environment (if ever needed) will either parameterize
  this root or extract modules at that point — a refactor with no
  scientific-scope impact.
- `dev` naming makes the ephemeral, non-production status of the single
  environment explicit in every resource name and tag.

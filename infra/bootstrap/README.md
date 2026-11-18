# Remote-state bootstrap (one-time, owner-run)

Creates the Terraform **remote state backend** required by INF-006: a
versioned, encrypted, private S3 bucket plus a DynamoDB lock table
(ADR-005 decision: the bootstrap is created in Phase 4; nothing pre-existed).
Since 2026-11-16 it also hosts the **persistent monthly budget** (INF-008),
moved out of the application stack so a destroy can never remove the cost
guardrail while retained resources remain.

This stack's own state stays **local** (gitignored via the repo-wide
`*.tfstate` rules) and is **intentionally retained**: deleting it would
orphan the main module's state. It is one of the documented remote-state
exceptions in the after-destroy check (`docs/runbook.md`).

## Safety legend

- ✅ SAFE — read-only or creates only the two state resources
- ⚠️ MUTATING — creates the state bucket + lock table (~$0.5/month total)
- ❌ DESTRUCTIVE — removes the state store; do not run while the main
  environment's state lives in it

## Steps

Run from `infra/bootstrap/` with AWS credentials for the target account
and region (`aws configure`, or your usual SSO/profile mechanism — no
static keys are stored anywhere in this repository).

```bash
cd infra/bootstrap

# ✅ SAFE: format and preview
terraform fmt -check -recursive
terraform init                      # local state (no backend needed here)
terraform plan -var owner=oleeveeuh -var budget_alert_email=you@example.com

# ⚠️ MUTATING: create the state bucket + lock table + persistent budget (one-time)
terraform apply -var owner=oleeveeuh -var budget_alert_email=you@example.com

# Record the outputs — they feed the main module and CI:
terraform output state_bucket_name   # -> -backend-config bucket=... AND -var state_bucket_name=...
terraform output lock_table_name     # -> -backend-config dynamodb_table=... AND -var state_lock_table=...
```

Then follow `docs/runbook.md` §"First plan/apply" for the main module.

## Why local state for the bootstrap?

Chicken-and-egg: the remote backend cannot store the state that creates
it. The bootstrap state is two resources and zero secrets; keeping it
local and gitignored is the standard, documented pattern (plan §14 lists
"remote-state exceptions intentionally retained" for exactly this).

## Teardown (rare)

❌ DESTRUCTIVE — only after the main environment is fully destroyed AND you
have exported/kept the main state file somewhere private:

```bash
# DynamoDB deletion protection must be disabled first (guardrail):
aws dynamodb update-table --table-name $(terraform output -raw lock_table_name) \
  --no-deletion-protection-enabled
terraform destroy -var owner=oleeveeuh
# The state bucket is not force-destroyed: empty it manually first, then
# delete it by hand if you truly intend to lose all state history.
```

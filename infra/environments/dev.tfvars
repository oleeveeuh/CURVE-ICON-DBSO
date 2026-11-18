# Public synthetic track — the single ephemeral environment (ADR-005).
#
# SAFE to commit: contains no account IDs, credentials, or secrets.
# The two state-backend names below come from `infra/bootstrap` outputs and
# are per-account; before running plan/apply locally, either edit them here
# or pass -var state_bucket_name=... -var state_lock_table=... (see
# docs/runbook.md). CI receives them from GitHub repository VARIABLES.

environment        = "dev"
aws_region         = "us-east-2" # the new-AWS-experience project Region (owner decision 2026-11-14; resources cannot be created elsewhere)
owner              = "oleeveeuh"
vpc_cidr           = "10.42.0.0/16"
desired_count      = 0     # INF-007: zero when idle; deploys pass 1 explicitly
enable_github_oidc = false # new-AWS-experience SCP denies OIDC providers; deploys run locally (variables.tf)
# budget_monthly_limit moved to the retained bootstrap stack (2026-11-16):
# it is a bootstrap variable now and must NOT appear in this file.

# Budget recipient: set your real address here before any apply (the
# deploy-workflow preflight blocks applies while any assignment still
# reads REPLACE...):
budget_alert_email = "REPLACE-with-your-budget-alert-email@example.com"

# Per-account values from infra/bootstrap outputs (bootstrap applied 2026-11-14):
state_bucket_name = "curve-icon-dbso-tfstate-fdf9b4aa"
state_lock_table  = "curve-icon-dbso-tfstate-locks"

# container_image is intentionally absent: it is passed per-deployment by
# .github/workflows/deploy.yml as an immutable digest reference (CIC-005).

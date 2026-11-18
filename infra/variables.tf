# Input variables for the public synthetic track root module (ADR-005:
# single root, per-environment tfvars under infra/environments/).
#
# NO variable may carry an account ID, credential, or subject identifier;
# tests/test_iam_policy_scope.py asserts this across the whole infra/ tree.

variable "aws_region" {
  description = "AWS region for the ephemeral public synthetic environment. MUST be the new-AWS-experience project Region (us-east-2): resources cannot be created in other Regions (managed SCP/RCP)."
  type        = string
  default     = "us-east-2"
}

variable "environment" {
  description = "Environment value used in every resource name and tag (ADR-005: dev)."
  type        = string
  default     = "dev"

  validation {
    condition     = can(regex("^[a-z0-9-]{2,12}$", var.environment))
    error_message = "environment must be 2-12 lowercase letters, digits, or hyphens."
  }
}

variable "owner" {
  description = "Owner tag value (INF-009). Set to the repository owner's handle."
  type        = string
  default     = "oleeveeuh"
}

variable "vpc_cidr" {
  description = "CIDR block for the dedicated VPC (no NAT gateways by design — see network.tf)."
  type        = string
  default     = "10.42.0.0/16"
}

variable "container_image" {
  description = <<-EOT
    Full container image reference INCLUDING the immutable digest
    (registry/repo@sha256:...), passed by the deploy workflow (CIC-005).
    Empty = the repository's bootstrap placeholder tag; valid because the
    service desired count is zero when idle (INF-007) and no task starts.
  EOT
  type        = string
  default     = ""
}

variable "desired_count" {
  description = "ECS service desired task count. ZERO when idle (INF-007); the deploy workflow passes 1 for a demonstration."
  type        = number
  default     = 0

  validation {
    condition     = var.desired_count >= 0 && var.desired_count <= 10
    error_message = "desired_count must be between 0 and 10 (this is a demonstration environment, not a scaling exercise)."
  }
}

variable "budget_alert_email" {
  description = "Email address for alarm notifications (SNS subscription; the persistent budget moved to the retained bootstrap stack on 2026-11-16). REQUIRED — replace the placeholder in environments/dev.tfvars with the real recipient BEFORE any apply."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_alert_email))
    error_message = "budget_alert_email must look like an email address."
  }
}

variable "enable_github_oidc" {
  description = <<-EOT
    Create the GitHub Actions OIDC provider + plan/apply/push CI roles
    (CIC-003). DEFAULT FALSE because the new-AWS-experience project account's
    managed service control policy explicitly DENIES
    iam:CreateOpenIDConnectProvider (observed 2026-11-14: SCP
    p-j328j09s) — on such accounts, deploys run locally with the owner's
    short-lived `aws login` session instead of GitHub OIDC. Set true on a
    standard account to enable the CI delivery path.
  EOT
  type        = bool
  default     = false
}

# --- Terraform remote state backend (INF-006, ADR-005) -----------------------
# Values produced by infra/bootstrap and consumed via -backend-config at
# `terraform init`; they are needed by IAM policy templates (below) but NOT
# as normal runtime inputs.

variable "state_bucket_name" {
  description = "Name of the Terraform STATE bucket created by infra/bootstrap (output: state_bucket_name)."
  type        = string
}

variable "state_lock_table" {
  description = "Name of the Terraform state LOCK TABLE created by infra/bootstrap (output: lock_table_name)."
  type        = string
}

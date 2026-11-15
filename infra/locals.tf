# Shared naming and tagging (INF-009). The repository identity and the
# GitHub environment name are PUBLIC values by design — they appear in OIDC
# trust policies and workflow files and must match docs/decisions/ADR-005.

locals {
  name_prefix = "curve-icon-dbso-${var.environment}"

  github_repo        = "oleeveeuh/CURVE-ICON-DBSO"
  github_environment = "aws-synthetic"

  # Header used to authenticate CloudFront -> ALB origin requests (ADR-004).
  # The NAME is public; only the VALUE (random_password.origin_verify) is
  # secret. It is an infrastructure-to-infrastructure credential: it never
  # becomes an application secret and no Secrets Manager resource exists.
  origin_verify_header = "X-Origin-Verify"

  # Versioned private artifact bucket (INF-002/004). The random suffix keeps
  # the GLOBAL S3 name unique; it is stable for the lifetime of the state.
  artifact_bucket_name = "${local.name_prefix}-artifacts-${random_id.suffix.hex}"
  artifact_prefix      = "artifacts/synthetic-demo"

  metrics_namespace = "curve-icon-dbso-${var.environment}"
  log_group_name    = "/ecs/${local.name_prefix}"

  tags = {
    Project     = "curve-icon-dbso"
    Environment = var.environment
    Track       = "public-synthetic"
    Owner       = var.owner
    ManagedBy   = "terraform"
  }
}

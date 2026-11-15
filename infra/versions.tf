# Pinned tool and provider versions (Prompt 5 requirement: pin versions).
#
# The S3 backend block is intentionally EMPTY here: backend values (state
# bucket, lock table, key) are supplied at `terraform init` time via
# -backend-config flags so that no account-specific value is committed to
# the repository (INF-006). See infra/bootstrap/README.md for the one-time
# bootstrap and infra/README.md for the init command used by CI and locally.

terraform {
  required_version = ">= 1.9.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {}
}

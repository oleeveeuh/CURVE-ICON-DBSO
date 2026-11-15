# One-time remote-state bootstrap stack (ADR-005). Same provider pins as
# the main root so the two stacks cannot drift in toolchain expectations.

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

  # NO remote backend here by design: this stack bootstraps the backend, so
  # its own state stays LOCAL (gitignored) and is intentionally retained —
  # it is one of the documented "remote-state exceptions" (plan §14).
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "curve-icon-dbso"
      Environment = "bootstrap"
      Track       = "public-synthetic"
      Owner       = var.owner
      ManagedBy   = "terraform"
    }
  }
}

# AWS provider with account-wide default tags (INF-009): every taggable
# resource created by this root module carries project, environment, track,
# owner, and managed-by tags without repeating them per resource.

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.tags
  }
}

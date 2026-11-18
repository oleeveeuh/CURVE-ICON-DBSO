# IAM: ECS execution/task roles (INF-003) and the GitHub Actions OIDC
# federation (CIC-003). No long-lived AWS keys exist anywhere; the three
# GitHub roles are trust-scoped to the exact repository and to the exact
# ref / protected environment (ADR-005). The policy documents live in
# infra/policies/*.json so the offline least-privilege tests
# (tests/test_iam_policy_scope.py) can assert their scoping without AWS.
#
# 2026-11-16 pre-publication review hardening: every project role is
# created on a role PATH with the permissions boundary
# (infra/policies/permissions-boundary.json — denies IAM user/credential
# lifecycle, OIDC-provider management, organizations and account
# management). Routine CI can never create OIDC providers: those actions
# are absent from infra/policies/ci-apply-policy.json (provider creation
# is a deliberate one-time owner-run action, e.g. from bootstrap, never
# part of routine deployment). Roles live on their path so every IAM
# resource constraint in the apply policy is path-scoped.

# Required by the OIDC trust principal ARN; interpolated at plan/apply time,
# never committed (the acceptance rule: no account IDs in the repository).
data "aws_caller_identity" "current" {}

resource "aws_iam_openid_connect_provider" "github" {
  # count = 0 on new-AWS-experience project accounts: their managed SCP
  # denies iam:CreateOpenIDConnectProvider (see variables.tf). The three
  # GHA roles below follow this switch so CI either works end-to-end or
  # is cleanly absent — no half-created federation.
  count = var.enable_github_oidc ? 1 : 0

  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # Since June 2024 AWS validates GitHub's issuing certificate chain
  # directly for token.actions.githubusercontent.com and ignores
  # thumbprints; the IAM API still requires a non-empty list. This is the
  # documented DigiCert root thumbprint for GitHub's OIDC endpoint.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  tags = { Name = "${local.name_prefix}-github-oidc" }
}

# Permissions boundary (2026-11-16 review): caps what ANY project role can
# ever do, regardless of which policy grants what. Deny-only boundary —
# see infra/policies/permissions-boundary.json and the offline tests.
resource "aws_iam_policy" "project_permissions_boundary" {
  name   = "${local.name_prefix}-permissions-boundary"
  policy = templatefile("${path.module}/policies/permissions-boundary.json", {})
}

# --- ECS roles ---------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name                 = "${local.name_prefix}-ecs-execution"
  path                 = "/${local.name_prefix}/"
  permissions_boundary = aws_iam_policy.project_permissions_boundary.arn
  assume_role_policy   = data.aws_iam_policy_document.ecs_assume.json

  tags = { Name = "${local.name_prefix}-ecs-execution" }
}

# Managed policy: ECR pull + CloudWatch Logs agent — the documented,
# narrow execution contract for Fargate tasks.
resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name                 = "${local.name_prefix}-ecs-task"
  path                 = "/${local.name_prefix}/"
  permissions_boundary = aws_iam_policy.project_permissions_boundary.arn
  assume_role_policy   = data.aws_iam_policy_document.ecs_assume.json

  tags = { Name = "${local.name_prefix}-ecs-task" }
}

# Application task role (INF-003, hardened 2026-11-16): the API loads its
# synthetic artifact from the image filesystem once at startup and touches
# no AWS service at request time, so the role carries NO Allow statements —
# only explicit Deny guardrails for artifact writes and secret stores
# (see infra/policies/task-role.json; the offline tests assert the absence
# of S3/CloudWatch allows).
resource "aws_iam_role_policy" "ecs_task" {
  name = "denied-artifact-writes-and-secret-stores"
  role = aws_iam_role.ecs_task.id
  policy = templatefile("${path.module}/policies/task-role.json", {
    artifact_bucket_arn = aws_s3_bucket.artifacts.arn
    artifact_prefix     = local.artifact_prefix
  })
}

# --- GitHub Actions OIDC roles (CIC-003) -------------------------------------

locals {
  # Trust sub patterns (CIC-003): exact repository + exact ref/environment.
  # plan trusts BOTH pull_request (PR plan job) and ref:refs/heads/main (the
  # read-only plan step of the deploy workflow); apply trusts ONLY the
  # protected environment; push trusts ONLY main.
  oidc_subs = {
    plan  = ["repo:${local.github_repo}:pull_request", "repo:${local.github_repo}:ref:refs/heads/main"]
    apply = "repo:${local.github_repo}:environment:${local.github_environment}"
    push  = "repo:${local.github_repo}:ref:refs/heads/main"
  }
}

resource "aws_iam_role" "gha_plan" {
  count = var.enable_github_oidc ? 1 : 0

  name                 = "${local.name_prefix}-gha-plan"
  path                 = "/${local.name_prefix}/"
  permissions_boundary = aws_iam_policy.project_permissions_boundary.arn
  assume_role_policy = templatefile("${path.module}/policies/oidc-trust.json", {
    account_id       = data.aws_caller_identity.current.account_id
    sub_pattern_json = jsonencode(local.oidc_subs.plan)
  })

  tags = { Name = "${local.name_prefix}-gha-plan" }
}

resource "aws_iam_role_policy" "gha_plan" {
  count = var.enable_github_oidc ? 1 : 0

  name = "terraform-plan-read-only"
  role = aws_iam_role.gha_plan[0].id
  policy = templatefile("${path.module}/policies/ci-plan-policy.json", {
    state_bucket_arn    = "arn:aws:s3:::${var.state_bucket_name}"
    lock_table_arn      = "arn:aws:dynamodb:${var.aws_region}:*:table/${var.state_lock_table}"
    artifact_bucket_arn = aws_s3_bucket.artifacts.arn
  })
}

resource "aws_iam_role" "gha_apply" {
  count = var.enable_github_oidc ? 1 : 0

  name                 = "${local.name_prefix}-gha-apply"
  path                 = "/${local.name_prefix}/"
  permissions_boundary = aws_iam_policy.project_permissions_boundary.arn
  assume_role_policy = templatefile("${path.module}/policies/oidc-trust.json", {
    account_id       = data.aws_caller_identity.current.account_id
    sub_pattern_json = jsonencode(local.oidc_subs.apply)
  })

  tags = { Name = "${local.name_prefix}-gha-apply" }
}

resource "aws_iam_role_policy" "gha_apply" {
  count = var.enable_github_oidc ? 1 : 0

  name = "terraform-apply-scoped"
  role = aws_iam_role.gha_apply[0].id
  policy = templatefile("${path.module}/policies/ci-apply-policy.json", {
    state_bucket_arn         = "arn:aws:s3:::${var.state_bucket_name}"
    lock_table_arn           = "arn:aws:dynamodb:${var.aws_region}:*:table/${var.state_lock_table}"
    artifact_bucket_arn      = aws_s3_bucket.artifacts.arn
    name_prefix              = local.name_prefix
    ecr_repo_arn             = aws_ecr_repository.api.arn
    permissions_boundary_arn = aws_iam_policy.project_permissions_boundary.arn
  })
}

# Build job only (push to main; pull requests never reach this role).
resource "aws_iam_role" "gha_push" {
  count = var.enable_github_oidc ? 1 : 0

  name                 = "${local.name_prefix}-gha-push"
  path                 = "/${local.name_prefix}/"
  permissions_boundary = aws_iam_policy.project_permissions_boundary.arn
  assume_role_policy = templatefile("${path.module}/policies/oidc-trust.json", {
    account_id       = data.aws_caller_identity.current.account_id
    sub_pattern_json = jsonencode(local.oidc_subs.push)
  })

  tags = { Name = "${local.name_prefix}-gha-push" }
}

resource "aws_iam_role_policy" "gha_push" {
  name  = "ecr-push-scoped"
  count = var.enable_github_oidc ? 1 : 0

  role = aws_iam_role.gha_push[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "GetEcrToken"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*" # not ARN-scopeable by AWS design
      },
      {
        Sid    = "PushToProjectRepository"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeImages",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:InitiateLayerUpload",
          "ecr:ListImages",
          "ecr:PutImage",
          "ecr:UploadLayerPart"
        ]
        Resource = aws_ecr_repository.api.arn
      }
    ]
  })
}

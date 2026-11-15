# Outputs consumed by the runbook, the deploy workflow, and the one-time
# GitHub repository setup (infra/README.md). None contain secrets.

output "artifact_bucket_name" {
  description = "Versioned private bucket holding the synthetic artifact + manifest (INF-002)."
  value       = aws_s3_bucket.artifacts.id
}

output "ecr_repository_url" {
  description = "ECR repository URL; the deploy workflow pushes sha-<12> tags and deploys by digest (CON-006, CIC-005)."
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster for the public synthetic service."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name (used by wait/rollback commands in docs/runbook.md)."
  value       = aws_ecs_service.api.name
}

output "cloudfront_domain" {
  description = "Public HTTPS entry point — the ONLY approved ingress (ADR-004, INF-012)."
  value       = aws_cloudfront_distribution.api.domain_name
}

output "alb_dns_name" {
  description = "ALB DNS name (origin only; not for reviewer traffic — ADR-004)."
  value       = aws_lb.api.dns_name
}

output "log_group_name" {
  description = "CloudWatch log group for metadata-only application logs (API-009)."
  value       = aws_cloudwatch_log_group.api.name
}

output "budget_name" {
  description = "Monthly cost budget with $5/$20 notifications (INF-008)."
  value       = aws_budgets_budget.monthly_cost.name
}

# --- GitHub repository setup (one-time; see infra/README.md) -----------------

output "github_oidc_role_arns" {
  description = "Role ARNs to store as GitHub repository VARIABLES (null when enable_github_oidc is false — new-AWS-experience accounts cannot create the OIDC federation; see variables.tf)."
  value = var.enable_github_oidc ? {
    plan  = aws_iam_role.gha_plan[0].arn
    apply = aws_iam_role.gha_apply[0].arn
    push  = aws_iam_role.gha_push[0].arn
  } : null
}

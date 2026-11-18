# Bootstrap outputs — the values that wire the main root module and CI to
# the state backend. None are secrets; they are still account-specific and
# therefore never committed (tests assert no account IDs under infra/).

output "state_bucket_name" {
  description = "S3 bucket holding Terraform state (INF-006). Feed to -backend-config bucket=... and -var state_bucket_name=..."
  value       = aws_s3_bucket.state.id
}

output "lock_table_name" {
  description = "DynamoDB table for state locking. Feed to -backend-config dynamodb_table=... and -var state_lock_table=..."
  value       = aws_dynamodb_table.state_locks.name
}

output "budget_name" {
  description = "Persistent monthly cost budget with $5/$20 notifications (INF-008) — retained across application-stack destroys (2026-11-16)."
  value       = aws_budgets_budget.monthly_cost.name
}

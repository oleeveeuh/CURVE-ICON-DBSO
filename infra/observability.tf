# CloudWatch log group for the container's awslogs driver (API-009).
#
# Short retention (plan §14: short log retention for the public
# demonstration). The application emits metadata-only JSON lines — request
# id, route, status, duration, row count, schema hash, model version, and
# error class — never payloads or feature values (API-008).
#
# Dashboards and alarms (OBS-001/OBS-002) live in alarms.tf and
# dashboard.tf, added in Phase 5.

resource "aws_cloudwatch_log_group" "api" {
  name              = local.log_group_name
  retention_in_days = 14

  tags = { Name = local.log_group_name }
}

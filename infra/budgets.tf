# AWS Budgets notifications (INF-008): $5 and $20 monthly thresholds.
#
# One COST budget with absolute-value thresholds (per INF-008 wording
# "notifications at $5 and $20"), including a FORECASTED notification so
# the $20 ceiling warns BEFORE it is reached. The subscriber email comes
# from var.budget_alert_email and must be replaced in environments/dev.tfvars
# before the first apply (enforced by variable validation).

resource "aws_budgets_budget" "monthly_cost" {
  name         = "${local.name_prefix}-monthly-cost"
  budget_type  = "COST"
  limit_amount = var.budget_monthly_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 5 # INF-008: $5 actual spend
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 20 # INF-008: $20 forecast
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 20 # INF-008: $20 actual (hard ceiling breached)
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}

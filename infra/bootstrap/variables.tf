variable "aws_region" {
  description = "Region for the state bucket and lock table (matches the main environment: us-east-2)."
  type        = string
  default     = "us-east-2"
}

variable "owner" {
  description = "Owner tag value (INF-009). Set to the repository owner's handle."
  type        = string
  default     = "oleeveeuh"
}

variable "budget_alert_email" {
  description = "Email address for AWS Budgets notifications at $5/$20 (INF-008). REQUIRED — the budget is RETAINED across application-stack destroys (2026-11-16 review), so pass the real recipient at apply time."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_alert_email))
    error_message = "budget_alert_email must look like an email address."
  }
}

variable "budget_monthly_limit" {
  description = "Monthly AWS Budgets limit in USD (INF-008: notifications at $5 and $20; ceiling $20)."
  type        = string
  default     = "20"
}

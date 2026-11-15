variable "aws_region" {
  description = "Region for the state bucket and lock table (matches the main environment: us-east-2)."
  type        = string
  default     = "us-east-2"
}

variable "owner" {
  description = "Owner tag value (INF-009)."
  type        = string
  default     = "olivialiau"
}

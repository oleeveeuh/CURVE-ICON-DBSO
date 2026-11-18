# Phase 5 observability: alarms (OBS-002), log metric filters, and the
# alarm notification topic. Low-cardinality only — the metric filters use
# the application's existing metadata-only JSON logs (API-009) and a fixed
# error_class dimension; nothing payload-derived is ever extracted.

locals {
  alarm_email = var.budget_alert_email # the same ops inbox as INF-008
}

resource "aws_sns_topic" "alarms" {
  name              = "${local.name_prefix}-alarms"
  kms_master_key_id = null # SSE-S3-style default encryption for SNS

  tags = { Name = "${local.name_prefix}-alarms" }
}

resource "aws_sns_topic_subscription" "alarm_email" {
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = local.alarm_email # requires one-time confirmation click by the owner
}

# CloudWatch alarms may publish to the topic (alarm-action principal).
# CloudWatch publishes with the ALARM's ARN as aws:SourceArn — the topic's
# own ARN would never match and delivery would be denied (2026-11-16
# review). The condition therefore allows this project's alarm ARN pattern
# and is additionally constrained to the project account via
# aws:SourceAccount (the confused-deputy guard AWS documents for
# alarm-to-SNS delivery).
data "aws_iam_policy_document" "alarms_topic" {
  statement {
    sid     = "AllowCloudWatchAlarmPublish"
    effect  = "Allow"
    actions = ["sns:Publish"]
    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }
    resources = [aws_sns_topic.alarms.arn]

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudwatch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:alarm:${local.name_prefix}-*"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_sns_topic_policy" "alarms" {
  arn    = aws_sns_topic.alarms.arn
  policy = data.aws_iam_policy_document.alarms_topic.json
}

# --- log metric filters (metadata-only; no payload values extracted) ---------

resource "aws_cloudwatch_log_metric_filter" "invalid_inputs" {
  name           = "${local.name_prefix}-invalid-inputs-by-class"
  log_group_name = aws_cloudwatch_log_group.api.name
  # CloudWatch filter syntax: "= *" matches an existing, non-null value, so
  # only error-path requests (error_class set) are counted.
  pattern = "{ $.error_class = * }"

  metric_transformation {
    name      = "${local.name_prefix}-InvalidInputs"
    namespace = local.metrics_namespace
    value     = "1"
    dimensions = {
      # finite app-defined classes only — never payload values
      error_class = "$.error_class"
    }
  }
}

resource "aws_cloudwatch_log_metric_filter" "server_restarts" {
  name           = "${local.name_prefix}-server-restarts"
  log_group_name = aws_cloudwatch_log_group.api.name
  # uvicorn logs this exact line once per server process start: any task
  # (re)start lands here, so crash-looping shows up as a burst.
  pattern = "\"Started server process\""

  metric_transformation {
    name      = "${local.name_prefix}-Restarts"
    namespace = local.metrics_namespace
    value     = "1"
  }
}

# --- alarms (OBS-002) ---------------------------------------------------------

# 1) 5xx rate above 1% for 5 minutes (metric math over ALB counters).
resource "aws_cloudwatch_metric_alarm" "http_5xx_ratio" {
  alarm_name          = "${local.name_prefix}-http-5xx-ratio-high"
  alarm_description   = "OBS-002: 5xx responses exceed 1% of requests over 5 minutes. Triage: docs/runbook.md #alarm-triage"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
  treat_missing_data  = "notBreaching" # idle scale-to-zero periods are healthy
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  threshold           = 1 # percent

  metric_query {
    id          = "e5xx"
    return_data = false
    metric {
      metric_name = "HTTPCode_Target_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = aws_lb.api.arn_suffix
      }
    }
  }

  metric_query {
    id          = "ereq"
    return_data = false
    metric {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = aws_lb.api.arn_suffix
      }
    }
  }

  metric_query {
    id          = "e1"
    expression  = "e5xx/ereq*100"
    label       = "5xx % of requests"
    return_data = true
  }
}

# 2) p95 latency above 200 ms for 5 minutes.
resource "aws_cloudwatch_metric_alarm" "target_p95_latency" {
  alarm_name          = "${local.name_prefix}-target-p95-latency-high"
  alarm_description   = "OBS-002: target p95 exceeds 200 ms over 5 minutes. Triage: docs/runbook.md #alarm-triage"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
  treat_missing_data  = "missing"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  threshold           = 0.2 # seconds
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  extended_statistic  = "p95"
  dimensions = {
    TargetGroup  = aws_lb_target_group.api.arn_suffix
    LoadBalancer = aws_lb.api.arn_suffix
  }
}

# 3) any unhealthy target for 5 minutes.
resource "aws_cloudwatch_metric_alarm" "unhealthy_targets" {
  alarm_name          = "${local.name_prefix}-unhealthy-targets"
  alarm_description   = "OBS-002: at least one target unhealthy for 5 minutes. Triage: docs/runbook.md #alarm-triage"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
  treat_missing_data  = "missing"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  threshold           = 1
  metric_name         = "UnHealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Maximum"
  dimensions = {
    TargetGroup  = aws_lb_target_group.api.arn_suffix
    LoadBalancer = aws_lb.api.arn_suffix
  }
}

# 4) unexpected task restarts: two or more server starts inside 5 minutes
# (a rolling deploy contributes exactly one; churn/crash-loops burst).
resource "aws_cloudwatch_metric_alarm" "task_restarts" {
  alarm_name          = "${local.name_prefix}-unexpected-restarts"
  alarm_description   = "OBS-002: two or more server (re)starts within 5 minutes — possible crash loop. Triage: docs/runbook.md #alarm-triage"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
  treat_missing_data  = "notBreaching"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  threshold           = 2
  metric_name         = "${local.name_prefix}-Restarts"
  namespace           = local.metrics_namespace
  period              = 300
  statistic           = "Sum"
}

# Phase 5 dashboard (OBS-001): requests, status classes, p50/p95, target
# health, task restarts, CPU/memory, invalid-input counts by error class,
# deployment state, and deployed model-version visibility. All widgets are
# metadata-only — no payload-derived dimensions anywhere.

locals {
  # Model-version visibility: the application logs one metadata-only JSON
  # line at startup containing model_version (API-009); a Logs Insights
  # query surfaces the currently deployed version on the dashboard.
  model_version_query = <<-EOT
    fields @timestamp, @message
    | filter @message like /model_version/
    | sort @timestamp desc
    | limit 5
  EOT

  dashboard_body = jsonencode({
    widgets = [
      # row 1: traffic + status classes + latency
      {
        type = "metric", x = 0, y = 0, width = 8, height = 6,
        properties = {
          title  = "Requests / 4xx / 5xx (sum per minute)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.api.arn_suffix, { stat = "Sum", label = "requests" }],
            [".", "HTTPCode_Target_4XX_Count", ".", ".", { stat = "Sum", label = "4xx" }],
            [".", "HTTPCode_Target_5XX_Count", ".", ".", { stat = "Sum", label = "5xx" }],
          ]
        }
      },
      {
        type = "metric", x = 8, y = 0, width = 8, height = 6,
        properties = {
          title  = "Target latency p50 / p95 (seconds)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "TargetGroup", aws_lb_target_group.api.arn_suffix, "LoadBalancer", aws_lb.api.arn_suffix, { stat = "p50", label = "p50" }],
            ["...", { stat = "p95", label = "p95" }],
          ]
        }
      },
      {
        type = "metric", x = 16, y = 0, width = 8, height = 6,
        properties = {
          title  = "Target health (healthy / unhealthy)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            ["AWS/ApplicationELB", "HealthyHostCount", "TargetGroup", aws_lb_target_group.api.arn_suffix, "LoadBalancer", aws_lb.api.arn_suffix, { stat = "Minimum", label = "healthy" }],
            ["...", "UnHealthyHostCount", ".", ".", ".", ".", { stat = "Maximum", label = "unhealthy" }],
          ]
        }
      },
      # row 2: task resources, restarts, invalid inputs
      {
        type = "metric", x = 0, y = 6, width = 8, height = 6,
        properties = {
          title  = "ECS CPU / memory utilization (%)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", aws_ecs_cluster.main.name, "ServiceName", aws_ecs_service.api.name, { stat = "Average", label = "cpu" }],
            ["...", "MemoryUtilization", ".", ".", ".", ".", { stat = "Average", label = "memory" }],
          ]
        }
      },
      {
        type = "metric", x = 8, y = 6, width = 8, height = 6,
        properties = {
          title  = "Server (re)starts (5-min sum; deploy = 1, burst = crash loop)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            [local.metrics_namespace, "${local.name_prefix}-Restarts", { stat = "Sum" }],
          ]
          annotations = {
            horizontal = [
              { value = 2, label = "OBS-002 alarm threshold", color = "#d62728" }
            ]
          }
        }
      },
      {
        type = "metric", x = 16, y = 6, width = 8, height = 6,
        properties = {
          title  = "Invalid inputs by error class (sum per minute)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            [local.metrics_namespace, "${local.name_prefix}-InvalidInputs", "error_class", "validation_error", { stat = "Sum" }],
            ["...", "missing_field", ".", ".", { stat = "Sum" }],
            ["...", "unknown_field", ".", ".", { stat = "Sum" }],
            ["...", "out_of_range", ".", ".", { stat = "Sum" }],
            ["...", "non_finite_value", ".", ".", { stat = "Sum" }],
            ["...", "batch_too_large", ".", ".", { stat = "Sum" }],
            ["...", "invalid_structure", ".", ".", { stat = "Sum" }],
          ]
        }
      },
      # row 3: deployment state + deployed model version + alarm statuses
      {
        type = "metric", x = 0, y = 12, width = 8, height = 6,
        properties = {
          title  = "Task count (deployment state)"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            # expression entries are single-element arrays with an
            # "expression" OBJECT (documented CloudWatch widget form)
            [{ expression = "SEARCH('MetricName=\"RunningCount\" AND ServiceName=\"${aws_ecs_service.api.name}\"', 'Average')", label = "running" }],
            [{ expression = "SEARCH('MetricName=\"DesiredCount\" AND ServiceName=\"${aws_ecs_service.api.name}\"', 'Average')", label = "desired" }],
          ]
        }
      },
      {
        type = "log", x = 8, y = 12, width = 16, height = 6,
        properties = {
          title  = "Deployed model version (startup line, metadata-only)"
          region = var.aws_region
          query  = "SOURCE '${aws_cloudwatch_log_group.api.arn}' | ${local.model_version_query}"
          view   = "table"
        }
      },
      {
        type = "alarm", x = 0, y = 18, width = 24, height = 6,
        properties = {
          title = "OBS-002 alarms"
          alarms = [
            aws_cloudwatch_metric_alarm.http_5xx_ratio.arn,
            aws_cloudwatch_metric_alarm.target_p95_latency.arn,
            aws_cloudwatch_metric_alarm.unhealthy_targets.arn,
            aws_cloudwatch_metric_alarm.task_restarts.arn,
          ]
        }
      },
    ]
  })
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${local.name_prefix}-dashboard"
  dashboard_body = local.dashboard_body
}

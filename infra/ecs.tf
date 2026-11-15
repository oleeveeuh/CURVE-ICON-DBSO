# ECS Fargate service running the Phase 2 synthetic-track inference image
# (plan §9.3: ECS-first architecture; no EKS unless separately justified).

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  # No execute-command configuration: the service sets
  # enable_execute_command = false, so there is no shell access into tasks.

  tags = { Name = "${local.name_prefix}-cluster" }
}

# Immutable image reference (CON-006, CIC-005): the deploy workflow passes
# registry/repo@sha256:<digest> as var.container_image. Empty (idle applies)
# falls back to a never-pushed bootstrap tag — safe because desired_count
# defaults to zero (INF-007), so no task ever starts from the placeholder.
locals {
  container_image = var.container_image != "" ? var.container_image : "${aws_ecr_repository.api.repository_url}:bootstrap"
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"

  execution_role_arn = aws_iam_role.ecs_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  # ARM64 (Graviton): ~10% cheaper than x86_64 and matches the locally
  # built images (Apple Silicon Docker VM builds native linux/arm64).
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = local.container_image
      essential = true

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "CURVE_ARTIFACT_DIR"
          value = "/app/artifact" # baked at build time; trained in-image (Dockerfile)
        }
      ]

      # Metadata-only logs (API-009): the application never logs payloads.
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "api"
        }
      }

      # No secrets/credentials in the container definition (prompt
      # constraint: no Secrets Manager when there is no application secret).
    }
  ])

  tags = { Name = "${local.name_prefix}-api" }
}

resource "aws_ecs_service" "api" {
  name            = "api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count # zero when idle (INF-007)
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = true # egress only; see network.tf header (no NAT by design)
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  health_check_grace_period_seconds = 30

  # Rolling deployments; the previous task revision stays available for
  # rollback (CIC-006), and the circuit breaker reverts a bad deployment
  # automatically instead of thrashing.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  enable_execute_command = false

  tags = { Name = "${local.name_prefix}-service" }
}

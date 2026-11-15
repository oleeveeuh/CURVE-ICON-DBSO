# Internet-facing ALB (ADR-004). Reachable ONLY from CloudFront:
# - security group restricts port 80 to the CloudFront origin-facing
#   managed prefix list (network.tf);
# - the default listener action is a fixed 403, and traffic is forwarded
#   to the target group ONLY when it carries the secret origin header.
# There is no HTTPS listener here: TLS terminates at CloudFront on the
# default *.cloudfront.net domain (no custom domain/ACM per ADR-004).

resource "aws_lb" "api" {
  name               = "${local.name_prefix}-alb" # <= 32 chars
  load_balancer_type = "application"
  internal           = false

  security_groups = [aws_security_group.alb.id]
  subnets         = aws_subnet.public[*].id

  idle_timeout               = 60
  drop_invalid_header_fields = true  # AWS-0052: the origin-verify header is well-formed, so dropping malformed headers is safe (AWS-0052)
  enable_deletion_protection = false # ephemeral environment (ADR-004)

  tags = { Name = "${local.name_prefix}-alb" }
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name_prefix}-tg" # <= 32 chars
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip" # awsvpc network mode: targets are task ENIs
  vpc_id      = aws_vpc.main.id

  health_check {
    path                = "/ready" # readiness-gated (CON-003, MOD-003)
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  deregistration_delay = 30

  tags = { Name = "${local.name_prefix}-tg" }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  # Anything that reaches the ALB WITHOUT the secret origin header is
  # rejected here — including direct hits on the ALB DNS name, which is
  # publicly resolvable but only accepts CloudFront prefix-list traffic at
  # the security group anyway (defense in depth).
  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      status_code  = "403"
      message_body = "forbidden: missing origin verification"
    }
  }
}

resource "aws_lb_listener_rule" "verified_origin" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    http_header {
      http_header_name = local.origin_verify_header
      values           = [random_password.origin_verify.result]
    }
  }
}

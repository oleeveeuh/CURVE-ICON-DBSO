# Network layout for the ephemeral public synthetic environment.
#
# DELIBERATE SIMPLICITY (prompt constraint: no NAT gateways unless proven
# required): public subnets ONLY. ECS Fargate tasks run here with a
# run-time-assigned public IP, which gives them egress (image pulls from
# ECR public endpoints, AWS API calls) through the internet gateway
# without a NAT gateway (~$32/month saved). The task's public IP is
# EGRESS-ONLY in practice: the task security group allows no inbound
# traffic except port 8000 from the ALB security group (INF-005), and the
# service is reachable only through the ALB (ADR-004).
#
# Residual risk accepted and documented: tasks use AWS public endpoints
# over the internet path rather than VPC endpoints. No private data ever
# transits this path (public synthetic track only, plan §5.1).

data "aws_availability_zones" "available" {
  state = "available"
}

# CloudFront origin-facing IP ranges: the ONLY addresses allowed to reach
# the ALB listener (ADR-004). AWS-managed and automatically updated.
data "aws_ec2_managed_prefix_list" "cloudfront_origin_facing" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${local.name_prefix}-vpc" }
}

resource "aws_subnet" "public" {
  count = 2 # ALB requires subnets in >= 2 AZs; us-east-2 has 3 AZs.

  vpc_id                  = aws_vpc.main.id
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  map_public_ip_on_launch = false # tasks get run-time-assigned IPs, not host IPs

  tags = { Name = "${local.name_prefix}-public-${count.index}" }
}

resource "aws_internet_gateway" "main" {
  # vpc_id is REQUIRED here: without it the provider v5 happily creates a
  # DETACHED gateway and the default route then fails with "route table and
  # network gateway belong to different networks".
  vpc_id = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${local.name_prefix}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# --- Security groups (INF-005: task reachable only through the ALB) ----------

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "ALB ingress: HTTP from CloudFront origin-facing IPs only (ADR-004)"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-alb-sg" }
}

resource "aws_security_group_rule" "alb_ingress_from_cloudfront" {
  type              = "ingress"
  security_group_id = aws_security_group.alb.id

  from_port       = 80
  to_port         = 80
  protocol        = "tcp"
  prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront_origin_facing.id]

  description = "HTTP from CloudFront origin-facing managed prefix list (ADR-004)"
}

resource "aws_security_group_rule" "alb_egress_to_tasks" {
  type              = "egress"
  security_group_id = aws_security_group.alb.id

  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.task.id
  description              = "Forward to task containers only"
}

resource "aws_security_group" "task" {
  name        = "${local.name_prefix}-task-sg"
  description = "ECS task ingress: port 8000 from the ALB only (INF-005)"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-task-sg" }
}

resource "aws_security_group_rule" "task_ingress_from_alb" {
  type              = "ingress"
  security_group_id = aws_security_group.task.id

  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.alb.id

  description = "Application traffic from the ALB only"
}

# Egress is restricted to what the workload actually needs — no unrestricted
# egress rule (AWS-0104): HTTPS (image pulls from ECR, AWS API calls to
# CloudWatch/S3 — all port 443) and DNS to the VPC resolver link-local.
resource "aws_security_group_rule" "task_egress_https" {
  type              = "egress"
  security_group_id = aws_security_group.task.id

  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]

  description = "HTTPS only: ECR pulls and AWS public endpoints (no NAT by design)"
}

resource "aws_security_group_rule" "task_egress_dns" {
  type              = "egress"
  security_group_id = aws_security_group.task.id

  from_port   = 53
  to_port     = 53
  protocol    = "udp"
  cidr_blocks = ["169.254.169.253/32"] # VPC DNS resolver (link-local)

  description = "DNS resolution via the VPC resolver"
}

resource "aws_security_group_rule" "task_egress_dns_tcp" {
  type              = "egress"
  security_group_id = aws_security_group.task.id

  from_port   = 53
  to_port     = 53
  protocol    = "tcp"
  cidr_blocks = ["169.254.169.253/32"] # VPC DNS resolver, TCP fallback

  description = "DNS resolution via the VPC resolver (TCP fallback)"
}

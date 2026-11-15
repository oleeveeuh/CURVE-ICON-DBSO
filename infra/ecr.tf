# ECR repository for the synthetic-track inference image (INF-002).
# Tags are IMMUTABLE (CON-006): every push uses a fresh sha-<12 hex> tag,
# so a tag can never be silently repointed to different bytes.

resource "aws_ecr_repository" "api" {
  name = "${local.name_prefix}-api"

  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256" # KMS not required for a public synthetic demo; keep cost/complexity down
  }

  tags = { Name = "${local.name_prefix}-api" }
}

# Lifecycle (prompt requirement): keep the last 10 sha-tagged images for
# rollback; expire untagged (untagged happens when a scan-pushed image's
# tag is later overwritten — cannot happen with IMMUTABLE, but belt and
# braces) after 7 days.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 10 immutable sha-tagged images"
        selection = {
          tagStatus      = "tagged"
          tagPatternList = ["sha-*"]
          countType      = "imageCountMoreThan"
          countNumber    = 10
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      }
    ]
  })
}

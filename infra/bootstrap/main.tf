# Remote-state resources (INF-006): a versioned, encrypted, private S3
# bucket for Terraform state and a DynamoDB table for state locking.
#
# This stack is run ONCE by the owner, manually, before the first apply of
# the main root module (infra/). Its own state is local (gitignored) and
# retained on purpose — destroying it would orphan the main state.

locals {
  base_name       = "curve-icon-dbso-tfstate"
  bucket_name     = "${local.base_name}-${random_id.suffix.hex}" # GLOBAL name uniqueness
  lock_table_name = "${local.base_name}-locks"
}

resource "random_id" "suffix" {
  byte_length = 4
}

# --- State bucket ------------------------------------------------------------

resource "aws_s3_bucket" "state" {
  bucket = local.bucket_name

  # NOT force_destroy: the bucket holding terraform state must require a
  # deliberate manual emptying before deletion (guardrail, INF-006).
  force_destroy = false

  tags = { Name = local.bucket_name }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket = aws_s3_bucket.state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id

  versioning_configuration {
    status = "Enabled" # state history preserved (also enables state recovery)
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    id     = "expire-noncurrent-state-versions"
    status = "Enabled"

    filter {} # all objects

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

data "aws_iam_policy_document" "state_tls_only" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]

    # Bucket policies REQUIRE a Principal on every statement.
    principals {
      type        = "*"
      identifiers = ["*"]
    }

    resources = [
      aws_s3_bucket.state.arn,
      "${aws_s3_bucket.state.arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id
  policy = data.aws_iam_policy_document.state_tls_only.json

  depends_on = [aws_s3_bucket_public_access_block.state]
}

# --- State lock table ---------------------------------------------------------

resource "aws_dynamodb_table" "state_locks" {
  name         = local.lock_table_name
  hash_key     = "LockID"
  billing_mode = "PAY_PER_REQUEST" # pennies at this scale

  attribute {
    name = "LockID"
    type = "S"
  }

  server_side_encryption {
    enabled = true
  }

  point_in_time_recovery {
    enabled = true
  }

  # Guardrail: deleting the lock table requires deliberately disabling this
  # first, so `terraform destroy` on the bootstrap stack cannot silently
  # remove the locking primitive.
  deletion_protection_enabled = true

  tags = { Name = local.lock_table_name }
}

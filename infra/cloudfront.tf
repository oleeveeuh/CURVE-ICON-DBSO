# CloudFront distribution — the ONLY public ingress (ADR-004, INF-012).
#
# HTTPS terminates here on the default *.cloudfront.net certificate (no
# custom domain per ADR-004). The origin is the ALB over HTTP, locked down
# two ways: the ALB security group (CloudFront prefix list only) and the
# secret origin header below. Caching is DISABLED: this is a dynamic
# inference API; nothing is cached at the edge, so no payloads or
# responses are ever stored in CloudFront caches.

resource "random_password" "origin_verify" {
  length  = 32
  special = false

  # The value is infrastructure-level (CloudFront -> ALB authentication).
  # It lives in Terraform state (private remote backend, INF-006) and in
  # the distribution config — NOT in the repository, NOT an application
  # secret, and no Secrets Manager resource is created for it.
}

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

resource "aws_cloudfront_distribution" "api" {
  enabled         = true
  is_ipv6_enabled = true
  comment         = "curve-icon-dbso public synthetic track (ephemeral; research only, not for clinical use)"
  price_class     = "PriceClass_100" # NA/EU edges only — cheapest class, fine for a demonstration
  http_version    = "http2"

  origin {
    origin_id   = "alb-api"
    domain_name = aws_lb.api.dns_name

    # Origin authentication (ADR-004): every CloudFront -> ALB request
    # carries this header; the ALB forwards only matching requests.
    custom_header {
      name  = local.origin_verify_header
      value = random_password.origin_verify.result
    }

    custom_origin_config {
      http_port                = 80
      https_port               = 443
      origin_protocol_policy   = "http-only" # no ACM cert without a custom domain (ADR-004)
      origin_ssl_protocols     = ["TLSv1.2"]
      origin_read_timeout      = 30
      origin_keepalive_timeout = 5
    }
  }

  default_cache_behavior {
    target_origin_id = "alb-api"

    # CloudFront only accepts [HEAD,GET], [HEAD,GET,OPTIONS], or the FULL
    # method set — and /predict needs POST, so the full set it is. This
    # widens nothing at the origin: the ALB/task still only accept what the
    # FastAPI service implements, and caching is disabled.
    allowed_methods = ["HEAD", "DELETE", "POST", "GET", "OPTIONS", "PUT", "PATCH"]
    cached_methods  = ["GET", "HEAD"]

    viewer_protocol_policy = "redirect-to-https" # HTTPS-only viewers
    compress               = true
    cache_policy_id        = data.aws_cloudfront_cache_policy.disabled.id

    # No origin request policy: CloudFront forwards no cookies or extra
    # headers to the origin (least exposure for a public API).
  }

  restrictions {
    geo_restriction {
      restriction_type = "none" # public reviewer access; ADR-004 accepted public HTTPS
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true # *.cloudfront.net (ADR-004)
  }

  tags = { Name = "${local.name_prefix}-cdn" }
}

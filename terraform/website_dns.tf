resource "aws_acm_certificate" "cert" {
  provider                  = aws.cert_provider
  domain_name               = var.site_domain
  subject_alternative_names = ["*.${var.site_domain}"]
  validation_method         = "DNS"

  tags = {
    Name = var.site_domain
  }
}

resource "aws_acm_certificate_validation" "cert" {
  provider        = aws.cert_provider
  certificate_arn = aws_acm_certificate.cert.arn
}

data "cloudflare_zones" "domain" {
  filter {
    name = var.site_domain
  }
}

resource "cloudflare_record" "site_cert_val" {
  zone_id = data.cloudflare_zones.domain.zones[0].id

  name  = aws_acm_certificate.cert.domain_validation_options.*.resource_record_name[0]
  type  = aws_acm_certificate.cert.domain_validation_options.*.resource_record_type[0]
  value = trimsuffix(aws_acm_certificate.cert.domain_validation_options.*.resource_record_value[0], ".")

  // Must be set to false. ACM validation false otherwise
  proxied = false
}

// This configuration uses Cloudfront defaults
// Cloudfront is required for static site hosting with S3 if bucket name is
// already taken.
resource "aws_cloudfront_distribution" "site_dist" {
  origin {
    domain_name = aws_s3_bucket_website_configuration.site.website_endpoint
    origin_id   = aws_s3_bucket.site.id
    custom_origin_config {
      http_port              = "80"
      https_port             = "443"
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1", "TLSv1.1", "TLSv1.2"]
    }
  }
  enabled             = true
  default_root_object = "index.html"

  aliases = [
    var.site_domain, "www.${var.site_domain}"
  ]

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = aws_s3_bucket.site.id
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    viewer_protocol_policy = "allow-all"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate_validation.cert.certificate_arn
    ssl_support_method  = "sni-only"
  }
}

resource "cloudflare_record" "site_cname" {
  zone_id = data.cloudflare_zones.domain.zones[0].id
  name    = var.site_domain
  value   = aws_cloudfront_distribution.site_dist.domain_name
  type    = "CNAME"

  ttl     = 1
  proxied = true
}

resource "cloudflare_record" "site_www" {
  zone_id = data.cloudflare_zones.domain.zones[0].id
  name    = "www"
  value   = aws_cloudfront_distribution.site_dist.domain_name
  type    = "CNAME"

  ttl     = 1
  proxied = true
}

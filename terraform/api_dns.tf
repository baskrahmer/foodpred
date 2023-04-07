resource "aws_acm_certificate" "api_cert" {
  domain_name       = "api.${var.site_domain}"
  validation_method = "DNS"

  #  subject_alternative_names = [var.site_domain, "*.${var.site_domain}"]

  tags = {
    Name = var.site_domain
  }
}


resource "cloudflare_record" "api_acm" {
  zone_id = data.cloudflare_zones.domain.zones[0].id

  name  = aws_acm_certificate.api_cert.domain_validation_options.*.resource_record_name[0]
  type  = aws_acm_certificate.api_cert.domain_validation_options.*.resource_record_type[0]
  value = trimsuffix(aws_acm_certificate.api_cert.domain_validation_options.*.resource_record_value[0], ".")

  // Must be set to false. ACM validation false otherwise
  proxied = false
}


resource "cloudflare_record" "api" {
  zone_id = data.cloudflare_zones.domain.zones[0].id

  name  = "api"
  value = aws_apigatewayv2_domain_name.api_domain.domain_name_configuration[0].target_domain_name
  type  = "CNAME"

  ttl     = 1
  proxied = true
}

resource "aws_apigatewayv2_domain_name" "api_domain" {
  domain_name = "api.${var.site_domain}"
  domain_name_configuration {
    certificate_arn = aws_acm_certificate.api_cert.arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }
}

resource "aws_apigatewayv2_api_mapping" "lambda_mapping" {
  api_id      = aws_apigatewayv2_api.lambda.id
  domain_name = aws_apigatewayv2_domain_name.api_domain.domain_name
  stage       = aws_apigatewayv2_stage.default.name
}

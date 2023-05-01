output "bucket_url" {
  value = aws_s3_bucket.site.bucket_domain_name
}

output "api_gateway_url" {
  value = aws_apigatewayv2_domain_name.api_domain.domain_name_configuration[0].target_domain_name
}

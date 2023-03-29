output "bucket_name" {
  value = aws_s3_bucket.site.bucket
}

output "bucket_url" {
  value = aws_s3_bucket.site.bucket_domain_name
}

output "api_gateway_url" {
  value = aws_apigatewayv2_api.lambda.api_endpoint
}

output "invoke_url" {
  value = aws_apigatewayv2_stage.default.invoke_url
}

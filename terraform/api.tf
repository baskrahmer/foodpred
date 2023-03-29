resource "aws_apigatewayv2_api" "lambda" {
  name                         = "apigw-http-lambda"
  protocol_type                = "HTTP"
  description                  = "Example"
  disable_execute_api_endpoint = true

  cors_configuration {
    allow_credentials = false
    allow_headers     = []
    allow_methods     = [
      "GET",
      "HEAD",
      "OPTIONS",
      "POST",
    ]
    allow_origins = [
      "*",
    ]
    expose_headers = []
    max_age        = 0
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id = aws_apigatewayv2_api.lambda.id

  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw.arn

    format = jsonencode({
      requestId               = "$context.requestId"
      sourceIp                = "$context.identity.sourceIp"
      requestTime             = "$context.requestTime"
      protocol                = "$context.protocol"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      routeKey                = "$context.routeKey"
      status                  = "$context.status"
      responseLength          = "$context.responseLength"
      integrationErrorMessage = "$context.integrationErrorMessage"
    }
    )
  }
  depends_on = [aws_cloudwatch_log_group.api_gw]
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id = aws_apigatewayv2_api.lambda.id

  integration_uri  = aws_lambda_function.lambda_test_function.invoke_arn
  integration_type = "AWS_PROXY"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.lambda.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

variable "apigw_log_retention" {
  default = 30
}

resource "aws_cloudwatch_log_group" "api_gw" {
  name = "/aws/api_gw/${aws_apigatewayv2_api.lambda.name}"

  retention_in_days = var.apigw_log_retention
}

resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_test_function.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.lambda.execution_arn}/*/*"
}
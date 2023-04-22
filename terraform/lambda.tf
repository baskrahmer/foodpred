data "aws_iam_policy_document" "lambda_assume_role_policy" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_role" {
  name                = "lambda-lambdaRole-waf"
  assume_role_policy  = data.aws_iam_policy_document.lambda_assume_role_policy.json
  managed_policy_arns = ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"]
}

#data "archive_file" "python_lambda_package" {
#  type        = "zip"
#  source_file = "${path.module}/../lambda_function.py"
#  output_path = "nametest.zip"
#}

resource "aws_lambda_function" "lambda_test_function" {
  function_name = "foodpred"
  image_uri     = "594760667920.dkr.ecr.eu-central-1.amazonaws.com/foodpred:v0"
  package_type  = "Image"
  role          = aws_iam_role.lambda_role.arn
  timeout       = 30
  memory_size   = 3008 # max 3008
  ephemeral_storage {
    size = 4096 # Min 512 MB and the Max 10240 MB
  }
}

resource "aws_cloudwatch_event_rule" "lambda_test" {
  name                = "run-lambda-function"
  description         = "Schedule lambda function"
  schedule_expression = "rate(60 minutes)"
}

resource "aws_cloudwatch_event_target" "lambda_function_target" {
  target_id = "lambda-function-target"
  rule      = aws_cloudwatch_event_rule.lambda_test.name
  arn       = aws_lambda_function.lambda_test_function.arn
}

resource "aws_lambda_permission" "allow_cloudwatch" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_test_function.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.lambda_test.arn
}

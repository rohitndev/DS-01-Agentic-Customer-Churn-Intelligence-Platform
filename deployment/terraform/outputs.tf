output "api_endpoint" {
  description = "Public API Gateway endpoint URL"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.churn_api.function_name
}

output "model_bucket" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_store.bucket
}

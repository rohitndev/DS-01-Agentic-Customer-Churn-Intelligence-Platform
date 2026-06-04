# deployment/terraform/main.tf — AWS infrastructure: Lambda + API Gateway + S3
# GCP equivalent is in infrastructure/modules/gcp_cloudrun.tf

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------- S3 bucket for model artifacts ----------
resource "aws_s3_bucket" "model_store" {
  bucket = "${var.project_name}-models-${var.environment}"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  versioning_configuration { status = "Enabled" }
}

# ---------- Lambda function ----------
resource "aws_lambda_function" "churn_api" {
  function_name = "${var.project_name}-api-${var.environment}"
  package_type  = "Image"
  image_uri     = "${var.ecr_image_uri}"
  role          = aws_iam_role.lambda_exec.arn
  timeout       = 30
  memory_size   = 1024

  environment {
    variables = {
      GROQ_API_KEY = var.groq_api_key
      ENVIRONMENT  = var.environment
    }
  }
  tags = local.tags
}

# ---------- API Gateway ----------
resource "aws_apigatewayv2_api" "churn_api" {
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"
  tags          = local.tags
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.churn_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.churn_api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.churn_api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.churn_api.id
  name        = "$default"
  auto_deploy = true
}

# ---------- IAM ----------
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-role-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

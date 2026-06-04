variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "ds01-churn"
}

variable "environment" {
  description = "Deployment environment: dev | staging | prod"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "ap-south-1"
}

variable "ecr_image_uri" {
  description = "Full ECR image URI (account.dkr.ecr.region.amazonaws.com/repo:tag)"
  type        = string
}

variable "groq_api_key" {
  description = "Groq API key for the retention agent LLM"
  type        = string
  sensitive   = true
}

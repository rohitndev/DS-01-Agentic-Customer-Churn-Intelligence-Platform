#!/usr/bin/env bash
# deployment/ecr_push.sh — Build and push Docker image to AWS ECR.
#
# Usage:
#   export AWS_REGION=ap-south-1
#   export AWS_ACCOUNT_ID=123456789012
#   bash deployment/ecr_push.sh

set -euo pipefail

REPO_NAME="ds01-churn-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGION="${AWS_REGION:-ap-south-1}"
ACCOUNT="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "==> Authenticating to ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Creating ECR repository (if it doesn't exist)..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

echo "==> Building image..."
docker build -t "${REPO_NAME}:${IMAGE_TAG}" -f deployment/Dockerfile .

echo "==> Tagging and pushing to ${ECR_URI}..."
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

echo "==> Done: ${ECR_URI}"

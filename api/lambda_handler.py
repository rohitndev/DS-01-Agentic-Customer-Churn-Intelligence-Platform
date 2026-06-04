"""
api/lambda_handler.py — AWS Lambda handler wrapping the FastAPI app via
Mangum for serverless deployment (API Gateway → Lambda).

Deploy:
  1. Build Docker image: docker build -t ds01-churn .
  2. Push to ECR: see deployment/ecr_push.sh
  3. Lambda function handler: api.lambda_handler.handler
"""

import sys

sys.path.insert(0, "/var/task")   # Lambda working directory

# Mangum bridges ASGI (FastAPI) → Lambda event/context
try:
    from mangum import Mangum
    from api.server import app
    handler = Mangum(app, lifespan="off")
except ImportError:
    # Mangum not installed locally — only needed in Lambda environment
    handler = None
    print("[lambda_handler] Mangum not installed. Install it for Lambda deployment: pip install mangum")


def predict_handler(event: dict, context) -> dict:
    """
    Alternative direct Lambda handler (no API Gateway) for batch scoring jobs.
    Expects event = { "customer": {...customer fields...} }
    Returns { "statusCode": 200, "body": {...prediction...} }
    """
    import json
    from src.models.predictor import predict

    try:
        customer = event.get("customer", event)
        result = predict(customer)
        return {"statusCode": 200, "body": json.dumps(result)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

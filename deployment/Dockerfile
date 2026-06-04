FROM public.ecr.aws/lambda/python:3.12

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /var/task

# Copy dependency files first (better layer caching)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# Copy source code
COPY src/       ./src/
COPY api/       ./api/
COPY mlops/     ./mlops/
COPY models/    ./models/

# Lambda handler entry point
CMD ["api.lambda_handler.handler"]

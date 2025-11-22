# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY requirements.txt ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY predict.py ./
COPY reports/ ./reports/

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Set environment variables for the predictor
ENV CATBOOST_MODEL_PATH=reports/catboost_model.cbm
ENV FEATURE_STATS_PATH=reports/feature_stats.json
ENV VIRAL_SCORE_THRESHOLD=250
ENV TITLE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Expose port for the API (using Cog's default)
EXPOSE 5000

# Activate venv and run the Cog prediction server
CMD ["/bin/bash", "-c", "source .venv/bin/activate && python -m cog.server.http"]


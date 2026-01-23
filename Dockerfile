# Use Python 3.13 slim image for compatibility with Singularity
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for the ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-install-project --no-dev

# Copy the source code
COPY src/ ./src/

# Install the project in editable mode
RUN uv pip install -e .

# Create data directory (data will be mounted or copied at runtime for HPC)
RUN mkdir -p data

# Create outputs directory for results
RUN mkdir -p outputs

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"

# Default command - can be overridden in Slurm scripts
CMD ["uv", "run", "python", "-m", "budget"]

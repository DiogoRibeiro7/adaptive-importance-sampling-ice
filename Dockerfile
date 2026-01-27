# Multi-stage Dockerfile for Safe-ICE

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install --no-cache-dir poetry==1.6.1

# Export requirements
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy package source
COPY safe_ice ./safe_ice
COPY setup.py README.md LICENSE ./

# Install the package
RUN pip install --no-cache-dir --user .

# Stage 2: Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-base \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash safeice

# Copy installed packages from builder
COPY --from=builder /root/.local /home/safeice/.local

# Set environment variables
ENV PATH=/home/safeice/.local/bin:$PATH
ENV PYTHONPATH=/home/safeice/.local/lib/python3.11/site-packages:$PYTHONPATH

# Set working directory
WORKDIR /workspace

# Copy examples for demonstration
COPY --chown=safeice:safeice examples /workspace/examples

# Switch to non-root user
USER safeice

# Default command
CMD ["safe-ice", "--help"]

# Labels
LABEL maintainer="Safe-ICE Contributors"
LABEL description="Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations"
LABEL version="0.1.0"

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import safe_ice; print('OK')" || exit 1
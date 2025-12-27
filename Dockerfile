# Use a modern Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies (FFMPEG is critical for librosa)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies into the system site-packages (simpler for Docker)
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY src/ ./src/
COPY models/ ./models/

# Create a non-root user for security (Hugging Face recommendation)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose the HF default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Launch the standalone app
ENTRYPOINT ["streamlit", "run", "src/ui/app_standalone.py", "--server.port=8501", "--server.address=0.0.0.0"]

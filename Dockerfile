FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no venv needed in container)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY portfolio/ portfolio/
COPY projects/ projects/

# Install the project itself
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "portfolio.app:app", "--host", "0.0.0.0", "--port", "8000"]

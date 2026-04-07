# Mirage_RL — Production Query Optimizer Environment
# Root-level Dockerfile for Hugging Face Spaces deployment.
# Delegates to the build logic defined in server/Dockerfile.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /home/user/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=Mirage_RL

# Copy full project context
COPY . /home/user/app/env
WORKDIR /home/user/app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

# Add non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --from=builder --chown=user:user /home/user/app/env/.venv $HOME/app/.venv
COPY --from=builder --chown=user:user /home/user/app/env       $HOME/app/env

ENV PATH="$HOME/app/.venv/bin:$PATH"
ENV PYTHONPATH="$HOME/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd $HOME/app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]

# benchmarks/beam/runner.Dockerfile
#
# Small orchestrator image for the bench. Lives separately from BrainDB's
# main image because:
#   - it needs the docker CLI + compose plugin (to recreate api_bench
#     between conversations)
#   - it only needs bench-side Python deps (no FastAPI, no embeddings, no
#     pgvector — just enough to drive the bench)
#
# Built only when running the bench:
#   docker compose -f docker-compose.bench.yml --profile runner build bench_runner
# Run with:
#   docker compose -f docker-compose.bench.yml run --rm bench_runner \
#     python -m benchmarks.beam.bench --split 100K --limit 1

FROM python:3.12-slim

# Install docker CLI + compose plugin directly from upstream static builds.
# The apt `docker.io` package on python:3.12-slim only ships docker-init,
# not the `docker` CLI itself, so we grab the official static binaries.
ARG DOCKER_VERSION=27.3.1
ARG COMPOSE_VERSION=v2.30.3
RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
       | tar -xzC /usr/local/bin --strip-components=1 docker/docker \
    && chmod +x /usr/local/bin/docker \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
       -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version \
    && docker compose version

# Bench Python deps. Repo code itself comes in via the `.:/app` bind mount;
# we just need the runtime libraries here. `docker` is the Python SDK used
# by bench.py to recreate api_bench between conversations (avoids the
# compose-from-inside-container relative-path resolution issue).
RUN pip install --no-cache-dir \
    datasets \
    huggingface_hub \
    requests \
    psycopg2-binary \
    docker

WORKDIR /app
ENV PYTHONPATH=/app

# Default: sit idle so `docker compose run --rm bench_runner <cmd>` can
# spawn ephemeral instances with whatever command we want.
CMD ["sleep", "infinity"]

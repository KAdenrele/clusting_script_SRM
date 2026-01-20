FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv_bin /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

# --frozen: ensures uv doesn't try to update the lockfile
# --no-cache: keeps the image size small
RUN uv sync --frozen --no-cache
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

VOLUME /data

CMD ["python", "clustering_local.py"]

#docker build -t clustering-app . && docker run --rm --gpus all -v /mnt/data2/sample_curated:/data -v /home/ade/outputs:/app/output clustering-app

# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY setup.py .
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -e ".[api,train]"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FT_PROJECT=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
COPY . .

# Non-root user
RUN groupadd -r mkd && useradd -r -g mkd -d /workspace mkd && \
    chown -R mkd:mkd /workspace
USER mkd

EXPOSE 7860 8000

CMD ["python3.11", "app.py"]

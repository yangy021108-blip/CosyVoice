FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG INSTALL_VLLM=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libsox-dev \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /tmp/requirements-runtime.txt
COPY requirements-vllm.txt /tmp/requirements-vllm.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt
RUN if [ "$INSTALL_VLLM" = "true" ]; then \
        python -m pip install --no-cache-dir -r /tmp/requirements-vllm.txt; \
    fi

WORKDIR /workspace/CosyVoice
ENV PYTHONPATH=/workspace/CosyVoice

CMD ["python", "-m", "api_server.main"]

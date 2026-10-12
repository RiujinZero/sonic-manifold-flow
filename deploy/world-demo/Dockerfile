FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04's default python3 is 3.10, which lacks enum.StrEnum (3.11+)
# that this codebase relies on. Use deadsnakes for a supported interpreter
# instead of Ubuntu's stock package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv git libglib2.0-0 libgl1 build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN python3.12 -m ensurepip --upgrade

WORKDIR /app
COPY . /app
RUN python3.12 -m pip install --upgrade pip setuptools wheel && \
    python3.12 -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
      -e '.[world]'

# YOLOX is installed from its official source rather than the stale PyPI
# package, whose pinned ONNX exporter does not build here; the ONNX
# simplifier is intentionally skipped since it is not used for PyTorch
# inference. See docs/world-demo.md.
RUN git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git /opt/yolox && \
    python3.12 -m pip install --no-build-isolation --no-deps -e /opt/yolox && \
    python3.12 -m pip install loguru thop ninja tabulate tensorboard pycocotools

RUN useradd --create-home --uid 10001 demo && mkdir -p /models /home/demo/.cache && \
    chown demo:demo /models /home/demo/.cache
USER demo

ENV MANIFOLDFLOW_HOST=0.0.0.0 MANIFOLDFLOW_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3.12 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["manifoldflow-world-demo"]

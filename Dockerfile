# Moore-AnimateAnyone has a pinned legacy dependency stack targeting CUDA 11.7.
# Build this image on a RunPod GPU worker; model weights are downloaded lazily on first job.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANIMATEANYONE_ROOT=/app/Moore-AnimateAnyone \
    AUTO_DOWNLOAD_WEIGHTS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git git-lfs ffmpeg wget ca-certificates \
    build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/MooreThreads/Moore-AnimateAnyone.git /app/Moore-AnimateAnyone
WORKDIR /app/Moore-AnimateAnyone

# Upstream pins the versions used by its inference implementation.
# Keep pkg_resources available: newer setuptools releases removed it, while the
# upstream CLIP setup script imports it during wheel metadata generation.
RUN python3.10 -m pip install --upgrade pip wheel "setuptools==69.5.1" && \
    python3.10 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu117 && \
    python3.10 -m pip install --no-build-isolation -r requirements.txt

# Handler dependencies are installed directly here; no separate requirements.txt needed.
RUN python3.10 -m pip install "runpod>=1.7.0,<2.0.0" "requests>=2.31.0,<3.0.0"

WORKDIR /app
COPY handler.py /app/handler.py

# Do not bake several GB of checkpoints into the image. They download to the container
# at first request; use a persistent volume or pre-populate pretrained_weights for faster cold starts.
CMD ["python3.10", "-u", "/app/handler.py"]

# RunPod Serverless image for Moore-AnimateAnyone full-body pose transfer.
# CUDA base remains 12.4.1; PyTorch wheels are installed for CUDA 12.4.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANIMATEANYONE_ROOT=/app/Moore-AnimateAnyone \
    AUTO_DOWNLOAD_WEIGHTS=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git git-lfs ffmpeg wget ca-certificates \
    build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip wheel "setuptools==69.5.1"

# PyTorch 2.4.1 has matching CUDA 12.4 wheels.
RUN python3.10 -m pip install \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN git clone --depth 1 https://github.com/MooreThreads/Moore-AnimateAnyone.git /app/Moore-AnimateAnyone
WORKDIR /app/Moore-AnimateAnyone

# Install the model runtime dependencies without overwriting the CUDA 12.4 PyTorch wheels.
RUN python3.10 -m pip install \
    "huggingface_hub==0.25.2" "transformers==4.30.2" "diffusers==0.24.0" \
    "accelerate==0.21.0" "av==11.0.0" "decord==0.6.0" "einops==0.4.1" \
    "imageio==2.33.0" "imageio-ffmpeg==0.4.9" "numpy==1.23.5" \
    "omegaconf==2.2.3" "onnxruntime-gpu==1.16.3" \
    "opencv-contrib-python==4.8.1.78" "Pillow==9.5.0" \
    "scikit-image==0.21.0" "scikit-learn==1.3.2" "scipy==1.11.4" \
    "torchdiffeq==0.2.3" "torchmetrics==1.2.1" "torchsde==0.2.5" \
    "tqdm==4.66.1" "open-clip-torch==2.20.0" "controlnet-aux==0.0.7" \
    "ftfy" "regex" "sentencepiece" "protobuf" "safetensors" \
    "runpod>=1.7.0,<2.0.0" "requests>=2.31.0,<3.0.0"

# Install OpenAI CLIP separately using the upstream-supported Git URL.
RUN python3.10 -m pip install \
    "git+https://github.com/openai/CLIP.git"

# Download weights during image build so the first serverless request does not download them.
# This step takes time and makes the container image large.
RUN python3.10 tools/download_weights.py

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python3.10", "-u", "/app/handler.py"]

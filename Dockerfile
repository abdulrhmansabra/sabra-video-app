FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir runpod torch torchvision torchaudio accelerate transformers diffusers safetensors huggingface_hub opencv-python

# تحميل وحرق النماذج مباشرة داخل الصورة أثناء البناء على GitHub
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance/AnimateDiff-Lightning', subfolder='animatediff_motion_adapter_4step', local_dir='/app/models/animatediff_adapter', max_workers=1)"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='emilianJR/Realistic_Vision_v5.0_noVAE', local_dir='/app/models/realistic_vision', max_workers=1)"

# نسخ ملف المعالجة
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

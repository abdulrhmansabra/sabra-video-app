FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir runpod torch torchvision torchaudio accelerate transformers diffusers safetensors huggingface_hub

# استقبال رمز Hugging Face الممرر من GitHub Actions
ARG HF_TOKEN

# تحميل النموذج باستخدام المفتاح وبشكل آمن ومستقر لمنع انهيار الذاكرة
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Lightricks/LTX-Video', local_dir='/app/models/ltx-video', token='${HF_TOKEN}', max_workers=1, resume_download=True)"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

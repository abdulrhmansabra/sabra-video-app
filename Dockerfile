FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir runpod torch torchvision torchaudio accelerate transformers diffusers safetensors huggingface_hub

# تحميل نموذج LTX-Video وحرقه داخل الحاوية أثناء البناء السحابي
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Lightricks/LTX-Video', local_dir='/app/models/ltx-video')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

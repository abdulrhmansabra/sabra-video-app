FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت المكتبات مع إضافة hf_transfer للتحميل السريع والمستقر
RUN pip3 install --no-cache-dir runpod torch torchvision torchaudio accelerate transformers diffusers safetensors huggingface_hub hf_transfer

# تفعيل نظام التحميل المحسّن لتجنب أخطاء التدفق والخوادم
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# استقبال رمز Hugging Face
ARG HF_TOKEN

# تحميل نموذج LTX-Video مع تفعيل استئناف التحميل (resume_download)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Lightricks/LTX-Video', local_dir='/app/models/ltx-video', token='${HF_TOKEN}', resume_download=True)"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

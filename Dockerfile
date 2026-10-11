FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (Nightly) لدعم كروت Blackwell (sm_120)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --pre torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu124

# باقي المكتبات
RUN pip3 install --no-cache-dir \
    runpod accelerate transformers diffusers safetensors huggingface_hub \
    opencv-python requests

# (اختياري) فحص سريع عند التشغيل — يطبع نسخة torch وبيانات GPU
# يمكنك إزالة هذا الجزء إذا لا تريده
ENV PYTHONUNBUFFERED=1

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

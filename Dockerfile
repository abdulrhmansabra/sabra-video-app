FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# تثبيت PyTorch المستقر (Stable) مع دعم CUDA 12.4
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# باقي المكتبات المطلوبة
RUN pip3 install --no-cache-dir \
    runpod accelerate transformers diffusers safetensors huggingface_hub \
    opencv-python requests

ENV PYTHONUNBUFFERED=1

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

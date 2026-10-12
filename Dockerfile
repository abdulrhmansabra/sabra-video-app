FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# تثبيت الحزم الأساسية ومكتبات النظام لـ OpenCV و Video Processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget ca-certificates \
    libxcb1 libx11-xcb1 libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# تثبيت PyTorch المستقر مع دعم CUDA 12.4
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# تثبيت أحدث نسخ من diffusers وبقية المكتبات اللازمة لـ Wan2.1 مع إضافة imageio و OpenCV
RUN pip3 install --no-cache-dir \
    runpod accelerate transformers "diffusers>=0.33.0" safetensors huggingface_hub \
    opencv-python requests sentencepiece protobuf imageio imageio-ffmpeg

ENV PYTHONUNBUFFERED=1

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

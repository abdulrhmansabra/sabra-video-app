FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# تثبيت الحزم الأساسية ومكتبات النظام لـ FFmpeg والمعالجة
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# تثبيت PyTorch المستقر مع دعم CUDA 12.4
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# تثبيت diffusers من المصدر لضمان دعم Wan2.2 مع بقية الحزم المطلوبة
RUN pip3 install --no-cache-dir \
    runpod accelerate transformers git+https://github.com/huggingface/diffusers.git safetensors huggingface_hub \
    numpy pillow opencv-python-headless requests sentencepiece protobuf imageio imageio-ffmpeg ftfy regex

ENV PYTHONUNBUFFERED=1

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

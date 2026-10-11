FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت PyTorch متوافق مع CUDA 12.1 لضمان دعم الكروت الحديثة بدون أخطاء
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# تثبيت باقي مكتبات الذكاء الاصطناعي والتواصل
RUN pip3 install --no-cache-dir runpod accelerate transformers diffusers safetensors huggingface_hub opencv-python requests

# نسخ ملف المعالجة
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

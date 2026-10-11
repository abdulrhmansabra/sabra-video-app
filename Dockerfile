FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت PyTorch المخصص لكرت الشاشة أولاً وبشكل منفصل ومستقر
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# تثبيت باقي مكتبات الذكاء الاصطناعي والتواصل
RUN pip3 install --no-cache-dir runpod accelerate transformers diffusers safetensors huggingface_hub opencv-python

# نسخ ملف المعالجة
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

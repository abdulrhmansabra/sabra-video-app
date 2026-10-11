FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت المكتبات اللازمة لـ Diffusers و AnimateDiff مع PyTorch
RUN pip3 install --no-cache-dir runpod torch torchvision torchaudio accelerate transformers diffusers safetensors huggingface_hub opencv-python

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


# تثبيت الحزم الأساسية ومكتبات النظام
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# تثبيت PyTorch المستقر مع دعم CUDA 12.4
RUN python3 -m pip install --no-cache-dir --upgrade "pip==24.0"
# تثبيت أحدث نسخة من diffusers حصرياً من الـ main branch بدون كاش
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# تثبيت إصدار diffusers المتوافق مع المشروع
RUN python3 -m pip install --no-cache-dir \
    "av==12.0.0" \
    "diffusers==0.24.0" \
    "transformers==4.30.2" \
    "accelerate==0.21.0" \
    "huggingface_hub==0.25.2" \
    "numpy==1.23.5" \
    "omegaconf==2.2.3" \
    "decord==0.6.0" \
    "einops==0.4.1" \
    "imageio==2.33.0" \
    "imageio-ffmpeg==0.4.9" \
    "opencv-contrib-python==4.8.1.78" \
    "Pillow==9.5.0" \
    "scipy==1.11.4" \
    "scikit-image==0.21.0" \
    "scikit-learn==1.3.2" \
    "torchdiffeq==0.2.3" \
    "torchmetrics==1.2.1" \
    "torchsde==0.2.5" \
    "onnxruntime-gpu==1.16.3" \
    "open-clip-torch==2.20.0" \
    "controlnet-aux==0.0.7" \
    "protobuf==3.20.3" \
"opentelemetry-proto==1.20.0" \
"opentelemetry-exporter-otlp-proto-http==1.20.0" \
ftfy regex sentencepiece safetensors \
runpod requests
    
ENV PYTHONUNBUFFERED=1

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ==================================================
# 1. تثبيت متطلبات النظام
# ==================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ca-certificates \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ==================================================
# 2. تثبيت إصدار pip المتوافق مع المكتبات القديمة
# ==================================================
RUN python3 -m pip install --upgrade "pip==24.0"

# ==================================================
# 3. تثبيت PyTorch المتوافق مع CUDA 12.4
# ==================================================
RUN python3 -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ==================================================
# 4. تثبيت مكتبات Moore-AnimateAnyone
# ==================================================
RUN python3 -m pip install \
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
    "matplotlib==3.8.4" \
    "protobuf==3.20.3" \
    "opentelemetry-proto==1.20.0" \
    "opentelemetry-exporter-otlp-proto-http==1.20.0" \
    ftfy \
    regex \
    sentencepiece \
    safetensors \
    runpod \
    requests

# ==================================================
# 5. التحقق من PyTorch وPyAV
# ==================================================
RUN python3 -c "import matplotlib; print('Matplotlib OK:', matplotlib.__version__)"
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda)" \
    && python3 -c "import av; print('PyAV:', av.__version__)"

# ==================================================
# 6. تنزيل المستودع الرسمي أثناء بناء الصورة
# ==================================================
RUN git clone --depth 1 \
    https://github.com/MooreThreads/Moore-AnimateAnyone.git \
    /app/Moore-AnimateAnyone

# ==================================================
# 7. ضبط مسارات المشروع
# ==================================================
ENV ANIMATEANYONE_ROOT=/app/Moore-AnimateAnyone
ENV PYTHONPATH=/app/Moore-AnimateAnyone

WORKDIR /app/Moore-AnimateAnyone

# ==================================================
# 8. التحقق من ملفات المشروع الأساسية
# ==================================================
RUN test -f /app/Moore-AnimateAnyone/tools/download_weights.py \
    && test -f /app/Moore-AnimateAnyone/configs/prompts/animation.yaml \
    && test -f /app/Moore-AnimateAnyone/src/utils/util.py

# ==================================================
# 9. تنزيل أوزان النموذج أثناء بناء الصورة
# ==================================================

# إصلاح اسم مستودع image_encoder في سكربت التنزيل الرسمي
RUN sed -i \
    's|lambdalabs/sd-image-variations-diffusers|lambda/sd-image-variations-diffusers|g' \
    /app/Moore-AnimateAnyone/tools/download_weights.py \
    && python3 /app/Moore-AnimateAnyone/tools/download_weights.py


# ==================================================
# 10. التحقق من وجود الأوزان المطلوبة
# ==================================================
RUN test -d /app/Moore-AnimateAnyone/pretrained_weights

# ==================================================
# 11. نسخ معالج RunPod بعد تنزيل الأوزان
# ==================================================
WORKDIR /app

COPY handler.py /app/handler.py

# ==================================================
# 12. تشغيل RunPod Serverless Handler
# ==================================================
CMD ["python3", "-u", "/app/handler.py"]

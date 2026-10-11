# اختيار صورة أساسية داعمة لـ Python و CUDA لكرت الشاشة
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# تثبيت الحزم الأساسية وبايثون
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت مكتبات بايثون المطلوبة
RUN pip3 install --no-cache-dir runpod torch torchvision torchaudio

# نسخ كود المعالجة الخاص بـ RunPod
COPY handler.py /app/handler.py

# أمر التشغيل الافتراضي عند إيقاظ السيرفر
CMD ["python3", "-u", "/app/handler.py"]

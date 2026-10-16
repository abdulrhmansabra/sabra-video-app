FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# تثبيت الحزم الأساسية ومكتبات النظام المطلوبة للمعالجة و FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# تثبيت PyTorch المستقر مع دعم CUDA 12.4
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# استنساخ مستودع Moore-AnimateAnyone الرسمي
RUN git clone https://github.com/MooreThreads/Moore-AnimateAnyone.git /app/Moore-AnimateAnyone

# الانتقال للمستودع وتثبيت المتطلبات الخاصة به
WORKDIR /app/Moore-AnimateAnyone
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir runpod accelerate transformers safetensors huggingface_hub opencv-python-headless requests

# العودة لمجلد العمل الرئيسي ونسخ الـ handler
WORKDIR /app
COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "/app/handler.py"]

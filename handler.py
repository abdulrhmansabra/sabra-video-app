import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback
import threading

from diffusers import WanPipeline
from diffusers.utils import export_to_video
import runpod

print("جاري تحميل وتهيئة نموذج Wan2.1...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

# قفل تزامني لحماية الحالة العالمية للـ Pipeline من التداخل
pipeline_lock = threading.Lock()

# -------------------------
# Model load at startup
# -------------------------
try:
    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    pipeline = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        **_token_kwargs()
    )
    
    pipeline.to("cuda")

    print("نموذج Wan2.1 جاهز بالكامل للاستجابة!")
    MODEL_READY = True

except Exception as e:
    print("="*50)
    print("❌ فشل تحميل النماذج أثناء الإقلاع:")
    traceback.print_exc()
    print("="*50)
    MODEL_READY = False
    pipeline = None


# -------------------------
# Frame Normalization Helpers
# -------------------------
def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x

def normalize_frame(frame):
    frame = to_numpy(frame)
    frame = np.asarray(frame)

    frame = np.squeeze(frame)

    if frame.ndim == 2:
        frame = frame[..., None]

    while frame.ndim > 3:
        frame = frame[0]

    if frame.ndim == 2:
        frame = frame[..., None]

    # معالجة channels-first لو كانت (C, H, W) -> (H, W, C)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    # تصحيح النطاق اللوني لمنع الألوان المعكوسة أو الحرارية
    if np.issubdtype(frame.dtype, np.floating) or frame.min() < 0 or frame.max() <= 1.0:
        if frame.min() < 0:
            frame = (frame + 1.0) / 2.0
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    else:
        frame = frame.clip(0, 255).astype(np.uint8)

    return frame

def normalize_video_frames(video_frames):
    if not isinstance(video_frames, (list, tuple)):
        video_frames = list(video_frames)

    out = []
    for idx, fr in enumerate(video_frames):
        try:
            out.append(normalize_frame(fr))
        except Exception as e:
            print(f"⚠️ تحذير: فشل تطبيع الإطار رقم {idx}: {e}")
    return out


# -------------------------
# Helpers
# -------------------------
def load_image_from_input(img_input: str):
    if img_input is None:
        return None
    if not isinstance(img_input, str):
        print(f"image must be a string. got: {type(img_input)}")
        return None

    img_input = img_input.strip()
    if img_input == "":
        return None

    try:
        if img_input.startswith("http://") or img_input.startswith("https://"):
            response = requests.get(img_input, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if "," in img_input:
                img_input = img_input.split(",", 1)[1]
            data = base64.b64decode(img_input)
            return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"خطأ في قراءة الصورة: {e}")
        return None


# -------------------------
# RunPod Handler
# -------------------------
def handler(event):
    if not MODEL_READY or pipeline is None:
        return {"status": "error", "message": "النماذج لم تُحمّل بنجاح عند الإقلاع. راجع Logs للتفاصيل."}

    try:
        job_input = event.get("input") or {}

        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt")

        image_input = job_input.get("image")
        pil_img = load_image_from_input(image_input)
        has_image = pil_img is not None

        if prompt is None or (isinstance(prompt, str) and prompt.strip() == ""):
            if has_image:
                prompt = "high quality, cinematic"
            else:
                return {"status": "error", "message": "لازم ترسل prompt أو image على الأقل."}

        if negative_prompt is None:
            negative_prompt = "bright, overexposed, static, blurry"

        num_inference_steps = int(job_input.get("num_inference_steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))
        num_frames = int(job_input.get("num_frames", 33))
        fps = int(job_input.get("fps", 16))
        
        width = int(job_input.get("width", 512))
        height = int(job_input.get("height", 384))

        print(f"🚀 بدء معالجة الطلب: prompt='{prompt}', steps={num_inference_steps}, frames={num_frames}, size=({width}x{height})")

        with pipeline_lock:
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                height=height,
                width=width,
            ).frames

        torch.cuda.empty_cache()

        if output is None:
            return {"status": "error", "message": "فشل التوليد: لم يتم إرجاع إطارات من الـ pipeline."}

        raw_frames = output[0] if isinstance(output, (list, tuple)) and len(output) > 0 else output
        
        # طباعة تفاصيل التشخيص للأبعاد للتأكد منها
        print(f"🔍 تشخيص الإطارات الخام -> type: {type(raw_frames)}, shape/len: {getattr(raw_frames, 'shape', len(raw_frames) if hasattr(raw_frames, '__len__') else 'unknown')}")

        video_frames = normalize_video_frames(raw_frames)

        if video_frames is None or len(video_frames) == 0:
            return {"status": "error", "message": "قائمة الإطارات فارغة بعد التطبيع."}

        video_path = "/tmp/output.mp4"
        export_to_video(video_frames, video_path, fps=fps)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {"status": "error", "message": "فشل تصدير الفيديو: الملف الناتج فارغ أو غير موجود."}

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("تم توليد وتصدير الفيديو بنجاح!")
        return {"status": "success", "message": "تم توليد الفيديو بنجاح", "video_base64": video_base64}

    except Exception as e:
        print("="*50)
        print("❌ حدث خطأ استثنائي أثناء تنفيذ الطلب (Handler Error):")
        traceback.print_exc()
        print("="*50)
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

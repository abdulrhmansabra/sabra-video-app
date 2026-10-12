import os
import torch
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
    # استخدام نسخة Wan2.1 الخفيفة والمناسبة للـ Serverless
    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

    pipeline = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        **_token_kwargs()
    )
    
    pipeline.to("cuda")

    print("نموذج Wan2.1 جاهز بالكامل للاستجابة!")
    MODEL_READY = True

except Exception as e:
    print(f"فشل تحميل النماذج أثناء الإقلاع: {e}")
    MODEL_READY = False
    pipeline = None


# -------------------------
# Helpers (نفس بنية كودك الأصلي تماماً)
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
            # base64 (قد يكون data URL)
            if "," in img_input:
                img_input = img_input.split(",", 1)[1]
            data = base64.b64decode(img_input)
            return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"خطأ في قراءة الصورة: {e}")
        return None


# -------------------------
# RunPod Handler (محافظ على نفس المدخلات والمخرجات)
# -------------------------
def handler(event):
    if not MODEL_READY or pipeline is None:
        return {"status": "error", "message": "النماذج لم تُحمّل بنجاح عند الإقلاع. راجع Logs."}

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

        num_inference_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))
        num_frames = int(job_input.get("num_frames", 81))
        fps = int(job_input.get("fps", 16))
        
        width = int(job_input.get("width", 832))
        height = int(job_input.get("height", 480))

        with pipeline_lock:
            # تنفيذ الاستدعاء للنموذج الجديد مع الحفاظ على القفل التزامني
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

        video_frames = output[0] if isinstance(output, list) and len(output) > 0 else output
        if not video_frames:
            return {"status": "error", "message": "قائمة الإطارات فارغة."}

        video_path = "/tmp/output.mp4"
        export_to_video(video_frames, video_path, fps=fps)

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "success", "message": "تم توليد الفيديو بنجاح", "video_base64": video_base64}

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

import os
# إعدادات الذاكرة المرنة لمنع تجزئة الذاكرة وأخطاء الـ OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback
import threading

from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
import runpod

print("🚀65 جاري تهيئة وتحميل نموذج تحريك الصور CogVideoX-5B-I2V...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

pipeline_lock = threading.Lock()

# -------------------------
# تحميل نموذج CogVideoX-5B Image-to-Video بالطريقة القياسية النظيفة
# -------------------------
try:
    MODEL_ID = "THUDM/CogVideoX-5b-I2V"
    
    i2v_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        **_token_kwargs()
    )
    # توزيع الحمل بذكاء بين الـ VRAM والـ RAM
    i2v_pipeline.enable_model_cpu_offload()
    
    # تفعيل Slicing للحماية من الـ OOM بدون تسبب في أي عيوب بكسلية
    i2v_pipeline.vae.enable_slicing()

    print("✅ تم تحميل النموذج وتفعيل أدوات التصدير القياسية بنجاح!")
    MODEL_READY = True

except Exception as e:
    print("❌ فشل تحميل النموذج:")
    traceback.print_exc()
    MODEL_READY = False
    i2v_pipeline = None


def load_image_from_input(img_input: str):
    if img_input is None: return None
    if not isinstance(img_input, str): return None
    img_input = img_input.strip()
    if img_input == "": return None
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
    if not MODEL_READY:
        return {"status": "error", "message": "النموذج غير محمل على السيرفر."}

    try:
        job_input = event.get("input") or {}
        
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted, bad motion, checkerboard, grid artifacts")
        
        image_input = job_input.get("image_reference") or job_input.get("image")
        pil_img = load_image_from_input(image_input)

        if pil_img is None:
            return {"status": "error", "message": "يلزم إرسال صورة (image) لتشغيل نموذج التحريك."}

        if not prompt or not prompt.strip():
            return {"status": "error", "message": "يلزم إرسال وصف الحركة (prompt) مع الصورة."}

        num_inference_steps = int(job_input.get("num_inference_steps", 35))
        guidance_scale = float(job_input.get("guidance_scale", 6.0))
        fps = int(job_input.get("fps", 8))
        
        num_frames = int(job_input.get("num_frames", 25))
        width = 720   
        height = 480

        pil_img = pil_img.resize((width, height))
        print(f"🎬 جاري معالجة وتوليد الفيديو بالدقة القياسية للنموذج ({width}x{height}, {num_frames} إطار)...")

        with pipeline_lock:
            output_result = i2v_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_frames=num_frames,
            ).frames

        torch.cuda.empty_cache()

        if output_result is None:
            return {"status": "error", "message": "لم يرجع النموذج أية إطارات."}

        video_path = "/tmp/output_cogvideo_i2v.mp4"
        
        # 🌟 استخدام الدالة الرسمية والمختبرة من diffusers لتصدير الفيديو بأعلى نقاء وألوان صحيحة 100%
        export_to_video(output_result, video_path, fps=fps)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {"status": "error", "message": "ملف الفيديو الناتج فارغ."}

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("✨ تم إخراج الفيديو بنجاح تام وبمعايير الجودة العالمية النظيفة!")
        return {"status": "success", "video_base64": video_base64}

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

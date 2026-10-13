import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback
import threading
import cv2

from diffusers import WanImageToVideoPipeline
import runpod

print("🚀 جاري تهيئة وتحميل نموذج تحريك الصور Wan2.1-I2V-14B (FP8)...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

pipeline_lock = threading.Lock()

# -------------------------
# تحميل نموذج الصورة فقط
# -------------------------
try:
    I2V_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    
    i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
        I2V_MODEL_ID,
        torch_dtype=torch.float8_e4m3fn,  # ضغط FP8 لتوفير VRAM
        **_token_kwargs()
    )
    i2v_pipeline.enable_model_cpu_offload()

    print("✅ تم تحميل نموذج تحريك الصور بنجاح!")
    MODEL_READY = True

except Exception as e:
    print("❌ فشل تحميل النموذج:")
    traceback.print_exc()
    MODEL_READY = False
    i2v_pipeline = None


# -------------------------
# دوال معالجة وتصدير الفيديو
# -------------------------
def to_numpy(x):
    if hasattr(x, "detach"): x = x.detach()
    if hasattr(x, "cpu"): x = x.cpu()
    if hasattr(x, "numpy"): x = x.numpy()
    return x

def normalize_frame(frame):
    frame = to_numpy(frame)
    frame = np.squeeze(np.asarray(frame))
    if frame.ndim == 2: frame = frame[..., None]
    while frame.ndim > 3: frame = frame[0]
    if frame.ndim == 2: frame = frame[..., None]
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if np.issubdtype(frame.dtype, np.floating) or frame.max() <= 1.0:
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    else:
        frame = frame.clip(0, 255).astype(np.uint8)
    return frame

def normalize_video_frames(video_frames):
    if isinstance(video_frames, list):
        return [normalize_frame(fr) for fr in video_frames]

    raw_frames = to_numpy(video_frames)
    if not isinstance(raw_frames, np.ndarray):
        raw_frames = np.asarray(raw_frames)

    if raw_frames.ndim == 5: 
        raw_frames = raw_frames[0]
    elif raw_frames.ndim == 4 and raw_frames.shape[0] == 1: 
        raw_frames = raw_frames[0]
    
    out = []
    for i in range(raw_frames.shape[0]):
        try:
            out.append(normalize_frame(raw_frames[i]))
        except Exception as e:
            print(f"⚠️ تحذير في الإطار {i}: {e}")
    return out

def export_video_with_opencv(frames, output_path, fps=16):
    if not frames: raise ValueError("الإطارات فارغة.")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

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
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted, bad motion, static")
        
        image_input = job_input.get("image_reference") or job_input.get("image")
        pil_img = load_image_from_input(image_input)

        if pil_img is None:
            return {"status": "error", "message": "يلزم إرسال صورة (image) لتشغيل نموذج التحريك."}

        if not prompt or not prompt.strip():
            return {"status": "error", "message": "يلزم إرسال وصف الحركة (prompt) مع الصورة."}

        num_inference_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))
        fps = int(job_input.get("fps", 16))
        
        width = int(job_input.get("width", 832))   
        height = int(job_input.get("height", 480))

        pil_img = pil_img.resize((width, height))
        print(f"🎬 جاري تحريك الصورة باستخدام Wan2.1 لدقة {width}x{height}...")

        with pipeline_lock:
            output_result = i2v_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).frames

        torch.cuda.empty_cache()

        if output_result is None:
            return {"status": "error", "message": "لم يرجع النموذج أية إطارات."}

        output = output_result[0] if isinstance(output_result, (list, tuple)) else output_result
        video_frames = normalize_video_frames(output)

        if not video_frames:
            return {"status": "error", "message": "فشل تحويل إطارات الفيديو."}

        video_path = "/tmp/output_wan_i2v.mp4"
        export_video_with_opencv(video_frames, video_path, fps=fps)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {"status": "error", "message": "ملف الفيديو الناتج فارغ."}

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("✨ تم تحريك الصورة بنجاح!")
        return {"status": "success", "video_base64": video_base64}

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

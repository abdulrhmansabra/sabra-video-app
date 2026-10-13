import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback
import threading
import cv2

from diffusers import LTXPipeline, LTXImageToVideoPipeline
import runpod

print("جاري تحميل نماذج LTX-Video بكفاءة عالية (مشاركة الأوزان)...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

# قفل تزامني لتجنب تداخل الطلبات
pipeline_lock = threading.Lock()

# -------------------------
# تحميل النماذج عند الإقلاع (بدون استهلاك ضعف الذاكرة)
# -------------------------
try:
    MODEL_ID = "Lightricks/LTX-Video"

    # 1. تحميل النموذج الأساسي الأول (Text-to-Video)
    t2v_pipeline = LTXPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        **_token_kwargs()
    )
    t2v_pipeline.to("cuda")

    # 2. اشتقاق بايبلاين الصورة منه مباشرة (من نفس الذاكرة وبدون استهلاك VRAM إضافي)
    i2v_pipeline = LTXImageToVideoPipeline.from_pipe(t2v_pipeline)
    i2v_pipeline.to("cuda")
    
    print("✅ تم تحميل ونشر نماذج LTX بنجاح وبأعلى كفاءة للذاكرة!")
    MODEL_READY = True

except Exception as e:
    print("❌ فشل تحميل النماذج:")
    traceback.print_exc()
    MODEL_READY = False
    t2v_pipeline = None
    i2v_pipeline = None


# -------------------------
# دوال معالجة وتصدير الإطارات
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
        out = []
        for fr in video_frames:
            out.append(normalize_frame(fr))
        return out

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
            print(f"⚠️ تحذير: فشل تطبيع الإطار رقم {i}: {e}")
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
        print(f"خطأ في قراءة الصورة: الخروج بدون صورة. التفاصيل: {e}")
        return None


# -------------------------
# RunPod Handler الأساسي
# -------------------------
def handler(event):
    if not MODEL_READY or t2v_pipeline is None or i2v_pipeline is None:
        return {"status": "error", "message": "النماذج غير محملة أو فشل إقلاع السيرفر."}

    try:
        job_input = event.get("input") or {}
        
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt", "blurry, distorted face, low quality, static, bad anatomy")
        
        image_input = job_input.get("image_reference") or job_input.get("image")
        pil_img = load_image_from_input(image_input)

        if not prompt or not prompt.strip():
            return {"status": "error", "message": "يجب إرسال prompt على الأقل."}

        num_inference_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        num_frames = int(job_input.get("num_frames", 49)) 
        fps = int(job_input.get("fps", 16))
        width = int(job_input.get("width", 768))   
        height = int(job_input.get("height", 512))

        pipeline_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }

        # التوجيه الذكي التلقائي بناءً على إرسال المستخدم للصورة أم لا
        if pil_img is not None:
            pil_img = pil_img.resize((width, height))
            pipeline_kwargs["image"] = pil_img
            active_pipeline = i2v_pipeline
            print(f"🚀 بدء معالجة Image-to-Video: {width}x{height}")
        else:
            active_pipeline = t2v_pipeline
            print(f"🚀 بدء توليد Text-to-Video من الصفر: {width}x{height}")

        with pipeline_lock:
            output_result = active_pipeline(**pipeline_kwargs).frames

        torch.cuda.empty_cache()

        if output_result is None:
            return {"status": "error", "message": "لم يتم إرجاع إطارات من النموذج."}

        output = output_result[0] if isinstance(output_result, (list, tuple)) else output_result
        video_frames = normalize_video_frames(output)

        if not video_frames:
            return {"status": "error", "message": "فشل تطبيع إطارات الفيديو."}

        video_path = "/tmp/output.mp4"
        export_video_with_opencv(video_frames, video_path, fps=fps)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {"status": "error", "message": "ملف الفيديو الناتج فارغ."}

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("✨ تم توليد وتصدير الفيديو بنجاح!")
        return {"status": "success", "video_base64": video_base64}

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

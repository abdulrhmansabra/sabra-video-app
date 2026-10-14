import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback
import threading
import subprocess
import cv2

from diffusers import WanPipeline, WanImageToVideoPipeline
import runpod

print("🚀 99جاري تهيئة وتحميل نموذج Wan2.2-TI2V-5B...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

pipeline_lock = threading.Lock()
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# تحميل البايپلاين المزدوج (يدعم النص والصورة)
try:
    print("📦 جاري تحميل WanImageToVideoPipeline...")
    pipeline = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        **_token_kwargs()
    )
    pipeline.enable_model_cpu_offload()
    print("✅ تم تحميل النموذج وتفعيل حماية الذاكرة بنجاح!")
    MODEL_READY = True
except Exception as e:
    print("❌ فشل تحميل النموذج الأساسي:")
    traceback.print_exc()
    MODEL_READY = False
    pipeline = None


def to_numpy(x):
    if hasattr(x, "detach"): x = x.detach()
    if hasattr(x, "cpu"): x = x.cpu()
    if hasattr(x, "numpy"): x = x.numpy()
    return x

def enhance_and_normalize_frame(frame):
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
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        frame_bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )
    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

def process_video_frames(video_frames):
    if isinstance(video_frames, list):
        return [enhance_and_normalize_frame(fr) for fr in video_frames]
    raw_frames = to_numpy(video_frames)
    if not isinstance(raw_frames, np.ndarray):
        raw_frames = np.asarray(raw_frames)
    if raw_frames.ndim == 5: raw_frames = raw_frames[0]
    elif raw_frames.ndim == 4 and raw_frames.shape[0] == 1: raw_frames = raw_frames[0]
    
    out = []
    for i in range(raw_frames.shape[0]):
        try:
            out.append(enhance_and_normalize_frame(raw_frames[i]))
        except Exception as e:
            print(f"⚠️ تحذير في إطار {i}: {e}")
    return out

def export_video_with_ffmpeg(frames, output_path, fps=24):
    if not frames: raise ValueError("الإطارات فارغة.")
    height, width, _ = frames[0].shape
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', output_path
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        err = process.stderr.read().decode()
        raise RuntimeError(f"فشل تصدير FFmpeg: {err}")

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
            if "," in img_input: img_input = img_input.split(",", 1)[1]
            data = base64.b64decode(img_input)
            return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"خطأ في قراءة الصورة: {e}")
        return None

def handler(event):
    if not MODEL_READY:
        return {"status": "error", "message": "النموذج غير محمل."}

    try:
        job_input = event.get("input") or {}
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted")
        
        image_input = job_input.get("image_reference") or job_input.get("image")
        pil_img = load_image_from_input(image_input)

        if not prompt or not prompt.strip():
            return {"status": "error", "message": "يلزم إرسال الوصف (prompt)."}

        num_inference_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))
        fps = int(job_input.get("fps", 24))
        num_frames = int(job_input.get("num_frames", 25))
        width = int(job_input.get("width", 720))
        height = int(job_input.get("height", 480))

        if pil_img is not None:
            pil_img = pil_img.resize((width, height))

        print(f"🎬 بدء المعالجة عبر Wan2.2 ({width}x{height}, {num_frames} إطار)...")

        with pipeline_lock:
            # استدعاء صحيح بالوسيط image المخصص لكلاس WanImageToVideoPipeline
            output_result = pipeline(
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
            return {"status": "error", "message": "لم يُرجع النموذج إطارات."}

        output = output_result[0] if isinstance(output_result, (list, tuple)) else output_result
        video_frames = process_video_frames(output)

        if not video_frames:
            return {"status": "error", "message": "فشل معالجة الإطارات."}

        video_path = "/tmp/output_wan22.mp4"
        export_video_with_ffmpeg(video_frames, video_path, fps=fps)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {"status": "error", "message": "ملف الفيديو فارغ."}

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("✨ تمت عملية التوليد بنجاح تام!")
        return {"status": "success", "video_base64": video_base64}

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

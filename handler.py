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
import sys
import glob

import runpod

# إضافة مسار مستودع AnimateAnyone لبيئة العمل
sys.path.append('/app/Moore-AnimateAnyone')

print("🚀 جاري إعداد بيئة عمل Moore-AnimateAnyone على RunPod...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

pipeline_lock = threading.Lock()

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

def load_file_from_input(input_data: str, output_path: str):
    """دالة عامة لتحميل الصور أو الفيديوهات من رابط أو Base64 وحفظها محلياً"""
    if not input_data: return False
    try:
        input_str = str(input_data).strip()
        if input_str.startswith("http://") or input_str.startswith("https://"):
            response = requests.get(input_str, timeout=60)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            if "," in input_str: input_str = input_str.split(",", 1)[1]
            data = base64.b64decode(input_str)
            with open(output_path, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"خطأ في قراءة وتحميل الملف إلى {output_path}: {e}")
        return False

def handler(event):
    try:
        job_input = event.get("input") or {}
        
        # مدخلات AnimateAnyone (صورة الشخصية + فيديو الحركة المرجعي)
        character_image_input = job_input.get("character_image") or job_input.get("image")
        pose_video_input = job_input.get("pose_video") or job_input.get("video")

        if not character_image_input or not pose_video_input:
            return {
                "status": "error", 
                "message": "يلزم إرسال صورة الشخصية (character_image) وفيديو الحركة المرجعي (pose_video)."
            }

        local_image_path = "/tmp/character_image.png"
        local_video_path = "/tmp/pose_video.mp4"
        output_video_path = "/tmp/output_animation.mp4"

        print("📥 جاري تجهيز وتحميل المدخلات...")
        if not load_file_from_input(character_image_input, local_image_path):
            return {"status": "error", "message": "فشل تحميل صورة الشخصية."}
        
        if not load_file_from_input(pose_video_input, local_video_path):
            return {"status": "error", "message": "فشل تحميل الفيديو المرجعي."}

        print("🎬 جاري تشغيل نموذج AnimateAnyone لنقل الحركة...")

        with pipeline_lock:
            # استدعاء سكريبت الاستدلال الخاص بالمستودع عبر الـ Subprocess ضمن قفل الحماية
            inference_cmd = [
                sys.executable, "/app/Moore-AnimateAnyone/inference.py",
                "--source_image", local_image_path,
                "--driving_video", local_video_path,
                "--output_dir", "/tmp"
            ]
            
            result = subprocess.run(inference_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ خطأ أثناء تنفيذ النمذجة: {result.stderr}")
                return {"status": "error", "message": f"فشل عملية المعالجة: {result.stderr[-500:]}"}

        torch.cuda.empty_cache()

        # البحث عن الفيديو الناتج والتحقق منه
        if not os.path.exists(output_video_path):
            output_files = glob.glob("/tmp/*.mp4")
            output_files = [f for f in output_files if f != local_video_path]
            if output_files:
                output_video_path = max(output_files, key=os.path.getmtime)
            else:
                return {"status": "error", "message": "لم يُرجع النموذج أي ملف فيديو."}

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            return {"status": "error", "message": "ملف الفيديو الناتج فارغ أو غير موجود."}

        # قراءة الفيديو الناتج وتحويله إلى Base64 للإرجاع بنفس الهيكل السابق
        with open(output_video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("✨ تمت عملية التوليد ونقل الحركة بنجاح تام!")
        return {"status": "success", "video_base64": video_base64}

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

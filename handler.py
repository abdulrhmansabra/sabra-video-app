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

from diffusers import LTXPipeline
import runpod

print("جاري تحميل وتهيئة نموذج LTX-Video للصور إلى فيديو...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

# قفل تزامني لحماية الحالة العالمية للـ Pipeline من التداخل
pipeline_lock = threading.Lock()

# -------------------------
# Model load at startup
# -------------------------
try:
    # المستودع الرسمي المعتمد لنموذج LTX-Video
    MODEL_ID = "Lightricks/LTX-Video"

    # تحميل البايبلاين الرسمي الموحد الداعم للنص والصور معاً
    pipeline = LTXPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        **_token_kwargs()
    )
    
    pipeline.to("cuda")

    print("نموذج LTX-Video جاهز بالكامل للاستجابة!")
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

    # ضبط النطاق اللوني الحقيقي للموديل [0, 1] وتحويله إلى uint8
    if np.issubdtype(frame.dtype, np.floating) or frame.max() <= 1.0:
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    else:
        frame = frame.clip(0, 255).astype(np.uint8)

    return frame

def normalize_video_frames(video_frames):
    raw_frames = to_numpy(video_frames)
    
    if raw_frames.ndim == 5:
        raw_frames = raw_frames[0]  # إزالة الـ Batch
    elif raw_frames.ndim == 4 and raw_frames.shape[0] == 1:
        raw_frames = raw_frames[0]

    out = []
    for i in range(raw_frames.shape[0]):
        fr = raw_frames[i]
        try:
            out.append(normalize_frame(fr))
        except Exception as e:
            print(f"⚠️ تحذير: فشل تطبيع الإطار رقم {i}: {e}")
    return out

def export_video_with_opencv(frames, output_path, fps=15):
    if not frames:
        raise ValueError("قائمة الإطارات فارغة، لا يمكن التصدير.")
    
    height, width, _ = frames[0].shape
    # استخدام ترميز mp4v لضمان توافق الفيديو وعرضه بسلاسة داخل المتصفح و RunPod
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # عكس القنوات لتصحيح الألوان نهائياً وتجنب مشكلة الكاميرا الحرارية
        corrected_frame = frame[..., ::-1]
        out.write(corrected_frame)
        
    out.release()


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

        # التقاط رابط الصورة من حقل image_reference المرسل في الـ JSON
        image_input = job_input.get("image_reference") or job_input.get("image")
        pil_img = load_image_from_input(image_input)
        has_image = pil_img is not None

        if not has_image:
            return {"status": "error", "message": "يجب إرفاق رابط صورة صالح في حقل image_reference لتثبيت الملامح."}

        if prompt is None or (isinstance(prompt, str) and prompt.strip() == ""):
            prompt = "cinematic close-up shot maintaining exact facial features"

        if negative_prompt is None:
            negative_prompt = (
                "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
                "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
                "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
                "many people in the background, walking backwards"
            )

        num_inference_steps = int(job_input.get("num_inference_steps", 25))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        num_frames = int(job_input.get("num_frames", 49))
        fps = int(job_input.get("fps", 16))
        
        width = int(job_input.get("width", 640))
        height = int(job_input.get("height", 352))

        print(f"🚀 بدء معالجة الطلب: prompt='{prompt}', steps={num_inference_steps}, frames={num_frames}, size=({width}x{height})")

        with pipeline_lock:
            # تمرير الصورة للبايبلاين عبر معامل image لتثبيت ملامح الوجه والشخصية
            output_result = pipeline(
                image=pil_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                height=height,
                width=width,
            ).frames

        torch.cuda.empty_cache()

        if output_result is None:
            return {"status": "error", "message": "فشل التوليد: لم يتم إرجاع إطارات من الـ pipeline."}

        if isinstance(output_result, (list, tuple)):
            output = output_result[0]
        else:
            output = output_result

        print(f"🔍 تشخيص الإطارات الخام -> type: {type(output)}, shape/len: {getattr(output, 'shape', len(output) if hasattr(output, '__len__') else 'unknown')}")

        video_frames = normalize_video_frames(output)

        if video_frames is None or len(video_frames) == 0:
            return {"status": "error", "message": "قائمة الإطارات فارغة بعد التطبيع."}

        video_path = "/tmp/output.mp4"
        export_video_with_opencv(video_frames, video_path, fps=fps)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {"status": "error", "message": "فشل تصدير الفيديو: الملف الناتج فارغ أو غير موجود."}

        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("تم توليد وتصدير الفيديو بنجاح بمعايير الألوان الدقيقة والدقة العالية!")
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

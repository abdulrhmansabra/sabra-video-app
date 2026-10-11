import os
import torch
from PIL import Image
import requests
from io import BytesIO
import base64

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
import runpod

print("جاري تحميل وتهيئة النماذج بواسطة Diffusers...")

# اقرأ توكن Hugging Face من البيئة
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    """
    بعض الإصدارات تستخدم token=
    وبعضها تستخدم use_auth_token=
    نجهز kwargs بالطريقتين ونستخدم اللي يشتغل.
    """
    if not HF_TOKEN:
        return {}
    return {"token": HF_TOKEN, "use_auth_token": HF_TOKEN}

# تحميل النماذج عند الإقلاع (يمكن نقلها داخل handler إذا تبغى lazy load)
try:
    adapter = MotionAdapter.from_pretrained(
        "ByteDance/AnimateDiff-Lightning",
        subfolder="animatediff_motion_adapter_4step",
        torch_dtype=torch.float16,
        **_token_kwargs()
    )

    pipeline = AnimateDiffPipeline.from_pretrained(
        "emilianJR/Realistic_Vision_v5.0_noVAE",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
        **_token_kwargs()
    )

    ip_adapter_loaded = False
    try:
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            **_token_kwargs()
        )
        pipeline.set_ip_adapter_scale(0.6)
        ip_adapter_loaded = True
        print("تم تفعيل دعم الصور (IP-Adapter) بنجاح!")
    except Exception as e:
        print(f"تحذير أثناء تحميل IP-Adapter (سيتم تعطيله): {e}")

    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing"
    )
    pipeline.to("cuda")
    print("النموذج جاهز بالكامل للاستجابة!")

    MODEL_READY = True

except Exception as e:
    # لا تخلي العامل ينهار بصمت ويعيد التشغيل بدون سبب مفهوم
    print(f"فشل تحميل النماذج أثناء الإقلاع: {e}")
    MODEL_READY = False
    ip_adapter_loaded = False
    pipeline = None


def load_image_from_input(img_input):
    if not img_input:
        return None
    try:
        if img_input.startswith("http://") or img_input.startswith("https://"):
            response = requests.get(img_input, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if "," in img_input:
                img_input = img_input.split(",")[1]
            data = base64.b64decode(img_input)
            return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"خطأ في قراءة الصورة: {e}")
        return None


def handler(event):
    if not MODEL_READY or pipeline is None:
        # ارجع خطأ واضح بدل ما يطيح العامل ويصير restart loop
        return {
            "status": "error",
            "message": "النماذج لم تُحمّل بنجاح عند الإقلاع. راجع Logs لمعرفة سبب الفشل (غالباً توكن HF أو صلاحيات الموديل أو إصدارات المكتبات)."
        }

    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "A cinematic stunning shot of a futuristic neon city, high quality")
        negative_prompt = job_input.get("negative_prompt", "bad quality, blurry, low resolution")
        image_input = job_input.get("image", None)

        print(f"استلام طلب توليد. Prompt: {prompt} | Image-to-Video: {image_input is not None}")

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "num_frames": 16,
        }

        if image_input and ip_adapter_loaded:
            pil_img = load_image_from_input(image_input)
            if pil_img:
                kwargs["ip_adapter_image"] = pil_img

        output = pipeline(**kwargs)
        frames = output.frames[0]

        video_path = "/tmp/output.mp4"
        export_to_video(frames, video_path, fps=8)

        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        return {
            "status": "success",
            "message": "تم توليد الفيديو بنجاح",
            "video_base64": video_base64
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})

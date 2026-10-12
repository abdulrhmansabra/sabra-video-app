import os
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback
import threading

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
import runpod

print("جاري تحميل وتهيئة النماذج بواسطة Diffusers...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

# قفل تزامني لحماية الحالة العالمية للـ Pipeline من التداخل
pipeline_lock = threading.Lock()

# -------------------------
# Model load at startup (متغيرات والتهيئة عند الإقلاع)
# -------------------------
try:
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
        **_token_kwargs()
    )

    weights_path = hf_hub_download(
        repo_id="ByteDance/AnimateDiff-Lightning",
        filename="animatediff_lightning_4step_diffusers.safetensors",
        **_token_kwargs()
    )
    
    state_dict = load_file(weights_path)
    adapter.load_state_dict(state_dict, strict=False)
    print("Loaded Lightning weights successfully.")

    pipeline = AnimateDiffPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.0_noVAE",
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
        ip_adapter_loaded = False

    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing"
    )
    pipeline.to("cuda")

    print("النموذج جاهز بالكامل للاستجابة!")
    MODEL_READY = True

except Exception as e:
    print(f"فشل تحميل النماذج أثناء الإقلاع: {e}")
    MODEL_READY = False
    ip_adapter_loaded = False
    pipeline = None


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
            negative_prompt = "bad quality, blurry, low resolution"

        num_inference_steps = int(job_input.get("num_inference_steps", 4))
        guidance_scale = float(job_input.get("guidance_scale", 1.0))
        num_frames = int(job_input.get("num_frames", 16))
        fps = int(job_input.get("fps", 8))

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
        }

        if has_image:
            if not ip_adapter_loaded:
                return {"status": "error", "message": "تم إرسال image لكن IP-Adapter غير محمّل."}
            kwargs["ip_adapter_image"] = pil_img

        with pipeline_lock:
            output = pipeline(**kwargs)

        torch.cuda.empty_cache()

        frames = None
        if hasattr(output, "frames"):
            frames = output.frames
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            frames = output[0]

        if frames is None:
            return {"status": "error", "message": "فشل التوليد: لا توجد frames في مخرجات الـ pipeline."}

        video_frames = frames[0] if isinstance(frames, list) and len(frames) > 0 else frames
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

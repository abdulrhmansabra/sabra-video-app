import os
import torch
from PIL import Image
import requests
from io import BytesIO
import base64

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
import runpod

print("جاري تحميل وتهيئة النماذج بواسطة Diffusers...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    if not HF_TOKEN:
        return {}
    return {"token": HF_TOKEN}

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
        return {
            "status": "error",
            "message": "النماذج لم تُحمّل بنجاح عند الإقلاع. راجع Logs لمعرفة سبب الفشل."
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
        print(f"Pipeline output object: {output}")

        # استخراج الإطارات بأمان تام مع دعم مختلف هياكل الإخراج
        frames = None
        if hasattr(output, "frames"):
            frames = output.frames
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            frames = output[0]

        if frames is None:
            return {"status": "error", "message": "فشل التوليد: output.frames قيمتها None."}

        # إذا كانت الإطارات عبارة عن دفعة (Batch) نأخذ العنصر الأول
        if isinstance(frames, list) and len(frames) > 0:
            video_frames = frames[0]
        else:
            video_frames = frames

        if not video_frames:
            return {"status": "error", "message": "قائمة الإطارات فارغة."}

        video_path = "/tmp/output.mp4"
        export_to_video(video_frames, video_path, fps=8)

        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        return {
            "status": "success",
            "message": "تم توليد الفيديو بنجاح",
            "video_base64": video_base64
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})

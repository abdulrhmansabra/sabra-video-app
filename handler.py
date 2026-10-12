import os
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
import runpod

print("جاري تحميل وتهيئة النماذج بواسطة Diffusers...")

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _token_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}

# -------------------------
# Model load at startup
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
    """
    Accepts:
      - URL string (http/https)
      - base64 string (optionally data URL: data:image/png;base64,...)
    Returns:
      PIL.Image or None
    """
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
            # handle data URLs and raw base64
            if "," in img_input:
                img_input = img_input.split(",", 1)[1]
            data

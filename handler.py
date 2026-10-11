import os
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
import runpod

print("جاري التحقق من النماذج أو تحميلها على سحابة RunPod...")

ADAPTER_DIR = "/tmp/models/animatediff_adapter"
MODEL_DIR = "/tmp/models/realistic_vision"

os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# التحميل السريع الفوري إذا لم تكن موجودة
if not os.listdir(ADAPTER_DIR):
    print("جاري تحميل محرك الحركة...")
    snapshot_download(
        repo_id="ByteDance/AnimateDiff-Lightning",
        subfolder="animatediff_motion_adapter_4step",
        local_dir=ADAPTER_DIR,
        max_workers=2
    )

if not os.listdir(MODEL_DIR):
    print("جاري تحميل النموذج الأساسي...")
    snapshot_download(
        repo_id="emilianJR/Realistic_Vision_v5.0_noVAE",
        local_dir=MODEL_DIR,
        max_workers=2
    )

print("جاري تهيئة النماذج في ذاكرة كرت الشاشة...")

adapter = MotionAdapter.from_pretrained(
    ADAPTER_DIR, 
    torch_dtype=torch.float16
)

pipeline = AnimateDiffPipeline.from_pretrained(
    MODEL_DIR, 
    motion_adapter=adapter, 
    torch_dtype=torch.float16
)

try:
    pipeline.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="models", 
        weight_name="ip-adapter_sd15.bin"
    )
    pipeline.set_ip_adapter_scale(0.6)
    print("تم تفعيل دعم الصور بنجاح!")
except Exception as e:
    print(f"تحذير أثناء تحميل IP-Adapter: {e}")

pipeline.scheduler = EulerDiscreteScheduler.from_config(
    pipeline.scheduler.config, 
    timestep_spacing="trailing"
)
pipeline.to("cuda")
print("النموذج جاهز بالكامل للاستجابة!")

def load_image_from_input(img_input):
    if not img_input:
        return None
    try:
        if img_input.startswith("http://") or img_input.startswith("https://"):
            response = requests.get(img_input)
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
        
        if image_input:
            pil_img = load_image_from_input(image_input)
            if pil_img:
                kwargs["ip_adapter_image"] = pil_img
        
        output = pipeline(**kwargs)
        frames = output.frames[0]
        
        video_path = "/tmp/output.mp4"
        export_to_video(frames, video_path, fps=8)
        
        return {
            "status": "success",
            "message": "تم توليد الفيديو بنجاح",
            "video_path": video_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

runpod.serverless.start({"handler": handler})

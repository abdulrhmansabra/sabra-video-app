import os
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
import runpod

print("جاري تهيئة نموذج AnimateDiff و IP-Adapter لدعم الصور والنصوص...")

# تحميل المحرك السريع ونموذج الرسم الأساسي
adapter = MotionAdapter.from_pretrained(
    "ByteDance/AnimateDiff-Lightning", 
    subfolder="animatediff_motion_adapter_4step", 
    torch_dtype=torch.float16
)
model_id = "emilianJR/Realistic_Vision_v5.0_noVAE"

pipeline = AnimateDiffPipeline.from_pretrained(
    model_id, 
    motion_adapter=adapter, 
    torch_dtype=torch.float16
)

# تحميل الـ IP-Adapter لدعم تحويل الصور إلى فيديو
try:
    pipeline.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="models", 
        weight_name="ip-adapter_sd15.bin"
    )
    pipeline.set_ip_adapter_scale(0.6)
    print("تم تفعيل دعم الصور (IP-Adapter) بنجاح!")
except Exception as e:
    print(f"تحذير أثناء تحميل IP-Adapter: {e}")

pipeline.scheduler = EulerDiscreteScheduler.from_config(
    pipeline.scheduler.config, 
    timestep_spacing="trailing"
)
pipeline.to("cuda")

def load_image_from_input(img_input):
    if not img_input:
        return None
    try:
        if img_input.startswith("http://") or img_input.startswith("https://"):
            response = requests.get(img_input)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # معالجة بيانات Base64 إذا تم إرسالها مباشرة
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
        image_input = job_input.get("image", None) # يمكن أن يكون رابط URL أو Base64 للصورة
        
        print(f"استلام طلب توليد. Prompt: {prompt} | Image-to-Video: {image_input is not None}")
        
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "num_frames": 16,
        }
        
        # إذا وُجدت صورة، يتم إضافتها كمرجع بصري للنموذج
        if image_input:
            pil_img = load_image_from_input(image_input)
            if pil_img:
                kwargs["ip_adapter_image"] = pil_img
                print("تم تضمين الصورة بنجاح لتوجيه حركة الفيديو.")
        
        # توليد الفيديو
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

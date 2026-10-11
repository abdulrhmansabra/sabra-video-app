import os
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
import runpod

print("جاري تهيئة نموذج AnimateDiff-Lightning...")

# تحميل محرك الحركة السريع والنموذج الأساسي
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
pipeline.scheduler = EulerDiscreteScheduler.from_config(
    pipeline.scheduler.config, 
    timestep_spacing="trailing"
)
pipeline.to("cuda")

def handler(event):
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "A cinematic stunning shot of a futuristic neon city, high quality")
        negative_prompt = job_input.get("negative_prompt", "bad quality, blurry, low resolution")
        
        print(f"استلام طلب توليد بـ Prompt: {prompt}")
        
        # توليد الإطارات
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            num_frames=16,
        )
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

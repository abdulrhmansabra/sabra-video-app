import os
import torch
import runpod
from diffusers import LTXVideoPipeline
from diffusers.utils import export_to_video

# تحديد جهاز التشغيل (كرت الشاشة GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading LTX-Video model on {device}...")

# مسار النموذج المضمن مسبقاً داخل حاوية الـ Docker
model_path = "/app/models/ltx-video"

# تحميل النموذج مرة واحدة عند إقلاع السيرفر (Cold Start) ليكون جاهزاً فوراً
pipe = LTXVideoPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float16
).to(device)

def handler(event):
    """
    الدالة الأساسية التي تستقبل الطلبات من RunPod Serverless وتولد الفيديو عبر LTX
    """
    job_input = event.get("input", {})
    prompt = job_input.get("prompt", "Cinematic shot of a futuristic city")
    
    print(f"Generating video for prompt: {prompt}")
    
    # توليد الإطارات باستخدام نموذج LTX-Video
    with torch.autocast("cuda"):
        video_frames = pipe(
            prompt=prompt,
            num_frames=49,       # عدد الإطارات الافتراضي
            height=512,
            width=768,
            num_inference_steps=30,
        ).frames[0]
    
    # حفظ الفيديو الناتج بصيغة mp4 في مسار مؤقت
    output_path = "/tmp/output.mp4"
    export_to_video(video_frames, output_path, fps=24)
    
    return {
        "status": "success",
        "message": f"Video successfully generated for prompt: '{prompt}'",
        "video_path": output_path
    }

# تشغيل خادم RunPod Serverless للاستماع للطلبات
runpod.serverless.start({"handler": handler})

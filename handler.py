import os
import torch
import runpod
from diffusers import LTXVideoPipeline
from diffusers.utils import export_to_video

# تحديد جهاز التشغيل (كرت الشاشة GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading LTX-Video model on {device}...")

# مسار النموذج المضمن داخل الحاوية
model_path = "/app/models/ltx-video"

# تحميل نموذج LTX-Video مرة واحدة أثناء إقلاع السيرفر (Cold Start)
pipe = LTXVideoPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float16
).to(device)

def handler(event):
    """
    الدالة التي تستقبل الطلبات من RunPod وتولد الفيديو عبر نموذج LTX
    """
    job_input = event.get("input", {})
    prompt = job_input.get("prompt", "Cinematic shot of a futuristic city")
    
    print(f"Generating video for prompt: {prompt}")
    
    # توليد الفيديو باستخدام النموذج
    with torch.autocast("cuda"):
        video_frames = pipe(
            prompt=prompt,
            num_frames=49,       # عدد الإطارات الافتراضي
            height=512,
            width=768,
            num_inference_steps=30,
        ).frames[0]
    
    # حفظ الفيديو الناتج مؤقتاً
    output_path = "/tmp/output.mp4"
    export_to_video(video_frames, output_path, fps=24)
    
    # [ملاحظة: يمكنك لاحقاً رفع ملف الفيديو إلى أي تخزين سحابي مجاني أو إعادته كـ Base64]
    
    return {
        "status": "success",
        "message": f"Video successfully generated for prompt: '{prompt}'",
        "video_path": output_path
    }

# تشغيل خادم RunPod Serverless
runpod.serverless.start({"handler": handler})

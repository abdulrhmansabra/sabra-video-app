import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import requests
from io import BytesIO
import base64
import traceback
import sys
import subprocess
import glob
import runpod

# إضافة مسار مستودع Moore-AnimateAnyone للاستيراد
sys.path.append('/app/Moore-AnimateAnyone')

print("🚀 جاري إعداد بيئة عمل Moore-AnimateAnyone على RunPod...")

def download_file_from_input(input_data, output_path):
    """تحميل الملف سواء كان رابط URL أو Base64 وحفظه محلياً"""
    if not input_data:
        return False
    try:
        input_str = str(input_data).strip()
        if input_str.startswith("http://") or input_str.startswith("https://"):
            response = requests.get(input_str, timeout=60)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            if "," in input_str:
                input_str = input_str.split(",", 1)[1]
            data = base64.b64decode(input_str)
            with open(output_path, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"⚠️ خطأ في تحميل الملف إلى {output_path}: {e}")
        return False

def handler(event):
    try:
        job_input = event.get("input") or {}
        
        # استقبال مدخلات AnimateAnyone (صورة الشخصية وفيديو الحركة المرجعي)
        character_image_input = job_input.get("character_image") or job_input.get("image")
        pose_video_input = job_input.get("pose_video") or job_input.get("video")

        if not character_image_input or not pose_video_input:
            return {
                "status": "error", 
                "message": "يجب إرسال صورة الشخصية (character_image) وفيديو الحركة المرجعي (pose_video)."
            }

        # مسارات مؤقتة للملفات
        local_image_path = "/tmp/character_image.png"
        local_video_path = "/tmp/pose_video.mp4"
        output_video_path = "/tmp/output_animation.mp4"

        print("📥 جاري تجهيز وتحميل الملفات المدخلة...")
        if not download_file_from_input(character_image_input, local_image_path):
            return {"status": "error", "message": "فشل تحميل صورة الشخصية."}
        
        if not download_file_from_input(pose_video_input, local_video_path):
            return {"status": "error", "message": "فشل تحميل الفيديو المرجعي."}

        print("🎬 جاري تنفيذ نموذج AnimateAnyone (Character & Pose Transfer)...")
        
        # استدعاء أمر التشغيل الخاص بالمستودع لنقل الحركة
        inference_cmd = [
            sys.executable, "/app/Moore-AnimateAnyone/inference.py",
            "--source_image", local_image_path,
            "--driving_video", local_video_path,
            "--output_dir", "/tmp"
        ]
        
        result = subprocess.run(inference_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ خطأ أثناء تنفيذ النمذجة: {result.stderr}")
            return {"status": "error", "message": f"فشل عملية المعالجة: {result.stderr[-500:]}"}

        # البحث عن الفيديو الناتج في مجلد الإخراج المؤقت
        if not os.path.exists(output_video_path):
            output_files = glob.glob("/tmp/*.mp4")
            output_files = [f for f in output_files if f != local_video_path]
            if output_files:
                output_video_path = max(output_files, key=os.path.getmtime)
            else:
                return {"status": "error", "message": "لم يتم العثور على ملف الفيديو الناتج."}

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            return {"status": "error", "message": "ملف الفيديو الناتج فارغ أو غير موجود."}

        print("📦 جاري قراءة وتحويل الفيديو الناتج إلى Base64...")
        with open(output_video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        # تنظيف ذاكرة الـ GPU
        torch.cuda.empty_cache()

        print("✨ تمت عملية تحريك الشخصية ونقل الحركة بنجاح تام!")
        return {
            "status": "success", 
            "video_base64": video_base64
        }

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

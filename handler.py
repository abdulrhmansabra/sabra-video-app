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
        return {"status": "error", "message": str(e)}

"""RunPod Serverless handler for Moore-AnimateAnyone full-body pose transfer.

Input: reference_image (URL or base64 data URI), and either driving_video
(raw video, DWPose is run automatically) or pose_video (already pose-rendered).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import base64
import gc
import subprocess
import threading
import traceback
from pathlib import Path
from io import BytesIO

import requests
import torch
from PIL import Image
import runpod

ROOT = Path(os.getenv("ANIMATEANYONE_ROOT", "/app/Moore-AnimateAnyone"))
WEIGHTS = ROOT / "pretrained_weights"
CONFIG = ROOT / "configs/prompts/animation.yaml"
PIPELINE_LOCK = threading.Lock()
_controller = None


def _download_input(value, dest: Path, timeout=120):
    """Write a URL/base64 string to a file. Also accepts raw base64 or data URI."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid input for {dest.name}")
    value = value.strip()
    if value.startswith(("http://", "https://")):
        with requests.get(value, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        out.write(chunk)
    else:
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            dest.write_bytes(base64.b64decode(encoded, validate=True))
        except Exception as exc:
            raise ValueError(f"{dest.name} must be a URL or base64/data URI") from exc
    if not dest.exists() or dest.stat().st_size == 0:
        raise ValueError(f"Downloaded input is empty: {dest.name}")
    return dest


def _ensure_weights():
    required = [
        WEIGHTS / "denoising_unet.pth",
        WEIGHTS / "reference_unet.pth",
        WEIGHTS / "pose_guider.pth",
        WEIGHTS / "motion_module.pth",
        WEIGHTS / "DWPose/dw-ll_ucoco_384.onnx",
        WEIGHTS / "DWPose/yolox_l.onnx",
        WEIGHTS / "image_encoder/config.json",
        WEIGHTS / "sd-vae-ft-mse/config.json",
        WEIGHTS / "stable-diffusion-v1-5/unet/config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if not missing:
        return
    if os.getenv("AUTO_DOWNLOAD_WEIGHTS", "1").lower() not in ("1", "true", "yes"):
        raise RuntimeError("AnimateAnyone weights are missing. Missing: " + ", ".join(missing))
    downloader = ROOT / "tools/download_weights.py"
    if not downloader.exists():
        raise RuntimeError(f"AnimateAnyone repository not found at {ROOT}; check Dockerfile build.")
    print("📦 Downloading missing AnimateAnyone model weights (first invocation may take time)...")
    result = subprocess.run(["python", str(downloader)], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Weight download failed. Check Hugging Face access/network.\n" + result.stdout[-4000:] + "\n" + result.stderr[-4000:])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Weight download completed but files remain missing: " + ", ".join(missing))


def _get_controller():
    global _controller
    if _controller is not None:
        return _controller
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Attach this RunPod endpoint to a CUDA GPU worker.")
    if not CONFIG.exists():
        raise RuntimeError(f"AnimateAnyone config not found: {CONFIG}")
    # Upstream config uses paths relative to the repository root.
    os.chdir(ROOT)
    _ensure_weights()

    # Import model modules from the upstream repository, without importing app.py
    # (which launches a Gradio server as a side effect).
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.models.pose_guider import PoseGuider
    from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline

    config = OmegaConf.load(str(CONFIG))
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to("cuda", dtype=dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet"
    ).to(device="cuda", dtype=dtype)
    inference_cfg = OmegaConf.load(config.inference_config)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=inference_cfg.unet_additional_kwargs,
    ).to(device="cuda", dtype=dtype)
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(device="cuda", dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to("cuda", dtype=dtype)

    denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))
    pose_guider.load_state_dict(torch.load(config.pose_guider_path, map_location="cpu"))
    scheduler = DDIMScheduler(**OmegaConf.to_container(inference_cfg.noise_scheduler_kwargs))
    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    ).to("cuda", dtype=dtype)
    _controller = pipe
    print("✅ AnimateAnyone pipeline loaded")
    return _controller


def _encode_file(path: Path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def handler(event):
    try:
        data = (event or {}).get("input") or {}
        image_value = data.get("reference_image") or data.get("image_reference") or data.get("image")
        driving_value = data.get("driving_video") or data.get("video_reference")
        pose_value = data.get("pose_video")
        if not image_value:
            return {"status": "error", "message": "أرسل reference_image: صورة الشخصية المستهدفة (URL أو base64)."}
        if not driving_value and not pose_value:
            return {"status": "error", "message": "أرسل driving_video (فيديو الحركة الأصلي) أو pose_video (فيديو الوضعيات الجاهز)."}

        width = int(data.get("width", 512))
        height = int(data.get("height", 768))
        length = int(data.get("num_frames", data.get("length", 24)))
        steps = int(data.get("num_inference_steps", 25))
        guidance = float(data.get("guidance_scale", 3.5))
        fps_override = int(data.get("fps", 0))
        seed = int(data.get("seed", 123))
        if width < 64 or height < 64 or length < 1 or length > 128:
            raise ValueError("width/height must be >=64 and num_frames must be between 1 and 128")

        work = Path("/tmp") / f"animateanyone_{os.getpid()}"
        work.mkdir(parents=True, exist_ok=True)
        image_path = _download_input(image_value, work / "reference_image.bin")
        with Image.open(image_path) as im:
            reference_image = im.convert("RGB")
        video_path = _download_input(pose_value or driving_value, work / "input_video.mp4")

        with PIPELINE_LOCK:
            import sys
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from src.utils.util import get_fps, save_videos_grid
            if driving_value and not pose_value:
                # DWPose produces <input-stem>_kps.mp4 next to the input.
                pose_path = work / "input_video_kps.mp4"
                proc = subprocess.run(
                    ["python", str(ROOT / "tools/vid2pose.py"), "--video_path", str(video_path)],
                    cwd=ROOT, text=True, capture_output=True,
                )
                if proc.returncode != 0 or not pose_path.exists():
                    raise RuntimeError("DWPose extraction failed.\n" + proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
            else:
                pose_path = video_path

            pipe = _get_controller()
            src_fps = fps_override if fps_override > 0 else int(get_fps(str(video_path)))
            from src.utils.util import read_frames
            pose_frames = read_frames(str(pose_path))
            if not pose_frames:
                raise RuntimeError("No frames could be read from the pose video.")
            length = min(length, len(pose_frames))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            result = pipe(
                reference_image,
                pose_frames[:length],
                width=width,
                height=height,
                video_length=length,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).videos
            output_path = work / "output_animateanyone.mp4"
            save_videos_grid(result, str(output_path), n_rows=1, fps=src_fps)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("AnimateAnyone did not produce a valid MP4 file.")
        response = {
            "status": "success",
            "video_base64": _encode_file(output_path),
            "content_type": "video/mp4",
            "num_frames": length,
            "fps": src_fps,
            "width": width,
            "height": height,
        }
        return response
    except Exception as exc:
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "error", "message": str(exc)}
    finally:
        gc.collect()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

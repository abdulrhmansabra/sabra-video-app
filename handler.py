"""RunPod Serverless handler for Moore-AnimateAnyone full-body pose transfer.

Inputs:
  reference_image: image URL, base64, or data URI
  driving_video: raw motion video URL/base64 (DWPose is run automatically)
  pose_video: already pose-rendered video URL/base64 (alternative to driving_video)

The official repository and weights are downloaded into temporary storage (/tmp) lazily on the first valid job.
Temporary files may be removed when the RunPod worker is recycled.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import base64
import gc
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import requests
import torch
from PIL import Image
import runpod

REPO_URL = "https://github.com/MooreThreads/Moore-AnimateAnyone.git"
ROOT = Path(os.getenv("ANIMATEANYONE_ROOT", "/tmp/Moore-AnimateAnyone")).resolve()
WEIGHTS = ROOT / "pretrained_weights"
CONFIG = ROOT / "configs/prompts/animation.yaml"
PIPELINE_LOCK = threading.Lock()
_controller = None


def _run(command, *, cwd=None, timeout=None):
    """Run a command and include useful output if it fails."""
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str, command))}\n"
            f"STDOUT:\n{result.stdout[-5000:]}\nSTDERR:\n{result.stderr[-5000:]}"
        )
    return result


def _ensure_repository():
    """Ensure the official source checkout exists before importing src.*."""
    source_file = ROOT / "src" / "utils" / "util.py"
    downloader = ROOT / "tools" / "download_weights.py"
    if source_file.is_file() and downloader.is_file() and CONFIG.is_file():
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        return

    # Never overwrite an existing non-empty directory: that could destroy weights.
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError(
            f"Found an incomplete repository at {ROOT}. Expected {source_file}, "
            f"{downloader}, and {CONFIG}. Correct ANIMATEANYONE_ROOT or use an image "
            "containing the official repository; refusing to overwrite existing files."
        )

    if not shutil_which("git"):
        raise RuntimeError("Official repository is missing and git is not installed in the image.")
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning official Moore-AnimateAnyone repository from {REPO_URL} into {ROOT}")
    _run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], timeout=600)

    missing = [str(p) for p in (source_file, downloader, CONFIG) if not p.is_file()]
    if missing:
        raise RuntimeError("Official repository checkout is incomplete. Missing: " + ", ".join(missing))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


def shutil_which(command):
    """Small PATH lookup helper; avoids an extra import at module initialization."""
    from shutil import which
    return which(command)


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
    """Download official weights only if required files are absent."""
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
    missing = [str(path) for path in required if not path.exists()]
    if not missing:
        return
    if os.getenv("AUTO_DOWNLOAD_WEIGHTS", "1").lower() not in ("1", "true", "yes"):
        raise RuntimeError("Model files are missing and AUTO_DOWNLOAD_WEIGHTS is disabled: " + ", ".join(missing))

    downloader = ROOT / "tools" / "download_weights.py"
    if not downloader.is_file():
        raise RuntimeError(f"Official weight downloader is missing: {downloader}")
    print("Downloading missing Moore-AnimateAnyone weights; first invocation may take several minutes...")
    _run([sys.executable, str(downloader)], cwd=ROOT, timeout=3600)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Weight download finished but required files are still missing: " + ", ".join(missing))


def _get_controller():
    global _controller
    if _controller is not None:
        return _controller
    _ensure_repository()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Attach this RunPod endpoint to a CUDA GPU worker.")
    if not CONFIG.is_file():
        raise RuntimeError(f"AnimateAnyone config not found: {CONFIG}")
    _ensure_weights()

    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.models.pose_guider import PoseGuider
    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline

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
    _controller = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    ).to("cuda", dtype=dtype)
    print("Moore-AnimateAnyone pipeline loaded successfully")
    return _controller


def _encode_file(path: Path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def handler(event):
    try:
        data = (event or {}).get("input") or {}
        image_value = data.get("character_image") or data.get("reference_image") or data.get("image_reference") or data.get("image")
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
            # Crucially, ensure repository/source exists before importing any src.* module.
            _ensure_repository()
            from src.utils.util import get_fps, save_videos_grid, read_frames

            if driving_value and not pose_value:
                pose_path = work / "input_video_kps.mp4"
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "tools/vid2pose.py"), "--video_path", str(video_path)],
                    cwd=ROOT, text=True, capture_output=True,
                )
                if proc.returncode != 0 or not pose_path.exists():
                    raise RuntimeError("DWPose extraction failed.\n" + proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
            else:
                pose_path = video_path

            pipe = _get_controller()
            src_fps = fps_override if fps_override > 0 else max(1, int(get_fps(str(video_path))))
            pose_frames = read_frames(str(pose_path))
            if not pose_frames:
                raise RuntimeError("No frames could be read from the pose video.")
            length = min(length, len(pose_frames))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode():
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
        return {
            "status": "success",
            "video_base64": _encode_file(output_path),
            "content_type": "video/mp4",
            "num_frames": length,
            "fps": src_fps,
            "width": width,
            "height": height,
            "file_size_bytes": output_path.stat().st_size,
        }
    except Exception as exc:
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "error", "message": str(exc)}
    finally:
        gc.collect()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

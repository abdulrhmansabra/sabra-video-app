
"""RunPod Serverless handler for Moore-AnimateAnyone.

Downloads the official repository and missing model weights into /tmp.
Inputs:
  reference_image: image URL or base64/data URI
  driving_video: raw motion video URL or base64
  pose_video: pre-rendered pose video URL or base64
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import base64
import gc
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import requests
import torch
from PIL import Image
import runpod


# Official repository and temporary storage
REPO_URL = "https://github.com/MooreThreads/Moore-AnimateAnyone.git"
ROOT = Path(
    os.getenv("ANIMATEANYONE_ROOT", "/app/Moore-AnimateAnyone")
).resolve()

WEIGHTS = ROOT / "pretrained_weights"
CONFIG = ROOT / "configs/prompts/animation.yaml"

PIPELINE_LOCK = threading.Lock()
_controller = None


def _run(command, cwd=None, timeout=600):
    """Run a command and report useful logs on failure."""
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, command))}\n"
            f"STDOUT:\n{result.stdout[-5000:]}\n"
            f"STDERR:\n{result.stderr[-5000:]}"
        )

    return result


def _ensure_repository():
    """Clone the official repository into temporary storage if necessary."""
    source_file = ROOT / "src" / "utils" / "util.py"
    downloader = ROOT / "tools" / "download_weights.py"

    if (
        source_file.is_file()
        and downloader.is_file()
        and CONFIG.is_file()
    ):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        return

    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError(
            f"Incomplete repository found at {ROOT}. "
            "Refusing to delete existing files. Check "
            "ANIMATEANYONE_ROOT or clear the incomplete temporary directory."
        )

    if shutil.which("git") is None:
        raise RuntimeError(
            "Git is not installed. Install git in the Docker image "
            "to download the official repository."
        )

    ROOT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cloning official repository: {REPO_URL}")

    _run(
        ["git", "clone", "--depth", "1", REPO_URL, str(ROOT)],
        timeout=600,
    )

    missing = [
        str(path)
        for path in (source_file, downloader, CONFIG)
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Official repository checkout is incomplete. Missing: "
            + ", ".join(missing)
        )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.chdir(ROOT)

    print(f"Official repository ready at {ROOT}")


def _download_input(value, dest: Path, timeout=120):
    """Download an input URL or decode a base64 string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid input: {dest.name}")

    value = value.strip()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if value.startswith(("http://", "https://")):
        with requests.get(
            value, stream=True, timeout=timeout
        ) as response:
            response.raise_for_status()

            with open(dest, "wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
    else:
        encoded = (
            value.split(",", 1)[1]
            if value.startswith("data:") and "," in value
            else value
        )

        try:
            dest.write_bytes(base64.b64decode(encoded, validate=True))
        except Exception as exc:
            raise ValueError(
                f"{dest.name} must be a URL or base64/data URI"
            ) from exc

    if not dest.is_file() or dest.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {dest.name}")

    return dest


def _ensure_weights():
    """Download missing weights using the official repository script."""
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
        print("Required model files already exist.")
        return

    if os.getenv("AUTO_DOWNLOAD_WEIGHTS", "1").lower() not in (
        "1", "true", "yes"
    ):
        raise RuntimeError(
            "Model files are missing and AUTO_DOWNLOAD_WEIGHTS is disabled: "
            + ", ".join(missing)
        )

    downloader = ROOT / "tools" / "download_weights.py"

    if not downloader.is_file():
        raise RuntimeError(f"Official downloader not found: {downloader}")

    print("Downloading missing model weights into temporary storage...")

    _run(
        [sys.executable, str(downloader)],
        cwd=ROOT,
        timeout=3600,
    )

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            "Weight download finished, but required files are still missing: "
            + ", ".join(missing)
        )

    print("Required model weights are available.")


def _get_controller():
    """Initialize the GPU pipeline once per worker."""
    global _controller

    if _controller is not None:
        return _controller

    _ensure_repository()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. Attach this endpoint to a GPU worker."
        )

    if not CONFIG.is_file():
        raise RuntimeError(f"Model configuration not found: {CONFIG}")

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

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path
    ).to("cuda", dtype=dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=dtype)

    inference_cfg = OmegaConf.load(config.inference_config)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=inference_cfg.unet_additional_kwargs,
    ).to(device="cuda", dtype=dtype)

    pose_guider = PoseGuider(
        320,
        block_out_channels=(16, 32, 96, 256),
    ).to(device="cuda", dtype=dtype)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to("cuda", dtype=dtype)

    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )

    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu")
    )

    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu")
    )

    scheduler = DDIMScheduler(
        **OmegaConf.to_container(inference_cfg.noise_scheduler_kwargs)
    )

    _controller = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    ).to("cuda", dtype=dtype)

    print("Moore-AnimateAnyone pipeline loaded successfully.")

    return _controller


def _encode_file(path: Path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def handler(event):
    try:
        data = (event or {}).get("input") or {}

        image_value = (
            data.get("reference_image")
            or data.get("character_image")
            or data.get("image_reference")
            or data.get("image")
        )

        driving_value = (
            data.get("driving_video")
            or data.get("video_reference")
        )

        pose_value = data.get("pose_video")

        if not image_value:
            raise ValueError(
                "Provide reference_image as an image URL or base64."
            )

        if not driving_value and not pose_value:
            raise ValueError(
                "Provide driving_video or pose_video."
            )

        width = int(data.get("width", 512))
        height = int(data.get("height", 768))
        length = int(data.get("num_frames", data.get("length", 24)))
        steps = int(data.get("num_inference_steps", 25))
        guidance = float(data.get("guidance_scale", 3.5))
        fps_override = int(data.get("fps", 0))
        seed = int(data.get("seed", 123))

        if width < 64 or height < 64:
            raise ValueError("width and height must be at least 64.")

        if length < 1 or length > 128:
            raise ValueError("num_frames must be between 1 and 128.")

        work = Path("/tmp") / f"animateanyone_{os.getpid()}"
        work.mkdir(parents=True, exist_ok=True)

        image_path = _download_input(
            image_value, work / "reference_image.bin"
        )

        with Image.open(image_path) as image:
            reference_image = image.convert("RGB")

        video_path = _download_input(
            pose_value or driving_value,
            work / "input_video.mp4",
        )

        with PIPELINE_LOCK:
            # Ensure repository exists BEFORE importing src.
            _ensure_repository()

            from src.utils.util import (
                get_fps,
                save_videos_grid,
                read_frames,
            )

            if driving_value and not pose_value:
                pose_path = work / "input_video_kps.mp4"

                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools/vid2pose.py"),
                        "--video_path",
                        str(video_path),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )

                if proc.returncode != 0 or not pose_path.exists():
                    raise RuntimeError(
                        "DWPose extraction failed.\n"
                        + proc.stdout[-3000:]
                        + "\n"
                        + proc.stderr[-3000:]
                    )
            else:
                pose_path = video_path

            pipe = _get_controller()

            src_fps = (
                fps_override
                if fps_override > 0
                else max(1, int(get_fps(str(video_path))))
            )

            pose_frames = read_frames(str(pose_path))

            if not pose_frames:
                raise RuntimeError(
                    "No frames could be read from the pose video."
                )

            length = min(length, len(pose_frames))

            generator = torch.Generator(
                device="cuda"
            ).manual_seed(seed)

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

            output_path = work / "animated_video.mp4"

            save_videos_grid(
                result,
                str(output_path),
                n_rows=1,
                fps=src_fps,
            )

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError("Generated MP4 is missing or empty.")

        return {
            "status": "success",
            "video_base64": _encode_file(output_path),
            "content_type": "video/mp4",
            "filename": "animated_video.mp4",
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

        return {
            "status": "error",
            "message": str(exc),
        }

    finally:
        gc.collect()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

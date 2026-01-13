"""
Modal deployment for LTX-2 19B 8-Step Distilled

Fast 8-step text-to-video and image-to-video generation using the distilled LTX-2 model.
Default resolution: 1216x704 @ 30 FPS
"""

import modal
from pathlib import Path

# ============================================================================
# Volume Configuration
# ============================================================================

model_volume = modal.Volume.from_name("ltx2-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("ltx2-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("ltx2-cache", create_if_missing=True)

# ============================================================================
# Container Image
# ============================================================================

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

def _get_repo_root() -> Path:
    """Resolve repo root on the *local* machine (Modal client side)."""
    return Path(__file__).resolve().parents[2]

base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "ffmpeg", "git", "build-essential", "ninja-build", "clang", "cmake",
        "rng-tools",
    )
    .run_commands("test -c /dev/urandom || mknod -m 644 /dev/urandom c 1 9")
    .pip_install(
        "wheel",
        "packaging",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "transformers[accelerate]==4.49.0",
    )
    # Flash attention
    .run_commands("pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation")
    # LTX-2 / lightx2v dependencies
    .pip_install(
        "diffusers>=0.32.0",
        "peft==0.17.0",
        "einops==0.8.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "pillow==11.3.0",
        "numpy==1.26.4",
        "scipy",
        "tqdm==4.67.1",
        "loguru==0.7.3",
        "safetensors==0.4.5",
        "huggingface-hub==0.34.0",
        "huggingface_hub[cli]",
        "sentencepiece",  # For T5 tokenizer
        "gguf",
        "easydict",
        "requests",
        "fastapi[standard]>=0.115.0",
        "python-multipart",
        "uvicorn[standard]",
        "prometheus-client",
    )
    .env({
        "HF_HOME": "/models/hf_cache",
        "PYTHONPATH": "/opt/lightx2v",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_LAUNCH_BLOCKING": "0",
        "OPENSSL_ia32cap": "~0x200000200000000",
        "LIGHTX2V_ENABLE_FLASH_ATTN": "1",
        "LIGHTX2V_ENABLE_SAGE_ATTN": "0",
        "LIGHTX2V_ENABLE_DRAFT_ATTN": "0",
        "LIGHTX2V_ATTN_MODE": "flash_attn2",
        "TORCH_CUDA_ARCH_LIST": "9.0",
    })
)

# Use local source code (this repo) instead of cloning an upstream repo that may not include LTX2.
# Important: this must only run on the Modal *client* (your laptop), not inside the remote container import.
image = base_image
if modal.is_local():
    _repo_root = _get_repo_root()
    image = (
        base_image
        .add_local_dir(str(_repo_root / "lightx2v"), "/opt/lightx2v/lightx2v", copy=True)
        .add_local_dir(str(_repo_root / "lightx2v_platform"), "/opt/lightx2v/lightx2v_platform", copy=True)
        .add_local_dir(str(_repo_root / "configs"), "/opt/lightx2v/configs", copy=True)
        # Remove PyAV to avoid SIGABRT issues
        .run_commands("pip uninstall -y av || true")
        # Patch LightX2V pipeline to only import LTX2 runners
        .run_commands(
            "python -c \"import re; from pathlib import Path; "
            "p=Path('/opt/lightx2v/lightx2v/pipeline.py'); "
            "t=p.read_text(); "
            "t=re.sub(r'^from lightx2v\\\\.models\\\\.runners\\\\..*\\\\n','',t,flags=re.M); "
            "ins='from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner  # noqa: F401\\\\n'"
            "'from lightx2v.models.runners.ltx2.ltx2_distill_runner import LTX2DistillRunner  # noqa: F401\\\\n\\\\n'"
            "'import os as _os\\\\n'"
            "'if _os.environ.get(\\\\\\\"LIGHTX2V_IMPORT_ALL_RUNNERS\\\\\\\",\\\\\\\"0\\\\\\\") == \\\\\\\"1\\\\\\\":\\\\n'"
            "'    from lightx2v.models.runners.wan.wan_runner import WanRunner  # noqa: F401\\\\n'"
            "'    from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401\\\\n'; "
            "t=t.replace('from loguru import logger\\\\n\\\\n','from loguru import logger\\\\n\\\\n'+ins+'\\\\n',1); "
            "p.write_text(t); "
            "print('Patched LightX2V pipeline runner imports for LTX2')\""
        )
        # Patch infer.py similarly
        .run_commands(
            "python -c \"import re; from pathlib import Path; "
            "p=Path('/opt/lightx2v/lightx2v/infer.py'); "
            "t=p.read_text(); "
            "t=re.sub(r'^from lightx2v\\\\.models\\\\.runners\\\\..*\\\\n','',t,flags=re.M); "
            "ins='from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner  # noqa: F401\\\\n'"
            "'from lightx2v.models.runners.ltx2.ltx2_distill_runner import LTX2DistillRunner  # noqa: F401\\\\n'; "
            "t=t.replace('from lightx2v.common.ops import *\\\\n', 'from lightx2v.common.ops import *\\\\n'+ins+'\\\\n', 1); "
            "p.write_text(t); "
            "print('Patched LightX2V infer runner imports for LTX2')\""
        )
        # Patch attention backends for lazy loading
        .run_commands(
            "python -c \"from pathlib import Path; "
            "p=Path('/opt/lightx2v/lightx2v/common/ops/attn/__init__.py'); "
            "txt='import os\\n\\n"
            "from .torch_sdpa import TorchSDPAWeight\\n"
            "from .ulysses_attn import Ulysses4090AttnWeight, UlyssesAttnWeight\\n"
            "from .ring_attn import RingAttnWeight\\n\\n"
            "if os.environ.get(\\\"LIGHTX2V_ENABLE_FLASH_ATTN\\\", \\\"0\\\") == \\\"1\\\":\\n"
            "    from .flash_attn import FlashAttn2Weight, FlashAttn3Weight\\n\\n"
            "if os.environ.get(\\\"LIGHTX2V_ENABLE_SAGE_ATTN\\\", \\\"0\\\") == \\\"1\\\":\\n"
            "    from .sage_attn import SageAttn2Weight, SageAttn3Weight\\n\\n"
            "if os.environ.get(\\\"LIGHTX2V_ENABLE_DRAFT_ATTN\\\", \\\"0\\\") == \\\"1\\\":\\n"
            "    from .draft_attn import DraftAttnWeight\\n'; "
            "p.write_text(txt) if p.exists() else None; "
            "print('Patched' if p.exists() else 'Did not find', p)\""
        )
    )

# Add demo images (optional): Modal's `Image.add_local_dir` API doesn't support `condition=` in newer SDKs,
# so we do the conditional logic in Python instead.
_demo_images_dir = None
for _pth in (Path("demo_images"), Path(__file__).resolve().parent / "demo_images"):
    if _pth.exists() and _pth.is_dir():
        _demo_images_dir = _pth
        break

if _demo_images_dir is not None:
    image = image.add_local_dir(str(_demo_images_dir), "/root/demo_images")

app = modal.App("ltx2-distill", image=image)

# ============================================================================
# Model Download Function
# ============================================================================

@app.function(
    volumes={"/models": model_volume},
    timeout=3600,
)
def download_models():
    """Download LTX-2 19B distilled model and dependencies."""
    import os
    from huggingface_hub import snapshot_download

    # LTX-2 19B Distilled model
    ltx2_path = "/models/Lightricks/LTX-2"
    if not os.path.exists(ltx2_path):
        print("📥 Downloading LTX-2 19B model...")
        snapshot_download(
            "Lightricks/LTX-2",
            local_dir=ltx2_path,
            local_dir_use_symlinks=False,
        )
    else:
        print("✅ LTX-2 model already cached")

    # Also download the distilled checkpoint specifically
    distill_files = [
        "ltx-2-19b-distilled.safetensors",
        "ltx-2-19b-distilled-fp8.safetensors",
    ]
    missing = [f for f in distill_files if not os.path.exists(os.path.join(ltx2_path, f))]
    if missing:
        print(f"📥 Fetching distilled checkpoints: {missing}")
        snapshot_download(
            "Lightricks/LTX-2",
            local_dir=ltx2_path,
            local_dir_use_symlinks=False,
            allow_patterns=missing,
        )

    # Download T5-XXL text encoder (if not bundled)
    t5_path = "/models/google/t5-v1_1-xxl"
    if not os.path.exists(t5_path):
        print("📥 Downloading T5-v1.1-XXL text encoder...")
        snapshot_download(
            "google/t5-v1_1-xxl",
            local_dir=t5_path,
            local_dir_use_symlinks=False,
        )
    else:
        print("✅ T5 text encoder already cached")

    # Commit volume
    from modal import Volume
    Volume.from_name("ltx2-models").commit()
    print("✅ All models downloaded!")


# ============================================================================
# LTX-2 Inference Engine
# ============================================================================

@app.cls(
    gpu="H100",  # LTX-2 19B needs ~40GB VRAM
    timeout=600,
    scaledown_window=120,
    max_containers=4,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={
        "/models": model_volume,
        "/outputs": outputs_volume,
        "/vol_cache": cache_volume,
    },
)
class LTX2Engine:
    """
    Fast 8-step text-to-video and image-to-video generation engine.

    Uses the distilled LTX-2 19B model via lightx2v for fast inference.
    Default: 1216x704 @ 30 FPS
    """

    @modal.enter(snap=True)
    def load_model(self):
        """Load the LTX-2 distilled model on container startup."""
        import os
        import subprocess

        # Set cache directories
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/vol_cache/inductor")
        os.environ.setdefault("TRITON_CACHE_DIR", "/vol_cache/triton")
        os.environ.setdefault("XDG_CACHE_HOME", "/vol_cache")
        os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

        # Start entropy daemon
        try:
            subprocess.run(["rngd", "-r", "/dev/urandom", "-f"],
                         check=False, capture_output=True, timeout=2)
        except Exception:
            pass

        import torch

        # Initialize CUDA
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()

        print("🔧 Loading LTX-2 19B (8-step distilled)...")

        from lightx2v import LightX2VPipeline

        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        # Determine model and text encoder paths
        model_path = "/models/Lightricks/LTX-2"
        text_encoder_path = "/models/google/t5-v1_1-xxl"

        # Check for distilled checkpoint(s)
        import glob
        distill_ckpts = glob.glob(f"{model_path}/*distilled*.safetensors")
        distill_ckpt = None
        lora_ckpt = None
        if distill_ckpts:
            # Prefer non-LoRA full distilled weights if present; otherwise fall back to a distilled LoRA adapter.
            non_lora = [p for p in distill_ckpts if "lora" not in p.lower()]
            lora_only = [p for p in distill_ckpts if "lora" in p.lower()]

            prefer_fp8 = os.environ.get("LTX2_PREFER_FP8", "0") == "1"
            candidates = sorted(non_lora) if non_lora else []
            if candidates:
                for ckpt in candidates:
                    if prefer_fp8 and "fp8" in ckpt.lower():
                        distill_ckpt = ckpt
                        break
                    if not prefer_fp8 and "fp8" not in ckpt.lower():
                        distill_ckpt = ckpt
                        break
                if distill_ckpt is None:
                    distill_ckpt = candidates[0]
            elif lora_only:
                # No full distilled weights found; use LoRA adapter on top of the base weights in `model_path`.
                lora_ckpt = sorted(lora_only)[0]

        print(f"   Model path: {model_path}")
        print(f"   Text encoder: {text_encoder_path}")
        if distill_ckpt:
            print(f"   Distilled checkpoint: {distill_ckpt}")
        if lora_ckpt:
            print(f"   Distilled LoRA checkpoint: {lora_ckpt}")

        self.pipe = LightX2VPipeline(
            model_path=model_path,
            model_cls="ltx2_distill",
            task="t2v",
            # Prefer loading the official distilled checkpoint directly when available.
            # (We also normalize official key names like `patchify_proj.*` → `patch_embed.proj.*` in the loader.)
            dit_original_ckpt=distill_ckpt,
        )

        # Set text encoder path in config
        self.pipe.text_encoder_path = text_encoder_path

        # Attention mode
        self.attn_mode = os.environ.get("LIGHTX2V_ATTN_MODE", "flash_attn2")

        # Enable offload to keep peak GPU memory under control (LTX-2 19B is very close to 80GB).
        self.pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=True,
            vae_offload=True,
        )

        # Default generator config for LTX-2
        self._generator_cfg = {
            "infer_steps": 8,
            "height": 704,
            "width": 1216,
            "num_frames": 97,  # ~3.2s at 30fps
            "guidance_scale": 1,  # No CFG for distilled
            "sample_shift": 3.0,
            # Use the repo config to normalize values like `patch_size`/`vae_stride` (some upstream configs use scalars).
            "config_json": "/opt/lightx2v/configs/ltx2/ltx2_t2v_distill_8step_offload.json",
        }

        # If we only have a LoRA distilled checkpoint, apply it on top of the base weights.
        if lora_ckpt and not distill_ckpt:
            self.pipe.enable_lora([{"path": lora_ckpt, "strength": 1.0}])

        try:
            self.pipe.create_generator(attn_mode=self.attn_mode, **self._generator_cfg)
        except Exception as e:
            if self.attn_mode != "torch_sdpa":
                print(f"⚠ Failed to init with attn_mode={self.attn_mode}: {e}")
                print("↪ Falling back to torch_sdpa")
                self.attn_mode = "torch_sdpa"
                self.pipe.create_generator(attn_mode=self.attn_mode, **self._generator_cfg)
            else:
                raise

        # Warmup
        if os.environ.get("ENABLE_TORCH_COMPILE", "1") != "0":
            print("🔥 Warming up...")
            self._warmup()

        print("✅ LTX-2 model loaded and ready!")

    def _warmup(self):
        """Warmup inference to trigger JIT compilation."""
        import torch
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out:
            out_path = out.name

        with torch.inference_mode():
            self.pipe.generate(
                seed=42,
                prompt="warmup test",
                negative_prompt="",
                save_result_path=out_path,
            )

        import os
        try:
            os.remove(out_path)
        except Exception:
            pass

        torch.cuda.synchronize()
        print("✅ Warmup complete")

    @modal.method()
    def generate_t2v(
        self,
        prompt: str = "A beautiful sunset over the ocean with gentle waves",
        negative_prompt: str = "",
        num_frames: int = 97,
        height: int = 704,
        width: int = 1216,
        seed: int = 42,
    ) -> dict:
        """
        Generate video from text using 8-step distilled inference.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            num_frames: Number of output frames
            height: Video height (divisible by 32)
            width: Video width (divisible by 32)
            seed: Random seed

        Returns:
            dict with 'video_b64' (base64 MP4) and timing info
        """
        import torch
        import base64
        import time
        import tempfile
        import os

        t0 = time.perf_counter()

        # Ensure dimensions are valid
        height = (height // 32) * 32
        width = (width // 32) * 32

        # Recreate generator if config differs
        if (self._generator_cfg.get("num_frames") != num_frames or
            self._generator_cfg.get("height") != height or
            self._generator_cfg.get("width") != width):
            self._generator_cfg.update({
                "num_frames": num_frames,
                "height": height,
                "width": width,
            })
            self.pipe.create_generator(attn_mode=self.attn_mode, **self._generator_cfg)

        t_prep = time.perf_counter() - t0

        # Generate video
        t_gen0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            out_path = tmp_vid.name

        with torch.inference_mode():
            self.pipe.generate(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                save_result_path=out_path,
            )

        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen0

        # Read MP4 and return as base64
        t_enc0 = time.perf_counter()
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        t_enc = time.perf_counter() - t_enc0

        try:
            os.remove(out_path)
        except Exception:
            pass

        return {
            "video_b64": video_b64,
            "num_frames": num_frames,
            "resolution": f"{width}x{height}",
            "timing": {
                "prep_ms": t_prep * 1000,
                "generate_ms": t_gen * 1000,
                "encode_ms": t_enc * 1000,
                "total_ms": (time.perf_counter() - t0) * 1000,
                "fps": num_frames / t_gen,
            }
        }

    @modal.method()
    def generate_i2v(
        self,
        image_b64: str,
        prompt: str = "Smooth camera motion with cinematic quality",
        negative_prompt: str = "",
        num_frames: int = 97,
        seed: int = 42,
    ) -> dict:
        """
        Generate video from image using 8-step distilled inference.

        Args:
            image_b64: Base64-encoded input image
            prompt: Text prompt describing motion
            negative_prompt: Negative prompt (optional)
            num_frames: Number of output frames
            seed: Random seed

        Returns:
            dict with 'video_b64' (base64 MP4) and timing info
        """
        import torch
        import base64
        import io
        import time
        import tempfile
        import os
        from PIL import Image

        t0 = time.perf_counter()

        # Decode input image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Get dimensions from image (ensure divisible by 32)
        w, h = image.size
        w = (w // 32) * 32
        h = (h // 32) * 32
        if (w, h) != image.size:
            image = image.resize((w, h))

        # Save image for pipeline
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            img_path = tmp_img.name
        image.save(img_path)

        t_prep = time.perf_counter() - t0

        # Recreate generator for i2v if needed
        if (self._generator_cfg.get("num_frames") != num_frames or
            self._generator_cfg.get("height") != h or
            self._generator_cfg.get("width") != w):
            self._generator_cfg.update({
                "num_frames": num_frames,
                "height": h,
                "width": w,
            })
            # Switch task to i2v
            self.pipe.task = "i2v"
            self.pipe.create_generator(attn_mode=self.attn_mode, **self._generator_cfg)

        # Generate video
        t_gen0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            out_path = tmp_vid.name

        with torch.inference_mode():
            self.pipe.generate(
                seed=seed,
                image_path=img_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                save_result_path=out_path,
            )

        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen0

        # Read MP4 and return as base64
        t_enc0 = time.perf_counter()
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        t_enc = time.perf_counter() - t_enc0

        # Cleanup
        for path in [img_path, out_path]:
            try:
                os.remove(path)
            except Exception:
                pass

        return {
            "video_b64": video_b64,
            "num_frames": num_frames,
            "resolution": f"{w}x{h}",
            "timing": {
                "prep_ms": t_prep * 1000,
                "generate_ms": t_gen * 1000,
                "encode_ms": t_enc * 1000,
                "total_ms": (time.perf_counter() - t0) * 1000,
                "fps": num_frames / t_gen,
            }
        }


# ============================================================================
# HTTP API
# ============================================================================

@app.function(timeout=900)
@modal.asgi_app()
def api():
    """REST API for LTX-2 video generation."""
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.responses import FileResponse, Response, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional
    import base64
    import tempfile

    web_app = FastAPI(title="LTX-2 19B Distilled API")

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class T2VRequest(BaseModel):
        prompt: str
        negative_prompt: str = ""
        num_frames: int = 97
        height: int = 704
        width: int = 1216
        seed: int = 42

    class I2VRequest(BaseModel):
        image_b64: str
        prompt: str = "Smooth camera motion with cinematic quality"
        negative_prompt: str = ""
        num_frames: int = 97
        seed: int = 42

    @web_app.post("/generate/t2v")
    async def generate_t2v(req: T2VRequest):
        """Generate video from text."""
        try:
            engine = LTX2Engine()
            result = engine.generate_t2v.remote(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_frames=req.num_frames,
                height=req.height,
                width=req.width,
                seed=req.seed,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/generate/i2v")
    async def generate_i2v(req: I2VRequest):
        """Generate video from image."""
        try:
            engine = LTX2Engine()
            result = engine.generate_i2v.remote(
                image_b64=req.image_b64,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_frames=req.num_frames,
                seed=req.seed,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/demo", response_class=HTMLResponse)
    async def demo_page():
        return HTMLResponse("""
<!doctype html>
<html>
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>LTX-2 Video Generation Demo</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .row { display: flex; gap: 24px; flex-wrap: wrap; }
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); flex: 1; min-width: 300px; }
        label { display: block; font-weight: 600; margin-top: 12px; color: #555; }
        input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-top: 4px; }
        textarea { min-height: 80px; resize: vertical; }
        button { margin-top: 16px; padding: 12px 24px; border-radius: 8px; border: none; background: #0066ff; color: white; cursor: pointer; font-size: 16px; }
        button:hover { background: #0052cc; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        video { width: 100%; border-radius: 12px; margin-top: 12px; background: #000; }
        .status { margin-top: 12px; padding: 12px; background: #f0f0f0; border-radius: 8px; font-family: monospace; white-space: pre-wrap; }
        .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
        .tab { padding: 8px 16px; border-radius: 8px; cursor: pointer; background: #e0e0e0; }
        .tab.active { background: #0066ff; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 LTX-2 19B Video Generation</h1>
        <p>8-step distilled inference • 1216×704 @ 30 FPS</p>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('t2v')">Text to Video</div>
            <div class="tab" onclick="switchTab('i2v')">Image to Video</div>
        </div>

        <div class="row">
            <div class="card" id="input-card">
                <h3>Input</h3>

                <div id="t2v-inputs">
                    <label>Prompt</label>
                    <textarea id="prompt">A majestic eagle soaring through a golden sunset sky, with vibrant orange and purple clouds reflecting on a calm ocean below. Cinematic, 4K quality.</textarea>
                </div>

                <div id="i2v-inputs" style="display:none;">
                    <label>Image</label>
                    <input type="file" id="image-file" accept="image/*"/>
                    <img id="image-preview" style="max-width:100%; margin-top:8px; border-radius:8px; display:none;"/>

                    <label>Motion Prompt</label>
                    <textarea id="motion-prompt">Smooth camera zoom with gentle movement</textarea>
                </div>

                <label>Frames</label>
                <input type="number" id="frames" value="97" min="33" max="257" step="8"/>

                <label>Seed</label>
                <input type="number" id="seed" value="42"/>

                <button id="generate-btn" onclick="generate()">Generate Video</button>
                <div id="status" class="status" style="display:none;"></div>
            </div>

            <div class="card">
                <h3>Output</h3>
                <video id="video" controls playsinline></video>
                <a id="download" href="#" download="ltx2_output.mp4" style="display:none; margin-top:8px;">Download MP4</a>
            </div>
        </div>
    </div>

    <script>
        let currentTab = 't2v';

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab === 't2v' ? 1 : 2})`).classList.add('active');
            document.getElementById('t2v-inputs').style.display = tab === 't2v' ? 'block' : 'none';
            document.getElementById('i2v-inputs').style.display = tab === 'i2v' ? 'block' : 'none';
        }

        document.getElementById('image-file').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('image-preview').src = e.target.result;
                    document.getElementById('image-preview').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        async function generate() {
            const btn = document.getElementById('generate-btn');
            const status = document.getElementById('status');
            const video = document.getElementById('video');
            const download = document.getElementById('download');

            btn.disabled = true;
            status.style.display = 'block';
            status.textContent = 'Generating...';
            video.removeAttribute('src');
            download.style.display = 'none';

            try {
                const t0 = performance.now();
                let response;

                if (currentTab === 't2v') {
                    response = await fetch('/generate/t2v', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: document.getElementById('prompt').value,
                            num_frames: parseInt(document.getElementById('frames').value),
                            seed: parseInt(document.getElementById('seed').value),
                        })
                    });
                } else {
                    const file = document.getElementById('image-file').files[0];
                    if (!file) {
                        throw new Error('Please select an image');
                    }
                    const imageB64 = await new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onload = () => resolve(reader.result.split(',')[1]);
                        reader.readAsDataURL(file);
                    });

                    response = await fetch('/generate/i2v', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            image_b64: imageB64,
                            prompt: document.getElementById('motion-prompt').value,
                            num_frames: parseInt(document.getElementById('frames').value),
                            seed: parseInt(document.getElementById('seed').value),
                        })
                    });
                }

                if (!response.ok) {
                    throw new Error(await response.text());
                }

                const result = await response.json();
                const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

                status.textContent = `Done in ${elapsed}s\\n` +
                    `Generation: ${result.timing.generate_ms.toFixed(0)}ms\\n` +
                    `FPS: ${result.timing.fps.toFixed(2)}`;

                const videoBlob = new Blob(
                    [Uint8Array.from(atob(result.video_b64), c => c.charCodeAt(0))],
                    {type: 'video/mp4'}
                );
                const videoUrl = URL.createObjectURL(videoBlob);
                video.src = videoUrl;
                download.href = videoUrl;
                download.style.display = 'block';

            } catch (e) {
                status.textContent = 'Error: ' + e.message;
            }

            btn.disabled = false;
        }
    </script>
</body>
</html>
        """.strip())

    @web_app.get("/test")
    async def test(
        prompt: str = "A beautiful sunset over the ocean with gentle waves",
        num_frames: int = 97,
        seed: int = 42,
    ):
        """Test T2V generation and return MP4."""
        try:
            engine = LTX2Engine()
            result = engine.generate_t2v.remote(
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
            )

            video_data = base64.b64decode(result["video_b64"])

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_data)
                tmp_path = tmp.name

            return FileResponse(
                tmp_path,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": 'attachment; filename="ltx2_test.mp4"',
                    "X-Generation-Time-Ms": str(result["timing"]["total_ms"]),
                    "X-Generation-FPS": str(result["timing"]["fps"]),
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": "LTX-2 19B Distilled (8-step)",
            "default_resolution": "1216x704",
            "fps": 30,
        }

    return web_app


# ============================================================================
# CLI
# ============================================================================

@app.local_entrypoint()
def main(
    download: bool = False,
    test: bool = False,
    prompt: str = "A majestic eagle soaring through a golden sunset sky",
    frames: int = 97,
    seed: int = 42,
    output: str = "ltx2_output.mp4",
):
    """
    CLI entrypoint for LTX-2 19B 8-step distilled video generation.

    Args:
        download: Download model weights
        test: Run test generation
        prompt: Text prompt for T2V generation
        frames: Number of output frames
        seed: Random seed
        output: Output video filename
    """
    if download:
        print("📥 Downloading LTX-2 models...")
        download_models.remote()
    elif test:
        test_t2v(prompt, frames, seed, output)
    else:
        print("Usage:")
        print("  modal run ltx2_modal.py --download          # Download models")
        print("  modal run ltx2_modal.py --test              # Test T2V generation")
        print("  modal run ltx2_modal.py --test --prompt 'your prompt'")
        print("")
        print("Test options:")
        print("  --prompt <text>   Prompt for generation")
        print("  --frames <int>    Number of frames (default: 97)")
        print("  --seed <int>      Random seed (default: 42)")
        print("  --output <path>   Output video path (default: ltx2_output.mp4)")
        print("")
        print("Note: Uses 8-step distilled inference at 1216x704 @ 30fps!")


def test_t2v(prompt: str, num_frames: int, seed: int, output_path: str):
    """Test T2V generation."""
    import base64
    from pathlib import Path

    print(f"📝 Prompt: {prompt}")
    print(f"🎬 Frames: {num_frames}")
    print(f"⚡ Steps: 8 (distilled)")
    print(f"🎲 Seed: {seed}")
    print("")
    print("🚀 Starting generation on Modal...")

    engine = LTX2Engine()
    result = engine.generate_t2v.remote(
        prompt=prompt,
        num_frames=num_frames,
        seed=seed,
    )

    video_data = base64.b64decode(result["video_b64"])
    output_file = Path(output_path)
    output_file.write_bytes(video_data)

    timing = result["timing"]
    print("")
    print("✅ Generation complete!")
    print(f"📁 Saved to: {output_file.absolute()}")
    print(f"📐 Resolution: {result['resolution']}")
    print("")
    print("⏱️  Timing:")
    print(f"   Prep:     {timing['prep_ms']:.1f} ms")
    print(f"   Generate: {timing['generate_ms']:.1f} ms ({timing['fps']:.2f} fps)")
    print(f"   Encode:   {timing['encode_ms']:.1f} ms")
    print(f"   Total:    {timing['total_ms']:.1f} ms")

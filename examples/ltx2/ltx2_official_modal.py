"""
Modal deployment using the **official Lightricks LTX-2 distilled pipeline** (not LightX2V).

Features:
- GPU memory snapshotting for fast cold starts
- FP8 transformer for reduced memory footprint
- 8-step distilled inference with 2x spatial upscaling
- Text-to-video (T2V) and Image-to-video (I2V) support
- Web UI for easy interaction
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ============================================================================
# Volumes
# ============================================================================

model_volume = modal.Volume.from_name("ltx2-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("ltx2-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("ltx2-cache", create_if_missing=True)

# ============================================================================
# Image
# ============================================================================

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

LTX2_REPO_DIR = "/opt/ltx2"
MODELS_DIR = "/models"
LTX2_MODELS_DIR = f"{MODELS_DIR}/Lightricks/LTX-2"
GEMMA_DIR = f"{MODELS_DIR}/gemma"
DEFAULT_GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "ffmpeg",
        "git",
        "build-essential",
        "libjpeg-turbo8",
        "libpng16-16",
        "libgl1",
    )
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.50.0",
        "accelerate",
        "einops",
        "safetensors",
        "huggingface-hub",
        "pillow",
        "numpy",
        "scipy",
        "tqdm",
        "torchaudio==2.5.1",
        "av",
        # Web API
        "fastapi[standard]>=0.115.0",
        "python-multipart",
        "uvicorn[standard]",
    )
    .run_commands(
        f"git clone https://github.com/Lightricks/LTX-2.git {LTX2_REPO_DIR}",
        f"pip install -e {LTX2_REPO_DIR}/packages/ltx-core --no-deps",
        f"pip install -e {LTX2_REPO_DIR}/packages/ltx-pipelines --no-deps",
        "python -c \"import torch; print('torch', torch.__version__)\"",
        "python -c \"from transformers.models.gemma3 import Gemma3ForConditionalGeneration; print('Gemma3 import OK')\"",
    )
    .env(
        {
            "HF_HOME": f"{MODELS_DIR}/hf_cache",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

app = modal.App("ltx2-official-distilled", image=image)


@app.function(
    volumes={MODELS_DIR: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_models():
    """Download the official LTX-2 weights into the shared model volume."""
    from huggingface_hub import snapshot_download

    os.makedirs(LTX2_MODELS_DIR, exist_ok=True)
    os.makedirs(GEMMA_DIR, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set. Ensure Modal secret `huggingface-secret` exports HF_TOKEN.")

    snapshot_download(
        "Lightricks/LTX-2",
        local_dir=LTX2_MODELS_DIR,
        local_dir_use_symlinks=False,
        token=hf_token,
    )

    gemma_repo_id = os.environ.get("LTX2_GEMMA_REPO_ID", DEFAULT_GEMMA_REPO_ID).strip()
    if gemma_repo_id:
        snapshot_download(
            gemma_repo_id,
            local_dir=GEMMA_DIR,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print(f"✅ Downloaded Gemma model: {gemma_repo_id} -> {GEMMA_DIR}")


@app.cls(
    gpu="H100",
    timeout=3600,
    scaledown_window=300,  # Keep warm for 5 minutes
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={
        MODELS_DIR: model_volume,
        "/outputs": outputs_volume,
        "/vol_cache": cache_volume,
    },
)
class OfficialLTX2Engine:
    """
    Run the official LTX-2 DistilledPipeline with GPU memory snapshotting.
    
    Optimized for lowest latency:
    - All models pre-loaded and kept in VRAM (~60GB)
    - torch.compile applied to transformer
    - GPU state snapshotted for instant cold starts
    """

    @modal.enter(snap=True)
    def load_model(self):
        """Load all models into VRAM and compile for maximum performance."""
        import torch
        
        # Set up cache directories for torch.compile artifacts
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/vol_cache/inductor")
        os.environ.setdefault("TRITON_CACHE_DIR", "/vol_cache/triton")
        os.environ.setdefault("XDG_CACHE_HOME", "/vol_cache")
        os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
        
        # Initialize CUDA
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        
        print("🔧 Loading LTX-2 DistilledPipeline (keeping all models in VRAM)...")
        
        from ltx_pipelines.distilled import DistilledPipeline
        
        ckpt = f"{LTX2_MODELS_DIR}/ltx-2-19b-distilled-fp8.safetensors"
        spatial_upsampler = f"{LTX2_MODELS_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        
        missing = [p for p in (ckpt, spatial_upsampler) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing required LTX-2 files. Run `--download` first.\n"
                + "\n".join(f"- {p}" for p in missing)
            )
        
        if not (Path(GEMMA_DIR).exists() and list(Path(GEMMA_DIR).rglob("model*.safetensors"))):
            raise FileNotFoundError(f"Missing Gemma model files under `{GEMMA_DIR}`.")
        
        # Create the pipeline
        self.pipeline = DistilledPipeline(
            checkpoint_path=ckpt,
            spatial_upsampler_path=spatial_upsampler,
            gemma_root=GEMMA_DIR,
            loras=[],
            fp8transformer=True,
        )
        
        # Pre-load ALL models into VRAM and keep references to prevent cleanup
        print("📦 Pre-loading all models into VRAM...")
        self._preload_models()
        
        # Note: torch.compile disabled - causes excessive recompilation with dynamic shapes
        # The LTX-2 model has variable tensor sizes which triggers constant recompiles
        # Main speedup comes from keeping models in VRAM, not compilation
        
        # Run warmup to warm up CUDA kernels
        print("🔥 Running warmup...")
        self._warmup()
        
        print("✅ All models loaded, compiled, and ready!")
        self._print_memory_usage()
    
    def _preload_models(self):
        """Pre-load all models and patch ModelLedger to return cached versions."""
        import torch
        
        ledger = self.pipeline.model_ledger
        
        # Load text encoder (Gemma) - ~24GB
        print("   Loading text encoder (Gemma)...")
        self._text_encoder = ledger.text_encoder()
        
        # Load transformer - ~19GB (FP8)
        print("   Loading transformer (19B FP8)...")
        self._transformer = ledger.transformer()
        
        # Load VAE components
        print("   Loading VAE encoder...")
        self._video_encoder = ledger.video_encoder()
        
        print("   Loading VAE decoder...")
        self._video_decoder = ledger.video_decoder()
        
        # Load spatial upsampler
        print("   Loading spatial upsampler...")
        self._spatial_upsampler = ledger.spatial_upsampler()
        
        torch.cuda.synchronize()
        print("   All models loaded!")
        
        # Monkey-patch the ModelLedger to return our cached models instead of reloading
        print("   Patching ModelLedger to use cached models...")
        self._patch_model_ledger(ledger)
    
    def _patch_model_ledger(self, ledger):
        """Patch ModelLedger methods to return cached models instead of reloading."""
        # Store original methods
        original_text_encoder = ledger.text_encoder
        original_transformer = ledger.transformer
        original_video_encoder = ledger.video_encoder
        original_video_decoder = ledger.video_decoder
        original_spatial_upsampler = ledger.spatial_upsampler
        
        # Create patched methods that return cached models
        cached_text_encoder = self._text_encoder
        cached_transformer = self._transformer
        cached_video_encoder = self._video_encoder
        cached_video_decoder = self._video_decoder
        cached_spatial_upsampler = self._spatial_upsampler
        
        def patched_text_encoder():
            return cached_text_encoder
        
        def patched_transformer():
            return cached_transformer
        
        def patched_video_encoder():
            return cached_video_encoder
        
        def patched_video_decoder():
            return cached_video_decoder
        
        def patched_spatial_upsampler():
            return cached_spatial_upsampler
        
        # Apply patches
        ledger.text_encoder = patched_text_encoder
        ledger.transformer = patched_transformer
        ledger.video_encoder = patched_video_encoder
        ledger.video_decoder = patched_video_decoder
        ledger.spatial_upsampler = patched_spatial_upsampler
        
        # Also disable cleanup_memory to prevent model unloading
        def noop_cleanup(*args, **kwargs):
            pass
        
        ledger.cleanup_memory = noop_cleanup
        
        print("   ModelLedger patched - models will stay in VRAM!")
        
        # Patch the pipeline to support FL2V (first+last frame conditioning)
        self._patch_pipeline_for_fl2v()
    
    def _patch_pipeline_for_fl2v(self):
        """
        Patch the conditioning function to use image_conditionings_by_adding_guiding_latent
        when multiple images are provided (for FL2V support).
        
        The DistilledPipeline uses image_conditionings_by_replacing_latent internally,
        which only supports single-frame conditioning. For FL2V (first+last frames),
        we need to use image_conditionings_by_adding_guiding_latent instead.
        """
        import ltx_pipelines.utils.helpers as helpers_module
        import ltx_pipelines.distilled as distilled_module
        
        # Get the guiding latent function
        guiding_fn = helpers_module.image_conditionings_by_adding_guiding_latent
        
        # Store original function from the distilled module (this is what actually gets called)
        original_replacing = distilled_module.image_conditionings_by_replacing_latent
        
        def smart_conditioning(images, height, width, video_encoder, dtype, device, **kwargs):
            """
            Smart conditioning that uses guiding latent for multiple images,
            and replacing latent for single image.
            """
            if len(images) > 1:
                print(f"   FL2V: Using guiding latent conditioning for {len(images)} keyframes")
                # Use guiding latent for multiple images (FL2V)
                return guiding_fn(
                    images=images,
                    height=height,
                    width=width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=device,
                )
            else:
                # Use original replacing latent for single image (I2V)
                return original_replacing(
                    images=images,
                    height=height,
                    width=width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=device,
                )
        
        # Monkey-patch in BOTH places to ensure it's applied
        helpers_module.image_conditionings_by_replacing_latent = smart_conditioning
        distilled_module.image_conditionings_by_replacing_latent = smart_conditioning
        
        print("   Conditioning function patched for FL2V support (helpers + distilled modules)!")
    
    def _warmup(self):
        """Run a single warmup to ensure CUDA kernels are ready."""
        import torch
        from ltx_core.model.video_vae import TilingConfig
        
        warmup_height, warmup_width, warmup_frames = 512, 768, 17
        tiling_config = TilingConfig.default()
        
        print(f"   Warmup: {warmup_height}x{warmup_width}, {warmup_frames} frames...")
        
        with torch.inference_mode():
            video_iter, audio = self.pipeline(
                prompt="warmup test",
                seed=42,
                height=warmup_height,
                width=warmup_width,
                num_frames=warmup_frames,
                frame_rate=30.0,
                images=[],
                tiling_config=tiling_config,
                enhance_prompt=False,
            )
            # Consume the iterator
            for _ in video_iter:
                pass
        
        torch.cuda.synchronize()
        print("   Warmup complete!")
    
    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        import torch
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"📊 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    def _generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list,
        output_name: str,
    ) -> bytes:
        """Internal generation method supporting both T2V and I2V."""
        import gc
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_pipelines.utils.media_io import encode_video

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        with torch.inference_mode():
            video_iter, audio = self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=False,
            )

            out_path = f"/outputs/{output_name}"
            encode_video(
                video=video_iter,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=out_path,
                video_chunks_number=video_chunks_number,
            )

        with open(out_path, "rb") as f:
            video_bytes = f.read()

        gc.collect()
        torch.cuda.empty_cache()
        
        return video_bytes

    @modal.method()
    def generate_t2v(
        self,
        prompt: str,
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 30.0,
        output_name: str = "t2v_output.mp4",
    ) -> bytes:
        """Generate video from text prompt (Text-to-Video)."""
        return self._generate(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            output_name=output_name,
        )

    @modal.method()
    def generate_i2v(
        self,
        image_b64: str,
        prompt: str,
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 30.0,
        output_name: str = "i2v_output.mp4",
    ) -> bytes:
        """Generate video from image + text prompt (Image-to-Video)."""
        import base64
        import io
        import tempfile
        from PIL import Image
        
        # Decode the base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize to target dimensions
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # LTX-2 expects file paths, not PIL Images
        # Save to temp file and pass the path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, "PNG")
            image_path = tmp.name
        
        # LTX-2 expects images as tuples: (image_path, frame_idx, strength)
        # frame_idx=0 means the image is the first frame
        # strength=1.0 means full conditioning strength
        images_with_config = [(image_path, 0, 1.0)]
        
        return self._generate(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images_with_config,
            output_name=output_name,
        )

    @modal.method()
    def generate_fl2v(
        self,
        first_image_b64: str,
        last_image_b64: str,
        prompt: str,
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 30.0,
        output_name: str = "fl2v_output.mp4",
    ) -> bytes:
        """Generate video from first + last frame images (First-Last-to-Video)."""
        import base64
        import io
        import tempfile
        from PIL import Image
        
        # Decode first image
        first_data = base64.b64decode(first_image_b64)
        first_image = Image.open(io.BytesIO(first_data)).convert("RGB")
        first_image = first_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Decode last image
        last_data = base64.b64decode(last_image_b64)
        last_image = Image.open(io.BytesIO(last_data)).convert("RGB")
        last_image = last_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Save to temp files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            first_image.save(tmp.name, "PNG")
            first_path = tmp.name
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            last_image.save(tmp.name, "PNG")
            last_path = tmp.name
        
        # LTX-2 expects images as tuples: (image_path, frame_idx, strength)
        # With our patched conditioning function, both images will be used
        # as guiding latents throughout the generation
        images_with_config = [
            (first_path, 0, 1.0),                 # First frame keyframe
            (last_path, num_frames - 1, 1.0),    # Last frame keyframe
        ]
        
        return self._generate(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images_with_config,
            output_name=output_name,
        )


# ============================================================================
# Web API
# ============================================================================

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

class T2VRequest(BaseModel):
    prompt: str
    width: int = 768
    height: int = 512
    num_frames: int = 97
    seed: int = 42


@app.function(timeout=900)
@modal.asgi_app()
def web():
    """FastAPI web endpoint with UI for T2V and I2V generation."""
    import base64
    
    web_app = FastAPI(title="LTX-2 Video Generation API")
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the web UI."""
        return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LTX-2 Video Generation</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1a 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff2daa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #888;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        .tabs {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .tab {
            padding: 0.75rem 1.5rem;
            border: 2px solid #333;
            border-radius: 12px;
            background: transparent;
            color: #888;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.2s;
        }
        .tab:hover { border-color: #555; color: #fff; }
        .tab.active {
            border-color: #7b2fff;
            background: rgba(123, 47, 255, 0.15);
            color: #fff;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
        }
        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
            color: #fff;
        }
        label {
            display: block;
            font-size: 0.875rem;
            color: #888;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        }
        label:first-of-type { margin-top: 0; }
        input[type="text"], input[type="number"], textarea, select {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid #333;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #7b2fff;
        }
        textarea { resize: vertical; min-height: 100px; }
        .row { display: flex; gap: 1rem; }
        .row > * { flex: 1; }
        .dropzone {
            border: 2px dashed #333;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 0.5rem;
        }
        .dropzone:hover { border-color: #7b2fff; background: rgba(123, 47, 255, 0.05); }
        .dropzone.dragover { border-color: #00d4ff; background: rgba(0, 212, 255, 0.1); }
        .dropzone img {
            max-width: 100%;
            max-height: 200px;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .dropzone p { color: #666; }
        input[type="file"] { display: none; }
        button.generate {
            width: 100%;
            padding: 1rem;
            margin-top: 1.5rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(90deg, #7b2fff, #ff2daa);
            color: #fff;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        button.generate:hover:not(:disabled) { transform: translateY(-2px); }
        button.generate:disabled { opacity: 0.6; cursor: not-allowed; }
        .status {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.3);
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            display: none;
        }
        .status.visible { display: block; }
        .status.error { border-left: 3px solid #ff4444; }
        .status.success { border-left: 3px solid #00ff88; }
        video {
            width: 100%;
            border-radius: 12px;
            background: #000;
            margin-top: 1rem;
        }
        .download {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            text-decoration: none;
            font-weight: 500;
            transition: background 0.2s;
        }
        .download:hover { background: rgba(0, 212, 255, 0.3); }
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LTX-2 Video Generation</h1>
        <p class="subtitle">19B parameter model • 8-step distilled • FP8 inference on H100</p>
        
        <div class="tabs">
            <button class="tab active" data-mode="t2v">Text to Video</button>
            <button class="tab" data-mode="i2v">Image to Video</button>
            <button class="tab" data-mode="fl2v">First + Last Frame</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Input</h2>
                
                <div id="image-input" class="hidden">
                    <label>Starting Image</label>
                    <div class="dropzone" id="dropzone">
                        <p>Drop image here or click to upload</p>
                        <img id="preview" class="hidden" />
                    </div>
                    <input type="file" id="file" accept="image/*" />
                </div>
                
                <div id="last-image-input" class="hidden">
                    <label>Ending Image</label>
                    <div class="dropzone" id="dropzone-last">
                        <p>Drop last frame image here</p>
                        <img id="preview-last" class="hidden" />
                    </div>
                    <input type="file" id="file-last" accept="image/*" />
                </div>
                
                <label>Prompt</label>
                <textarea id="prompt" placeholder="Describe the video you want to generate...">A majestic eagle soaring through a golden sunset sky, cinematic lighting, smooth motion</textarea>
                
                <div class="row">
                    <div>
                        <label>Width</label>
                        <input type="number" id="width" value="768" step="64" min="256" max="1920" />
                    </div>
                    <div>
                        <label>Height</label>
                        <input type="number" id="height" value="512" step="64" min="256" max="1080" />
                    </div>
                </div>
                
                <div class="row">
                    <div>
                        <label>Frames</label>
                        <input type="number" id="frames" value="97" step="1" min="17" max="300" />
                    </div>
                    <div>
                        <label>Seed</label>
                        <input type="number" id="seed" value="42" />
                    </div>
                </div>
                
                <button class="generate" id="generate">Generate Video</button>
                <div class="status" id="status"></div>
            </div>
            
            <div class="card">
                <h2>Output</h2>
                <video id="video" controls playsinline></video>
                <a id="download" class="download hidden" download="ltx2_output.mp4">Download MP4</a>
            </div>
        </div>
    </div>
    
    <script>
        const $ = id => document.getElementById(id);
        let mode = 't2v';
        let imageData = null;
        let lastImageData = null;
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                mode = tab.dataset.mode;
                $('image-input').classList.toggle('hidden', mode === 't2v');
                $('last-image-input').classList.toggle('hidden', mode !== 'fl2v');
            });
        });
        
        // First image drag and drop
        const dropzone = $('dropzone');
        const fileInput = $('file');
        const preview = $('preview');
        
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], 'first');
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0], 'first'); });
        
        // Last image drag and drop
        const dropzoneLast = $('dropzone-last');
        const fileInputLast = $('file-last');
        const previewLast = $('preview-last');
        
        dropzoneLast.addEventListener('click', () => fileInputLast.click());
        dropzoneLast.addEventListener('dragover', e => { e.preventDefault(); dropzoneLast.classList.add('dragover'); });
        dropzoneLast.addEventListener('dragleave', () => dropzoneLast.classList.remove('dragover'));
        dropzoneLast.addEventListener('drop', e => {
            e.preventDefault();
            dropzoneLast.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], 'last');
        });
        fileInputLast.addEventListener('change', () => { if (fileInputLast.files.length) handleFile(fileInputLast.files[0], 'last'); });
        
        function handleFile(file, which) {
            const reader = new FileReader();
            reader.onload = e => {
                const b64 = e.target.result.split(',')[1];
                if (which === 'first') {
                    imageData = b64;
                    preview.src = e.target.result;
                    preview.classList.remove('hidden');
                    dropzone.querySelector('p').textContent = file.name;
                } else {
                    lastImageData = b64;
                    previewLast.src = e.target.result;
                    previewLast.classList.remove('hidden');
                    dropzoneLast.querySelector('p').textContent = file.name;
                }
            };
            reader.readAsDataURL(file);
        }
        
        // Generate
        $('generate').addEventListener('click', async () => {
            const btn = $('generate');
            const status = $('status');
            const video = $('video');
            const dl = $('download');
            
            btn.disabled = true;
            status.className = 'status visible';
            status.textContent = 'Starting generation...';
            video.removeAttribute('src');
            dl.classList.add('hidden');
            
            const params = {
                prompt: $('prompt').value,
                width: parseInt($('width').value),
                height: parseInt($('height').value),
                num_frames: parseInt($('frames').value),
                seed: parseInt($('seed').value),
            };
            
            try {
                const t0 = performance.now();
                let resp;
                
                if (mode === 'fl2v') {
                    if (!imageData) throw new Error('Please upload a first frame image');
                    if (!lastImageData) throw new Error('Please upload a last frame image');
                    const fd = new FormData();
                    fd.append('first_image', await fetch(`data:image/png;base64,${imageData}`).then(r => r.blob()), 'first.png');
                    fd.append('last_image', await fetch(`data:image/png;base64,${lastImageData}`).then(r => r.blob()), 'last.png');
                    fd.append('prompt', params.prompt);
                    fd.append('width', params.width);
                    fd.append('height', params.height);
                    fd.append('num_frames', params.num_frames);
                    fd.append('seed', params.seed);
                    resp = await fetch('/api/fl2v', { method: 'POST', body: fd });
                } else if (mode === 'i2v') {
                    if (!imageData) throw new Error('Please upload an image first');
                    const fd = new FormData();
                    fd.append('image', await fetch(`data:image/png;base64,${imageData}`).then(r => r.blob()), 'image.png');
                    fd.append('prompt', params.prompt);
                    fd.append('width', params.width);
                    fd.append('height', params.height);
                    fd.append('num_frames', params.num_frames);
                    fd.append('seed', params.seed);
                    resp = await fetch('/api/i2v', { method: 'POST', body: fd });
                } else {
                    resp = await fetch('/api/t2v', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(params)
                    });
                }
                
                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }
                
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const dt = ((performance.now() - t0) / 1000).toFixed(1);
                
                video.src = url;
                dl.href = url;
                dl.classList.remove('hidden');
                
                status.className = 'status visible success';
                status.textContent = `Done in ${dt}s • ${params.num_frames} frames @ ${(params.num_frames / parseFloat(dt)).toFixed(1)} fps`;
            } catch (e) {
                status.className = 'status visible error';
                status.textContent = 'Error: ' + e.message;
            } finally {
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
        """)

    @web_app.post("/api/t2v")
    async def api_t2v(req: T2VRequest):
        """Text-to-Video API endpoint."""
        try:
            engine = OfficialLTX2Engine()
            video_bytes = engine.generate_t2v.remote(
                prompt=req.prompt,
                seed=req.seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                frame_rate=30.0,
            )
            return Response(content=video_bytes, media_type="video/mp4")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/api/i2v")
    async def api_i2v(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        width: int = Form(768),
        height: int = Form(512),
        num_frames: int = Form(97),
        seed: int = Form(42),
    ):
        """Image-to-Video API endpoint."""
        try:
            import base64
            image_data = await image.read()
            image_b64 = base64.b64encode(image_data).decode()
            
            engine = OfficialLTX2Engine()
            video_bytes = engine.generate_i2v.remote(
                image_b64=image_b64,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=30.0,
            )
            return Response(content=video_bytes, media_type="video/mp4")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/api/fl2v")
    async def api_fl2v(
        first_image: UploadFile = File(...),
        last_image: UploadFile = File(...),
        prompt: str = Form(...),
        width: int = Form(768),
        height: int = Form(512),
        num_frames: int = Form(97),
        seed: int = Form(42),
    ):
        """First+Last Frame to Video API endpoint."""
        try:
            import base64
            first_data = await first_image.read()
            last_data = await last_image.read()
            first_b64 = base64.b64encode(first_data).decode()
            last_b64 = base64.b64encode(last_data).decode()
            
            engine = OfficialLTX2Engine()
            video_bytes = engine.generate_fl2v.remote(
                first_image_b64=first_b64,
                last_image_b64=last_b64,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=30.0,
            )
            return Response(content=video_bytes, media_type="video/mp4")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "model": "LTX-2 19B DistilledPipeline (8-step, FP8)"}

    return web_app


# ============================================================================
# CLI
# ============================================================================

@app.local_entrypoint()
def main(
    download: bool = False,
    test: bool = False,
    prompt: str = "A majestic eagle soaring through a golden sunset sky",
    seed: int = 42,
    height: int = 512,
    width: int = 768,
    num_frames: int = 97,
    frame_rate: float = 30.0,
    output: str = "outputs/ltx2_output.mp4",
):
    if download:
        download_models.remote()
        return
    if test:
        print(f"📝 Prompt: {prompt}")
        print(f"🎬 Resolution: {height}x{width}, {num_frames} frames @ {frame_rate}fps")
        print(f"⚡ Pipeline: 8-step distilled + 2x upscaling")
        print(f"🎲 Seed: {seed}")
        print("")
        print("🚀 Starting generation on Modal...")
        
        engine = OfficialLTX2Engine()
        video_bytes = engine.generate_t2v.remote(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            output_name=Path(output).name,
        )
        
        local_path = Path(output)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(video_bytes)
        print(f"✅ Saved video to: {local_path.absolute()}")
        return
    
    print("Usage:")
    print("  modal run examples/ltx2/ltx2_official_modal.py --download")
    print("  modal run examples/ltx2/ltx2_official_modal.py --test --prompt '...'")
    print("")
    print("Deploy for web UI:")
    print("  modal deploy examples/ltx2/ltx2_official_modal.py")
    print("")
    print("Options:")
    print("  --height 512 --width 768 --num-frames 97 --seed 42")
    print("  --output outputs/ltx2_output.mp4")

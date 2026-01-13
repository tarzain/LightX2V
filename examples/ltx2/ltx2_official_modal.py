"""
Modal deployment using the **official Lightricks LTX-2 distilled pipeline** (not LightX2V).

This mirrors the reference implementation in:
`https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-pipelines/src/ltx_pipelines/distilled.py`
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
    # torchvision wheels require common image/GL libs; without them, torchvision C++ ops may fail to load
    # and you can see errors like `operator torchvision::nms does not exist`.
    .apt_install(
        "ffmpeg",
        "git",
        "build-essential",
        "libjpeg-turbo8",
        "libpng16-16",
        "libgl1",
    )
    .pip_install(
        # Core runtime
        "torch==2.5.1",
        # NOTE: do NOT install torchvision here. In this Modal base image it can load in a broken state
        # (e.g. failing to import `torchvision._C`), which then breaks Gemma3 imports inside transformers.
        # transformers will gracefully fall back to PIL-based vision utils if torchvision is absent.
        # LTX-2 deps
        "transformers>=4.50.0",
        "accelerate",
        "einops",
        "safetensors",
        "huggingface-hub",
        "pillow",
        "numpy",
        "scipy",
        "tqdm",
        # ltx-core imports torchaudio in a few audio utilities
        "torchaudio==2.5.1",
        # Official pipeline uses `ltx_pipelines.utils.media_io.encode_video` which depends on PyAV.
        "av",
        # NOTE: Not installing xformers or flash-attn. The ltx-core attention.py will fall back to
        # PyTorch SDPA (scaled_dot_product_attention) which is efficient and memory-friendly.
    )
    # Install official LTX-2 packages from source
    .run_commands(
        f"git clone https://github.com/Lightricks/LTX-2.git {LTX2_REPO_DIR}",
        # Important: prevent these editable installs from upgrading torch/torchvision.
        # We'll install any missing dependencies explicitly via `.pip_install(...)` instead.
        f"pip install -e {LTX2_REPO_DIR}/packages/ltx-core --no-deps",
        f"pip install -e {LTX2_REPO_DIR}/packages/ltx-pipelines --no-deps",
        # Sanity check: torch + transformers Gemma3 import works in the built image.
        "python -c \"import torch; print('torch', torch.__version__)\"",
        "python -c \"from transformers.models.gemma3 import Gemma3ForConditionalGeneration; print('Gemma3 import OK')\"",
    )
    .env(
        {
            "HF_HOME": f"{MODELS_DIR}/hf_cache",
            # Helps fragmentation a bit for large models
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    """
    Download the official LTX-2 weights into the shared model volume.

    Requirements:
    - `Lightricks/LTX-2` HF repo for the main weights, upsampler, and distilled LoRA.
    - A Gemma3 checkpoint directory for `--gemma-root` (must contain model*.safetensors, tokenizer.model,
      and preprocessor_config.json). Set `LTX2_GEMMA_REPO_ID` to auto-download it, otherwise place files
      under `/models/gemma` yourself.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(LTX2_MODELS_DIR, exist_ok=True)
    os.makedirs(GEMMA_DIR, exist_ok=True)

    # Main LTX-2 repo weights (includes distilled ckpt, LoRA, upscalers, etc.)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set. Ensure Modal secret `huggingface-secret` exports HF_TOKEN.")

    snapshot_download(
        "Lightricks/LTX-2",
        local_dir=LTX2_MODELS_DIR,
        local_dir_use_symlinks=False,
        token=hf_token,
    )

    # Optional Gemma3 download (may require auth depending on the repo)
    gemma_repo_id = os.environ.get("LTX2_GEMMA_REPO_ID", DEFAULT_GEMMA_REPO_ID).strip()
    if gemma_repo_id:
        snapshot_download(
            gemma_repo_id,
            local_dir=GEMMA_DIR,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print(f"✅ Downloaded Gemma model: {gemma_repo_id} -> {GEMMA_DIR}")
    else:
        print(
            "⚠️ Skipping Gemma download (LTX2_GEMMA_REPO_ID not set). "
            "To auto-download Gemma into /models/gemma, set LTX2_GEMMA_REPO_ID (and HF_TOKEN if needed)."
        )


@app.cls(
    gpu="H100",  # H100 80GB should be enough with proper memory management
    timeout=3600,
    volumes={MODELS_DIR: model_volume, "/outputs": outputs_volume},
)
class OfficialLTX2Engine:
    """
    Run the official LTX-2 DistilledPipeline.
    
    Note: The pipeline creates models on-demand and cleans them up between stages.
    We create the pipeline fresh for each generation to ensure clean memory state.
    """

    @modal.method()
    def generate_t2v(
        self,
        prompt: str,
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 30.0,
        output_name: str = "ltx2_official_distilled.mp4",
    ) -> bytes:
        """
        Generate video using DistilledPipeline (8-step distilled with 2x upscaling).
        Returns MP4 bytes for local saving.
        """
        import gc
        import torch
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_pipelines.utils.media_io import encode_video

        # Verify files exist
        ckpt = f"{LTX2_MODELS_DIR}/ltx-2-19b-distilled-fp8.safetensors"
        spatial_upsampler = f"{LTX2_MODELS_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"

        missing = [p for p in (ckpt, spatial_upsampler) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing required LTX-2 files under /models. Run `--download` first.\n"
                + "\n".join(f"- {p}" for p in missing)
            )

        if not (Path(GEMMA_DIR).exists() and list(Path(GEMMA_DIR).rglob("model*.safetensors"))):
            raise FileNotFoundError(
                f"Missing Gemma model files under `{GEMMA_DIR}`. "
                "Set `LTX2_GEMMA_REPO_ID` and rerun `--download`, or populate /models/gemma manually."
            )

        # Create pipeline fresh for each call (ensures clean memory state)
        pipeline = DistilledPipeline(
            checkpoint_path=ckpt,
            spatial_upsampler_path=spatial_upsampler,
            gemma_root=GEMMA_DIR,
            loras=[],
            fp8transformer=True,
        )

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        # CRITICAL: Must use inference_mode() to prevent gradient tracking,
        # which would keep computation graphs (and Gemma) alive in memory!
        with torch.inference_mode():
            video_iter, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[],
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

        # Read video bytes to return to caller
        with open(out_path, "rb") as f:
            video_bytes = f.read()

        # Cleanup
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        return video_bytes


@app.local_entrypoint()
def main(
    download: bool = False,
    test: bool = False,
    prompt: str = "A majestic eagle soaring through a golden sunset sky",
    seed: int = 42,
    # Default to official recommended resolution for DistilledPipeline
    height: int = 704,
    width: int = 1216,
    num_frames: int = 97,
    frame_rate: float = 30.0,
    output: str = "outputs/ltx2_output.mp4",
):
    if download:
        download_models.remote()
        return
    if test:
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
        
        # Save video locally
        local_path = Path(output)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(video_bytes)
        print(f"✅ Saved video to: {local_path.absolute()}")
        return
    
    print("Usage:")
    print("  modal run examples/ltx2/ltx2_official_modal.py --download")
    print("  modal run examples/ltx2/ltx2_official_modal.py --test --prompt '...'")
    print("")
    print("Options (defaults shown):")
    print("  --height 704 --width 1216 --num-frames 97 --frame-rate 30 --seed 42")
    print("  --output outputs/ltx2_output.mp4")
    print("")
    print("Pipeline: DistilledPipeline (8-step distilled + 2x spatial upscaling)")



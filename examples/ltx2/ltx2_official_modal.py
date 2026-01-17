"""
Modal deployment using the **official Lightricks LTX-2 distilled pipeline** (not LightX2V).

Features:
- GPU memory snapshotting for fast cold starts
- FP8 transformer for reduced memory footprint
- 8-step distilled inference with 2x spatial upscaling
- Text-to-video (T2V) and Image-to-video (I2V) support
- Audio-to-video (A2V) conditioning support
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

# Pipeline configuration
USE_FP8 = False  # False = BF16 checkpoint (~38GB), True = FP8 checkpoint (~19GB)
SKIP_UPSCALING = True  # True = generate at full res, skip stage 2 (faster), False = 2-stage with upscaling

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

# Global state for conditioning (using a dict for reliable access)
_CONDITIONING_STATE = {"audio_latent": None}


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
    gpu="H200",
    timeout=3600,
    scaledown_window=300,  # Keep warm for 5 minutes
    enable_memory_snapshot=False,  # Disabled for debugging
    # experimental_options={"enable_gpu_snapshot": True},
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

    @modal.enter()  # snap=True disabled for debugging
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
        
        # Store configuration
        self.use_fp8 = USE_FP8
        self.skip_upscaling = SKIP_UPSCALING
        
        # Select checkpoint based on configuration
        if USE_FP8:
            ckpt = f"{LTX2_MODELS_DIR}/ltx-2-19b-distilled-fp8.safetensors"
            print(f"🔧 Loading LTX-2 DistilledPipeline with FP8 (~19GB)...")
        else:
            ckpt = f"{LTX2_MODELS_DIR}/ltx-2-19b-distilled.safetensors"
            print(f"🔧 Loading LTX-2 DistilledPipeline with BF16 (~38GB)...")
        
        if SKIP_UPSCALING:
            print(f"   Audio mode: Will skip stage 2 upscaling (skip_upscaling=True)")
        else:
            print(f"   Audio mode: Full two-stage with upscaling")
        
        spatial_upsampler = f"{LTX2_MODELS_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        
        # Check for required files (always require upsampler for runtime toggling)
        required_files = [ckpt, spatial_upsampler]
        
        missing = [p for p in required_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing required LTX-2 files. Run `--download` first.\n"
                + "\n".join(f"- {p}" for p in missing)
            )
        
        if not (Path(GEMMA_DIR).exists() and list(Path(GEMMA_DIR).rglob("model*.safetensors"))):
            raise FileNotFoundError(f"Missing Gemma model files under `{GEMMA_DIR}`.")
        
        # Create the distilled pipeline (always load upsampler for runtime toggling)
        from ltx_pipelines.distilled import DistilledPipeline
        self.pipeline = DistilledPipeline(
            checkpoint_path=ckpt,
            spatial_upsampler_path=spatial_upsampler,
            gemma_root=GEMMA_DIR,
            loras=[],
            fp8transformer=USE_FP8,
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
        
        # Always load spatial upsampler (for runtime toggling)
        print("   Loading spatial upsampler...")
        self._spatial_upsampler = ledger.spatial_upsampler()
        
        # Load audio components for A2V
        print("   Loading audio encoder...")
        self._audio_encoder = self._build_audio_encoder(ledger)
        
        print("   Loading audio decoder...")
        self._audio_decoder = ledger.audio_decoder()
        
        print("   Loading vocoder...")
        self._vocoder = ledger.vocoder()
        
        torch.cuda.synchronize()
        print("   All models loaded!")
        
        # Apply patches after all models are loaded
        print("   Patching ModelLedger to use cached models...")
        self._patch_model_ledger(ledger)
    
    def _build_audio_encoder(self, ledger):
        """Build the audio encoder (not exposed in ModelLedger by default)."""
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
        from ltx_core.model.audio_vae import (
            AudioEncoderConfigurator,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        
        audio_encoder_builder = Builder(
            model_path=ledger.checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=ledger.registry,
        )
        
        return audio_encoder_builder.build(
            device=ledger.device,
            dtype=ledger.dtype
        ).to(ledger.device).eval()
    
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
        cached_audio_encoder = self._audio_encoder
        cached_audio_decoder = self._audio_decoder
        cached_vocoder = self._vocoder
        
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
        
        def patched_audio_encoder():
            return cached_audio_encoder
        
        def patched_audio_decoder():
            return cached_audio_decoder
        
        def patched_vocoder():
            return cached_vocoder
        
        # Apply patches
        ledger.text_encoder = patched_text_encoder
        ledger.transformer = patched_transformer
        ledger.video_encoder = patched_video_encoder
        ledger.video_decoder = patched_video_decoder
        ledger.spatial_upsampler = patched_spatial_upsampler
        ledger.audio_encoder = patched_audio_encoder
        ledger.audio_decoder = patched_audio_decoder
        ledger.vocoder = patched_vocoder
        
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
        
        # Patch pipeline for audio conditioning support
        self._patch_pipeline_for_audio()
    
    def _patch_pipeline_for_audio(self):
        """
        Audio conditioning is now handled directly in _generate_with_audio.
        This method is kept for compatibility but doesn't patch anything.
        """
        print("   Audio conditioning: Using direct injection method")
    
    def _encode_audio(self, audio_path: str, target_duration_seconds: float):
        """
        Load and encode audio to latent representation.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            target_duration_seconds: Target duration in seconds
            
        Returns:
            Audio latent tensor for conditioning
        """
        import torch
        import torchaudio
        from ltx_core.model.audio_vae.ops import AudioProcessor
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        
        # Audio encoder parameters (from AudioEncoderConfigurator defaults)
        sample_rate = 16000
        mel_hop_length = 160
        n_fft = 1024
        mel_bins = 64
        
        # Create audio processor
        audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            mel_bins=mel_bins,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft,
        ).to(self.pipeline.device)
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # AudioEncoder expects stereo (2 channels)
        if waveform.shape[0] == 1:
            # Duplicate mono to stereo
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            # Take first 2 channels if more than stereo
            waveform = waveform[:2]
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Trim or pad to target duration
        target_samples = int(target_duration_seconds * sample_rate)
        num_channels = waveform.shape[0]  # Should be 2 (stereo)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            padding = torch.zeros(num_channels, target_samples - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        # Add batch dimension: [1, channels, samples]
        waveform = waveform.unsqueeze(0).to(self.pipeline.device, dtype=torch.float32)
        
        # Convert to mel spectrogram
        mel = audio_processor.waveform_to_mel(waveform, sample_rate)
        
        # Encode to latent
        with torch.no_grad():
            audio_latent = self._audio_encoder(mel.to(self.pipeline.dtype))
        
        print(f"   Audio encoded: {audio_path} -> latent shape {audio_latent.shape}")
        return audio_latent

    def _warmup(self):
        """Run a single warmup to ensure CUDA kernels are ready."""
        import torch
        from ltx_core.model.video_vae import TilingConfig
        
        warmup_height, warmup_width, warmup_frames = 512, 768, 17
        tiling_config = TilingConfig.default()
        
        print(f"   Warmup: {warmup_height}x{warmup_width}, {warmup_frames} frames...")
        
        with torch.inference_mode():
            if self.skip_upscaling:
                # Use skip_upscaling path for warmup
                video_iter, audio = self._generate_skip_upscaling(
                    prompt="warmup test",
                    seed=42,
                    height=warmup_height,
                    width=warmup_width,
                    num_frames=warmup_frames,
                    frame_rate=30.0,
                    images=[],
                )
            else:
                # Full 2-stage pipeline warmup
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
        skip_upscaling: bool = True,
    ) -> bytes:
        """Internal generation method using DistilledPipeline."""
        import gc
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_pipelines.utils.media_io import encode_video

        tiling_config = TilingConfig.default()

        with torch.inference_mode():
            if skip_upscaling:
                # Custom logic: Generate at full resolution in stage 1, skip stage 2
                video_iter, audio = self._generate_skip_upscaling(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                )
                video_chunks_number = 1
            else:
                # Normal 2-stage pipeline
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
                video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

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
    
    def _generate_skip_upscaling(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list,
    ):
        """
        Generate video at full resolution using only stage 1 (skip upscaling and stage 2).
        This is faster but produces slightly lower quality output.
        """
        import torch
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils import helpers as ltx_helpers  # Use module to get patched FL2V function
        from ltx_pipelines.utils.helpers import (
            euler_denoising_loop,
            noise_video_state,
            noise_audio_state,
            simple_denoising_func,
        )

        device = self.pipeline.device
        dtype = torch.bfloat16
        
        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        
        # Use cached models
        text_encoder = self._text_encoder
        video_encoder = self._video_encoder
        video_decoder = self._video_decoder
        transformer = self._transformer
        
        # Encode text
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p
        
        # Stage 1 sigmas (only 8 steps!)
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
        
        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )
        
        print(f"   Skip-upscale: Generating at full resolution {width}x{height} (8 steps only)")
        
        output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )
        
        # Image conditioning (use module reference to get patched FL2V function)
        if len(images) > 0:
            conditionings = ltx_helpers.image_conditionings_by_replacing_latent(
                images=images,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=device,
            )
        else:
            conditionings = ()
        
        # Initialize video state from noise
        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=conditionings,
            components=self.pipeline.pipeline_components,
            dtype=dtype,
            device=device,
            noise_scale=1.0,
            initial_latent=None,
        )
        
        # Initialize empty audio state (no audio input)
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=[],
            components=self.pipeline.pipeline_components,
            dtype=dtype,
            device=device,
            noise_scale=1.0,
            initial_latent=None,
        )
        
        # Run stage 1 denoising only
        video_state, audio_state = denoising_loop(
            stage_1_sigmas,
            video_state,
            audio_state,
            stepper,
        )
        
        # Clear conditioning and unpatchify
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        
        # Decode video directly (skip upscaling and stage 2!)
        print("   Skip-upscale: Decoding video and audio (skipping upscale and stage 2)")
        video_iterator = vae_decode_video(
            video_decoder=video_decoder,
            latent=video_state.latent[:1],
        )
        
        # Also decode audio (LTX-2 generates both simultaneously)
        audio_decoder = self._audio_decoder
        vocoder = self._vocoder
        audio = vae_decode_audio(
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            latent=audio_state.latent[:1],
        )
        
        return video_iterator, audio

    def _generate_with_audio(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list,
        audio_latent,
        output_name: str,
        audio_conditioning_strength: float = 0.3,
        skip_upscaling: bool = True,
    ) -> bytes:
        """
        Generation with audio conditioning.
        
        This method directly calls the internal pipeline components to inject
        the audio latent into the denoising process.
        """
        import gc
        import torch
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import (
            AUDIO_SAMPLE_RATE,
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )
        from ltx_pipelines.utils import helpers as ltx_helpers  # Use module to get patched FL2V function
        from ltx_pipelines.utils.helpers import (
            denoise_audio_video,
            euler_denoising_loop,
            noise_video_state,
            noise_audio_state,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import encode_video

        print(f"   A2V: Generating with audio conditioning, latent shape: {audio_latent.shape}")
        print(f"   A2V: Audio conditioning strength: {audio_conditioning_strength} (0.0=preserve, 1.0=full diffusion)")
        
        tiling_config = TilingConfig.default()
        # When skipping upscaling, use 1 chunk (no tiling needed at lower res)
        if skip_upscaling:
            video_chunks_number = 1
        else:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        with torch.inference_mode():
            device = self.pipeline.device
            dtype = torch.bfloat16
            
            generator = torch.Generator(device=device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            stepper = EulerDiffusionStep()
            
            # Use cached models (preloaded during snapshot)
            text_encoder = self._text_encoder
            video_encoder = self._video_encoder
            transformer = self._transformer
            
            # Encode text
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            video_context, audio_context = context_p
            
            # Stage 1: Generate at half resolution with audio conditioning
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
            
            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=transformer,
                    ),
                )
            
            # If skip_upscaling, generate at full resolution in stage 1
            if skip_upscaling:
                stage_1_width, stage_1_height = width, height
                print(f"   A2V: Skipping upscaling - generating at full resolution {width}x{height}")
            else:
                stage_1_width, stage_1_height = width // 2, height // 2
                print(f"   A2V: Stage 1 at {stage_1_width}x{stage_1_height}, will upscale to {width}x{height}")
            
            stage_1_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=stage_1_width,
                height=stage_1_height,
                fps=frame_rate,
            )
            
            # Image conditioning (use module reference to get patched FL2V function)
            if len(images) > 0:
                stage_1_conditionings = ltx_helpers.image_conditionings_by_replacing_latent(
                    images=images,
                    height=stage_1_height,
                    width=stage_1_width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=device,
                )
            else:
                stage_1_conditionings = ()  # Empty tuple for no conditioning
            
            # Compute expected audio latent shape and resize if needed
            from ltx_core.types import AudioLatentShape
            expected_audio_shape = AudioLatentShape.from_video_pixel_shape(stage_1_output_shape)
            expected_frames = expected_audio_shape.frames
            actual_frames = audio_latent.shape[2]
            
            if actual_frames != expected_frames:
                print(f"   A2V: Resizing audio latent from {actual_frames} to {expected_frames} frames")
                # Audio latent shape is [B, C, T, H] = [1, 8, 251, 16]
                # We need to resize T (dimension 2) from 251 to 250
                # interpolate works on last 2 dims, so we need to reshape
                B, C, T, H = audio_latent.shape
                # Reshape to [B*C, 1, T, H] for 2D interpolation
                audio_flat = audio_latent.reshape(B * C, 1, T, H)
                audio_resized = torch.nn.functional.interpolate(
                    audio_flat,
                    size=(expected_frames, H),
                    mode='bilinear',
                    align_corners=False,
                )
                audio_latent_resized = audio_resized.reshape(B, C, expected_frames, H)
            else:
                audio_latent_resized = audio_latent
            
            # Stage 1 denoising WITH audio latent
            # Use different noise scales for video and audio:
            # - Video: noise_scale=1.0 (generated from scratch, full noise)
            # - Audio: noise_scale=audio_conditioning_strength (0.0=preserve, 1.0=full diffusion)
            # Lower values preserve more of the original audio but may have weaker conditioning.
            # Higher values allow stronger audio-video synchronization but may introduce artifacts.
            print(f"   A2V: Stage 1 - Initializing video (noise_scale=1.0) and audio (noise_scale={audio_conditioning_strength})...")
            
            # Initialize video state with full noise (generated from scratch)
            video_state, video_tools = noise_video_state(
                output_shape=stage_1_output_shape,
                noiser=noiser,
                conditionings=stage_1_conditionings,
                components=self.pipeline.pipeline_components,
                dtype=dtype,
                device=device,
                noise_scale=1.0,  # Full noise for video
                initial_latent=None,  # Start from pure noise
            )
            
            # Initialize audio state with configurable noise level
            audio_state, audio_tools = noise_audio_state(
                output_shape=stage_1_output_shape,
                noiser=noiser,
                conditionings=[],  # No audio conditionings
                components=self.pipeline.pipeline_components,
                dtype=dtype,
                device=device,
                noise_scale=audio_conditioning_strength,  # Configurable: 0.0=preserve, 1.0=full noise
                initial_latent=audio_latent_resized,  # Our encoded audio
            )
            
            print(f"   A2V: Stage 1 - Running denoising loop with preserved audio...")
            # Run the denoising loop
            video_state, audio_state = denoising_loop(
                stage_1_sigmas,
                video_state,
                audio_state,
                stepper,
            )
            
            # Clear conditioning and unpatchify (same as denoise_audio_video does)
            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            
            # Stage 2: Upsample and refine (skip if skip_upscaling is True)
            if not skip_upscaling:
                print("   A2V: Stage 2 upsampling...")
                spatial_upsampler = self._spatial_upsampler
                stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
                
                stage_2_output_shape = VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width,
                    height=height,
                    fps=frame_rate,
                )
                
                # Upsample video latent
                upsampled_video = upsample_video(
                    latent=video_state.latent[:1],
                    video_encoder=video_encoder,
                    upsampler=spatial_upsampler,
                )
                
                if len(images) > 0:
                    stage_2_conditionings = ltx_helpers.image_conditionings_by_replacing_latent(
                        images=images,
                        height=height,
                        width=width,
                        video_encoder=video_encoder,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    stage_2_conditionings = ()  # Empty tuple for no conditioning
                
                # Stage 2: Continue with consistent noise scales
                stage_2_audio_noise = audio_conditioning_strength * float(stage_2_sigmas[0].item())
                print(f"   A2V: Stage 2 - Initializing upsampled video and audio (noise_scale={stage_2_audio_noise:.3f})...")
                
                video_state_2, video_tools_2 = noise_video_state(
                    output_shape=stage_2_output_shape,
                    noiser=noiser,
                    conditionings=stage_2_conditionings,
                    components=self.pipeline.pipeline_components,
                    dtype=dtype,
                    device=device,
                    noise_scale=float(stage_2_sigmas[0].item()),
                    initial_latent=upsampled_video,
                )
                
                audio_state_2, audio_tools_2 = noise_audio_state(
                    output_shape=stage_2_output_shape,
                    noiser=noiser,
                    conditionings=[],
                    components=self.pipeline.pipeline_components,
                    dtype=dtype,
                    device=device,
                    noise_scale=stage_2_audio_noise,
                    initial_latent=audio_state.latent,
                )
                
                print(f"   A2V: Stage 2 - Running refinement denoising loop...")
                video_state, audio_state = denoising_loop(
                    stage_2_sigmas,
                    video_state_2,
                    audio_state_2,
                    stepper,
                )
                
                video_state = video_tools_2.clear_conditioning(video_state)
                video_state = video_tools_2.unpatchify(video_state)
                audio_state = audio_tools_2.clear_conditioning(audio_state)
                audio_state = audio_tools_2.unpatchify(audio_state)
            else:
                print("   A2V: Skipping Stage 2 (skip_upscaling=True)")
            
            # Decode video (returns a generator yielding frame batches)
            print("   A2V: Decoding video...")
            video_generator = vae_decode_video(
                video_state.latent,
                self._video_decoder,
                tiling_config=tiling_config,
            )
            
            # Decode audio based on conditioning strength
            # - strength=0: Use original audio (no artifacts, weaker conditioning)
            # - strength>0: Blend original and denoised for balance
            if audio_conditioning_strength < 0.1:
                # Very low strength: use original to avoid artifacts
                print("   A2V: Decoding original input audio (strength < 0.1)...")
                final_audio_latent = audio_latent_resized
            else:
                # Blend original and denoised audio latents for balance
                # Higher strength = more denoised (better sync but more artifacts)
                blend_factor = min(audio_conditioning_strength, 0.7)  # Cap at 0.7 to avoid severe artifacts
                print(f"   A2V: Blending audio latents (original:{1-blend_factor:.1f} + denoised:{blend_factor:.1f})...")
                final_audio_latent = (1 - blend_factor) * audio_latent_resized + blend_factor * audio_state.latent
            
            decoded_audio = vae_decode_audio(
                final_audio_latent,
                self._audio_decoder,
                self._vocoder,
            )
            
            # Use official encode_video function which handles color conversion correctly
            out_path = f"/outputs/{output_name}"
            
            print("   A2V: Encoding video with audio...")
            # encode_video expects:
            # - video: generator yielding frame tensors
            # - audio: tensor [C, samples]
            # - output_path: str
            # - fps: float
            # - audio_sample_rate: int
            encode_video(
                video=video_generator,
                audio=decoded_audio.squeeze(0),  # Remove batch dim
                output_path=out_path,
                fps=frame_rate,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                video_chunks_number=video_chunks_number,
            )
        
        # Read the encoded video file
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print("   A2V: Generation complete!")
        return video_bytes

    @modal.method()
    def generate(
        self,
        prompt: str,
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 30.0,
        output_name: str = "output.mp4",
        first_frame_b64: str | None = None,
        last_frame_b64: str | None = None,
        audio_b64: str | None = None,
        audio_conditioning_strength: float = 0.3,
        skip_upscaling: bool | None = None,
        extend_video_b64: str | None = None,
    ) -> bytes:
        """
        Unified generation method supporting all conditioning combinations:
        - Text only (T2V)
        - Text + first frame (I2V)
        - Text + first + last frame (FL2V)
        - Text + audio (A2V)
        - Text + first frame + audio (I2V+A2V)
        - Video extension (extend existing video)
        - Any combination!
        """
        import base64
        import io
        import tempfile
        from PIL import Image
        
        images = []
        extend_video_path = None
        
        # Process video to extend if provided
        if extend_video_b64:
            video_data = base64.b64decode(extend_video_b64)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_data)
                extend_video_path = tmp.name
        
        # Process first frame if provided
        if first_frame_b64:
            first_data = base64.b64decode(first_frame_b64)
            first_image = Image.open(io.BytesIO(first_data)).convert("RGB")
            first_image = first_image.resize((width, height), Image.Resampling.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                first_image.save(tmp.name, "PNG")
                first_path = tmp.name
            
            images.append((first_path, 0, 1.0))
        
        # Process last frame if provided
        if last_frame_b64:
            last_data = base64.b64decode(last_frame_b64)
            last_image = Image.open(io.BytesIO(last_data)).convert("RGB")
            last_image = last_image.resize((width, height), Image.Resampling.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                last_image.save(tmp.name, "PNG")
                last_path = tmp.name
            
            images.append((last_path, num_frames - 1, 1.0))
        
        # Use default skip_upscaling if not specified
        do_skip = skip_upscaling if skip_upscaling is not None else self.skip_upscaling
        
        # If video extension is requested
        if extend_video_path:
            return self._generate_video_extension(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                extend_video_path=extend_video_path,
                output_name=output_name,
                skip_upscaling=do_skip,
            )
        
        # If audio is provided, use audio-conditioned generation
        if audio_b64:
            audio_data = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                audio_path = tmp.name
            
            video_duration = num_frames / frame_rate
            audio_latent = self._encode_audio(audio_path, video_duration)
            
            return self._generate_with_audio(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                audio_latent=audio_latent,
                output_name=output_name,
                audio_conditioning_strength=audio_conditioning_strength,
                skip_upscaling=do_skip,
            )
        else:
            # Standard generation without audio
            return self._generate(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                output_name=output_name,
                skip_upscaling=do_skip,
            )

    def _generate_video_extension(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        extend_video_path: str,
        output_name: str,
        skip_upscaling: bool = True,
    ) -> bytes:
        """
        Extend an existing video by conditioning on its frames.
        
        Uses VideoConditionByKeyframeIndex to condition generation on the
        encoded latents from the input video, then generates new frames
        that continue the sequence.
        """
        import gc
        import torch
        import av
        import numpy as np
        from PIL import Image
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByKeyframeIndex
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape, AudioLatentShape
        import torchaudio
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, AUDIO_SAMPLE_RATE
        from ltx_pipelines.utils import helpers as ltx_helpers
        from ltx_pipelines.utils.helpers import (
            euler_denoising_loop,
            noise_video_state,
            noise_audio_state,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import encode_video
        
        print(f"   Video Extension: Loading input video from {extend_video_path}")
        
        # Load video frames and audio
        container = av.open(extend_video_path)
        video_stream = container.streams.video[0]
        
        # Check if video has audio
        has_audio = len(container.streams.audio) > 0
        audio_samples = []
        input_sample_rate = None
        
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB")
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            frames.append(np.array(img))
        
        # Extract audio if present
        if has_audio:
            container.seek(0)  # Reset to beginning
            audio_stream = container.streams.audio[0]
            input_sample_rate = audio_stream.rate
            for frame in container.decode(audio=0):
                audio_samples.append(frame.to_ndarray())
        
        container.close()
        
        input_num_frames = len(frames)
        print(f"   Video Extension: Loaded {input_num_frames} frames from input video")
        if has_audio:
            print(f"   Video Extension: Found audio track at {input_sample_rate}Hz")
        
        # Convert frames to tensor [B, C, T, H, W] format
        # frames is list of [H, W, C] numpy arrays
        frames_array = np.stack(frames, axis=0)  # [T, H, W, C]
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames_tensor = frames_tensor.unsqueeze(0)  # [1, T, C, H, W]
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        frames_tensor = frames_tensor.float() / 127.5 - 1.0  # Normalize to [-1, 1]
        
        device = self.pipeline.device
        dtype = torch.bfloat16
        frames_tensor = frames_tensor.to(device=device, dtype=dtype)
        
        print(f"   Video Extension: Encoding input video to latent space...")
        
        # Encode the input video to latents using VAE
        video_encoder = self._video_encoder
        video_decoder = self._video_decoder
        text_encoder = self._text_encoder
        transformer = self._transformer
        
        with torch.inference_mode():
            # Encode video frames to latent using the VAE encoder directly
            # VideoEncoder is called directly: encoder(video) -> latent
            video_conditioning_latent = video_encoder(frames_tensor)
            
            print(f"   Video Extension: Video conditioning latent shape: {video_conditioning_latent.shape}")
            
            # Encode audio if present
            audio_conditioning_latent = None
            if has_audio and len(audio_samples) > 0:
                print(f"   Video Extension: Encoding input audio to latent space...")
                
                # Concatenate audio samples
                audio_array = np.concatenate(audio_samples, axis=1)  # [channels, samples]
                audio_waveform = torch.from_numpy(audio_array).float()
                
                # Ensure stereo (2 channels)
                if audio_waveform.shape[0] == 1:
                    audio_waveform = audio_waveform.repeat(2, 1)
                elif audio_waveform.shape[0] > 2:
                    audio_waveform = audio_waveform[:2]
                
                # Resample to expected sample rate if needed
                if input_sample_rate != AUDIO_SAMPLE_RATE:
                    audio_waveform = torchaudio.functional.resample(
                        audio_waveform, input_sample_rate, AUDIO_SAMPLE_RATE
                    )
                
                # Convert to mel spectrogram
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=AUDIO_SAMPLE_RATE,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=128,
                ).to(device)
                
                audio_waveform = audio_waveform.to(device)
                mel_spec = mel_transform(audio_waveform)  # [2, 128, T]
                mel_spec = mel_spec.unsqueeze(0)  # [1, 2, 128, T]
                mel_spec = mel_spec.to(dtype)
                
                # Encode to latent using the audio encoder directly
                audio_encoder = self._audio_encoder
                audio_conditioning_latent = audio_encoder(mel_spec)
                print(f"   Video Extension: Audio conditioning latent shape: {audio_conditioning_latent.shape}")
            
            # Set up generation
            generator = torch.Generator(device=device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            stepper = EulerDiffusionStep()
            
            # Encode text
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            video_context, audio_context = context_p
            
            # Sigmas for denoising
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
            
            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=transformer,
                    ),
                )
            
            # Total frames = conditioning frames + new frames
            total_frames = input_num_frames + num_frames
            print(f"   Video Extension: Generating {num_frames} new frames (total: {total_frames})")
            
            output_shape = VideoPixelShape(
                batch=1,
                frames=total_frames,
                width=width,
                height=height,
                fps=frame_rate,
            )
            
            # Create conditioning from the input video latent
            # frame_idx=0 means the conditioning starts at frame 0
            # strength=1.0 means denoise_mask=0 (don't denoise these frames, keep them frozen)
            video_conditioning = VideoConditionByKeyframeIndex(
                keyframes=video_conditioning_latent,
                frame_idx=0,
                strength=1.0,  # Keep conditioning frames frozen
            )
            
            # Initialize video state with the conditioning
            video_state, video_tools = noise_video_state(
                output_shape=output_shape,
                noiser=noiser,
                conditionings=[video_conditioning],
                components=self.pipeline.pipeline_components,
                dtype=dtype,
                device=device,
                noise_scale=1.0,
                initial_latent=None,
            )
            
            # Handle audio from input video
            audio_initial_latent = None
            audio_noise_scale = 1.0  # Default: generate fresh audio
            
            if audio_conditioning_latent is not None:
                # Resize audio latent to match total output frames
                expected_audio_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
                expected_frames = expected_audio_shape.frames
                actual_frames = audio_conditioning_latent.shape[2]
                
                print(f"   Video Extension: Audio latent frames: {actual_frames}, expected: {expected_frames}")
                
                if actual_frames != expected_frames:
                    # Resize audio latent to match output shape
                    B, C, T, H = audio_conditioning_latent.shape
                    audio_flat = audio_conditioning_latent.reshape(B * C, 1, T, H)
                    audio_resized = torch.nn.functional.interpolate(
                        audio_flat,
                        size=(expected_frames, H),
                        mode='bilinear',
                        align_corners=False,
                    )
                    audio_conditioning_latent = audio_resized.reshape(B, C, expected_frames, H)
                
                audio_initial_latent = audio_conditioning_latent
                # Use low noise scale to preserve most of input audio while allowing some adaptation
                audio_noise_scale = 0.3  # Preserve 70% of original audio
                print(f"   Video Extension: Using input audio as conditioning (noise_scale={audio_noise_scale})")
            
            # Initialize audio state
            audio_state, audio_tools = noise_audio_state(
                output_shape=output_shape,
                noiser=noiser,
                conditionings=[],
                components=self.pipeline.pipeline_components,
                dtype=dtype,
                device=device,
                noise_scale=audio_noise_scale,
                initial_latent=audio_initial_latent,
            )
            
            print(f"   Video Extension: Running denoising loop...")
            
            # Run denoising
            video_state, audio_state = denoising_loop(
                stage_1_sigmas,
                video_state,
                audio_state,
                stepper,
            )
            
            # Clear conditioning and unpatchify
            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            
            print(f"   Video Extension: Decoding video and audio...")
            
            # Decode video
            video_iterator = vae_decode_video(
                video_decoder=video_decoder,
                latent=video_state.latent[:1],
            )
            
            # Decode audio
            audio_decoder = self._audio_decoder
            vocoder = self._vocoder
            audio = vae_decode_audio(
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                latent=audio_state.latent[:1],
            )
            
            # Encode output video
            out_path = f"/outputs/{output_name}"
            encode_video(
                video=video_iterator,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=out_path,
                video_chunks_number=1,
            )
        
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"   Video Extension: Complete! Output: {out_path}")
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
        """Generate video from text prompt (Text-to-Video). [DEPRECATED: Use generate() instead]"""
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

    # A2V method - version 3 with configurable audio conditioning strength
    @modal.method()
    def generate_a2v(
        self,
        audio_b64: str,
        prompt: str,
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 30.0,
        output_name: str = "a2v_output.mp4",
        image_b64: str | None = None,
        audio_conditioning_strength: float = 0.3,  # 0.0 = preserve audio exactly, 1.0 = full diffusion
    ) -> bytes:
        """
        Generate video conditioned on audio (Audio-to-Video).
        
        The audio latent is used to condition the video generation through
        the bidirectional audio-video cross-attention in the transformer.
        
        Args:
            audio_b64: Base64-encoded audio file (WAV, MP3, etc.)
            prompt: Text prompt describing the video
            seed: Random seed
            height: Video height
            width: Video width
            num_frames: Number of frames
            frame_rate: Frame rate
            output_name: Output filename
            image_b64: Optional base64 image for I2V+A2V combined conditioning
        """
        import base64
        import io
        import tempfile
        from PIL import Image
        
        # Decode audio and save to temp file
        audio_data = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            audio_path = tmp.name
        
        # Calculate video duration
        video_duration = num_frames / frame_rate
        
        # Encode audio to latent
        audio_latent = self._encode_audio(audio_path, video_duration)
        
        # Handle optional image conditioning
        images = []
        if image_b64:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, "PNG")
                image_path = tmp.name
            
            images = [(image_path, 0, 1.0)]
        
        # Use custom generation path that passes audio latent directly
        return self._generate_with_audio(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_latent=audio_latent,
            output_name=output_name,
            audio_conditioning_strength=audio_conditioning_strength,
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
    """FastAPI web endpoint with unified UI for video generation."""
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
        """Serve the unified web UI."""
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
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff2daa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle { color: #888; margin-bottom: 2rem; font-size: 1.1rem; }
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
        .card h2 { font-size: 1.25rem; margin-bottom: 1rem; color: #fff; }
        .card h3 { font-size: 1rem; margin: 1.5rem 0 0.75rem; color: #7b2fff; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
        label { display: block; font-size: 0.875rem; color: #888; margin-bottom: 0.5rem; margin-top: 1rem; }
        label:first-of-type { margin-top: 0; }
        input[type="text"], input[type="number"], textarea {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid #333;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            font-size: 1rem;
        }
        input:focus, textarea:focus { outline: none; border-color: #7b2fff; }
        textarea { resize: vertical; min-height: 80px; }
        .row { display: flex; gap: 1rem; }
        .row > * { flex: 1; }
        .dropzone {
            border: 2px dashed #333;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 0.5rem;
            min-height: 100px;
        }
        .dropzone:hover { border-color: #7b2fff; background: rgba(123, 47, 255, 0.05); }
        .dropzone.dragover { border-color: #00d4ff; background: rgba(0, 212, 255, 0.1); }
        .dropzone.has-file { border-color: #00ff88; border-style: solid; }
        .dropzone img { max-width: 100%; max-height: 120px; border-radius: 8px; margin-top: 0.5rem; }
        .dropzone p { color: #666; font-size: 0.9rem; }
        .dropzone .filename { color: #00ff88; font-size: 0.85rem; margin-top: 0.5rem; }
        .dropzone .clear { color: #ff4444; font-size: 0.75rem; cursor: pointer; margin-top: 0.25rem; }
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
        video { width: 100%; border-radius: 12px; background: #000; margin-top: 1rem; }
        .download {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            text-decoration: none;
            font-weight: 500;
        }
        .download:hover { background: rgba(0, 212, 255, 0.3); }
        .hidden { display: none !important; }
        .optional-tag { color: #666; font-size: 0.75rem; font-weight: normal; }
        .conditioning-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        @media (max-width: 600px) { .conditioning-row { grid-template-columns: 1fr; } }
        .slider-container { margin-top: 1rem; }
        .slider-container input[type="range"] { width: 100%; margin-top: 0.5rem; }
        .slider-hint { color: #666; font-size: 0.75rem; margin-top: 0.25rem; }
        .mode-indicator {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: rgba(123, 47, 255, 0.2);
            border-radius: 20px;
            font-size: 0.8rem;
            color: #7b2fff;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LTX-2 Video Generation</h1>
        <p class="subtitle">19B parameter model • 8-step distilled • FP8 inference on H100</p>
        
        <div class="grid">
            <div class="card">
                <h2>Generation Settings</h2>
                <div class="mode-indicator" id="mode-indicator">Text-to-Video</div>
                
                <label>Prompt <span style="color: #ff2daa;">*</span></label>
                <textarea id="prompt" placeholder="Describe the video you want to generate...">A majestic eagle soaring through a golden sunset sky, cinematic lighting, smooth motion</textarea>
                
                <h3>🖼️ Image Conditioning <span class="optional-tag">(optional)</span></h3>
                <div class="conditioning-row">
                    <div>
                        <label>First Frame</label>
                        <div class="dropzone" id="dropzone-first">
                            <p>Drop image or click</p>
                            <img id="preview-first" class="hidden" />
                            <div class="filename hidden" id="filename-first"></div>
                            <div class="clear hidden" id="clear-first">✕ Remove</div>
                        </div>
                        <input type="file" id="file-first" accept="image/*" />
                    </div>
                    <div>
                        <label>Last Frame</label>
                        <div class="dropzone" id="dropzone-last">
                            <p>Drop image or click</p>
                            <img id="preview-last" class="hidden" />
                            <div class="filename hidden" id="filename-last"></div>
                            <div class="clear hidden" id="clear-last">✕ Remove</div>
                        </div>
                        <input type="file" id="file-last" accept="image/*" />
                    </div>
                </div>
                
                <h3>🎵 Audio Conditioning <span class="optional-tag">(optional)</span></h3>
                <div class="dropzone" id="dropzone-audio">
                    <p>🎵 Drop audio file (WAV, MP3) or click</p>
                    <div class="filename hidden" id="filename-audio"></div>
                    <div class="clear hidden" id="clear-audio">✕ Remove</div>
                </div>
                <input type="file" id="file-audio" accept="audio/*" />
                
                <div class="slider-container hidden" id="audio-strength-container">
                    <label>Audio Conditioning Strength: <span id="strength-value">0.3</span></label>
                    <input type="range" id="audio-strength" min="0" max="1" step="0.1" value="0.3" />
                    <div class="slider-hint">0.0 = preserve audio (weak conditioning) → 1.0 = full diffusion (may have artifacts)</div>
                </div>
                
                <h3>🎬 Video Extension <span class="optional-tag">(optional)</span></h3>
                <div class="dropzone" id="dropzone-video">
                    <p>🎬 Drop video to extend (MP4) or click</p>
                    <video id="preview-video" class="hidden" style="max-height: 120px; max-width: 100%;" muted></video>
                    <div class="filename hidden" id="filename-video"></div>
                    <div class="clear hidden" id="clear-video">✕ Remove</div>
                </div>
                <input type="file" id="file-video" accept="video/*" />
                <div class="slider-hint">Upload a video clip to continue/extend it with new frames</div>
                
                <h3>⚙️ Video Settings</h3>
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
                
                <label style="display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; cursor: pointer;">
                    <input type="checkbox" id="skip-upscaling" checked style="width: auto;" />
                    <span>Skip upscaling (8 steps only, faster)</span>
                </label>
                <div class="slider-hint">Unchecked = 12 steps with 2x upscaling (slower, higher quality)</div>
                
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
        
        // State
        let firstFrameData = null;
        let lastFrameData = null;
        let audioData = null;
        let videoData = null;
        
        // Update mode indicator based on what's selected
        function updateMode() {
            const hasFirst = !!firstFrameData;
            const hasLast = !!lastFrameData;
            const hasAudio = !!audioData;
            const hasVideo = !!videoData;
            
            let mode = 'Text-to-Video';
            if (hasVideo) mode = 'Video Extension';
            else if (hasAudio && hasFirst) mode = 'Image + Audio → Video';
            else if (hasAudio) mode = 'Audio-to-Video';
            else if (hasFirst && hasLast) mode = 'First + Last Frame → Video';
            else if (hasFirst) mode = 'Image-to-Video';
            
            $('mode-indicator').textContent = mode;
            
            // Show/hide audio strength slider
            $('audio-strength-container').classList.toggle('hidden', !hasAudio);
        }
        
        // Generic dropzone setup
        function setupDropzone(dropzoneId, fileInputId, previewId, filenameId, clearId, type, onData) {
            const dropzone = $(dropzoneId);
            const fileInput = $(fileInputId);
            const preview = previewId ? $(previewId) : null;
            const filename = $(filenameId);
            const clear = $(clearId);
            
            dropzone.addEventListener('click', (e) => {
                if (e.target !== clear) fileInput.click();
            });
            dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
            dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
            dropzone.addEventListener('drop', e => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            });
            fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
            
            clear.addEventListener('click', (e) => {
                e.stopPropagation();
                onData(null);
                dropzone.classList.remove('has-file');
                dropzone.querySelector('p').classList.remove('hidden');
                if (preview) { preview.classList.add('hidden'); preview.src = ''; }
                filename.classList.add('hidden');
                clear.classList.add('hidden');
                fileInput.value = '';
                updateMode();
            });
            
            function handleFile(file) {
                const reader = new FileReader();
                reader.onload = e => {
                    const b64 = e.target.result.split(',')[1];
                    onData(b64);
                    dropzone.classList.add('has-file');
                    dropzone.querySelector('p').classList.add('hidden');
                    if (preview && type === 'image') {
                        preview.src = e.target.result;
                        preview.classList.remove('hidden');
                    }
                    if (preview && type === 'video') {
                        preview.src = e.target.result;
                        preview.classList.remove('hidden');
                    }
                    const icon = type === 'audio' ? '🎵 ' : (type === 'video' ? '🎬 ' : '');
                    filename.textContent = icon + file.name;
                    filename.classList.remove('hidden');
                    clear.classList.remove('hidden');
                    updateMode();
                };
                reader.readAsDataURL(file);
            }
        }
        
        // Setup all dropzones
        setupDropzone('dropzone-first', 'file-first', 'preview-first', 'filename-first', 'clear-first', 'image', d => firstFrameData = d);
        setupDropzone('dropzone-last', 'file-last', 'preview-last', 'filename-last', 'clear-last', 'image', d => lastFrameData = d);
        setupDropzone('dropzone-audio', 'file-audio', null, 'filename-audio', 'clear-audio', 'audio', d => audioData = d);
        setupDropzone('dropzone-video', 'file-video', 'preview-video', 'filename-video', 'clear-video', 'video', d => videoData = d);
        
        // Audio strength slider
        $('audio-strength').addEventListener('input', () => {
            $('strength-value').textContent = $('audio-strength').value;
        });
        
        // Generate
        $('generate').addEventListener('click', async () => {
            const btn = $('generate');
            const status = $('status');
            const video = $('video');
            const dl = $('download');
            
            if (!$('prompt').value.trim()) {
                status.className = 'status visible error';
                status.textContent = 'Error: Please enter a prompt';
                return;
            }
            
            btn.disabled = true;
            status.className = 'status visible';
            status.textContent = 'Starting generation...';
            video.removeAttribute('src');
            dl.classList.add('hidden');
            
            try {
                const t0 = performance.now();
                
                // Build form data with all conditionings
                const fd = new FormData();
                fd.append('prompt', $('prompt').value);
                fd.append('width', $('width').value);
                fd.append('height', $('height').value);
                fd.append('num_frames', $('frames').value);
                fd.append('seed', $('seed').value);
                fd.append('skip_upscaling', $('skip-upscaling').checked ? 'true' : 'false');
                
                if (firstFrameData) {
                    fd.append('first_frame', await fetch(`data:image/png;base64,${firstFrameData}`).then(r => r.blob()), 'first.png');
                }
                if (lastFrameData) {
                    fd.append('last_frame', await fetch(`data:image/png;base64,${lastFrameData}`).then(r => r.blob()), 'last.png');
                }
                if (audioData) {
                    fd.append('audio', await fetch(`data:audio/wav;base64,${audioData}`).then(r => r.blob()), 'audio.wav');
                    fd.append('audio_conditioning_strength', $('audio-strength').value);
                }
                if (videoData) {
                    fd.append('extend_video', await fetch(`data:video/mp4;base64,${videoData}`).then(r => r.blob()), 'extend.mp4');
                }
                
                const resp = await fetch('/api/generate', { method: 'POST', body: fd });
                
                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }
                
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const dt = ((performance.now() - t0) / 1000).toFixed(1);
                const numFrames = parseInt($('frames').value);
                
                video.src = url;
                dl.href = url;
                dl.classList.remove('hidden');
                
                status.className = 'status visible success';
                status.textContent = `Done in ${dt}s • ${numFrames} frames @ ${(numFrames / parseFloat(dt)).toFixed(1)} fps`;
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

    @web_app.post("/api/generate")
    async def api_generate(
        prompt: str = Form(...),
        width: int = Form(768),
        height: int = Form(512),
        num_frames: int = Form(97),
        seed: int = Form(42),
        skip_upscaling: str = Form("true"),
        first_frame: UploadFile = File(None),
        last_frame: UploadFile = File(None),
        audio: UploadFile = File(None),
        audio_conditioning_strength: float = Form(0.3),
        extend_video: UploadFile = File(None),
    ):
        """Unified video generation endpoint supporting all conditioning combinations."""
        try:
            import base64
            
            # Process optional first frame
            first_frame_b64 = None
            if first_frame:
                data = await first_frame.read()
                if data:
                    first_frame_b64 = base64.b64encode(data).decode()
            
            # Process optional last frame
            last_frame_b64 = None
            if last_frame:
                data = await last_frame.read()
                if data:
                    last_frame_b64 = base64.b64encode(data).decode()
            
            # Process optional audio
            audio_b64 = None
            if audio:
                data = await audio.read()
                if data:
                    audio_b64 = base64.b64encode(data).decode()
            
            # Process optional video to extend
            extend_video_b64 = None
            if extend_video:
                data = await extend_video.read()
                if data:
                    extend_video_b64 = base64.b64encode(data).decode()
            
            # Parse skip_upscaling (comes as string from form)
            do_skip_upscaling = skip_upscaling.lower() == "true"
            
            engine = OfficialLTX2Engine()
            video_bytes = engine.generate.remote(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=30.0,
                first_frame_b64=first_frame_b64,
                last_frame_b64=last_frame_b64,
                audio_b64=audio_b64,
                audio_conditioning_strength=audio_conditioning_strength,
                skip_upscaling=do_skip_upscaling,
                extend_video_b64=extend_video_b64,
            )
            return Response(content=video_bytes, media_type="video/mp4")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Legacy endpoints (kept for backwards compatibility)
    @web_app.post("/api/t2v")
    async def api_t2v(req: T2VRequest):
        """Text-to-Video API endpoint. [DEPRECATED: Use /api/generate instead]"""
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

    @web_app.post("/api/a2v")
    async def api_a2v(
        audio: UploadFile = File(...),
        prompt: str = Form(...),
        width: int = Form(768),
        height: int = Form(512),
        num_frames: int = Form(97),
        seed: int = Form(42),
        image: UploadFile = File(None),
        audio_conditioning_strength: float = Form(0.3),
    ):
        """Audio-to-Video API endpoint."""
        try:
            import base64
            audio_data = await audio.read()
            audio_b64 = base64.b64encode(audio_data).decode()
            
            # Optional image for combined I2V+A2V
            image_b64 = None
            if image:
                image_data = await image.read()
                if image_data:
                    image_b64 = base64.b64encode(image_data).decode()
            
            engine = OfficialLTX2Engine()
            video_bytes = engine.generate_a2v.remote(
                audio_b64=audio_b64,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=30.0,
                image_b64=image_b64,
                audio_conditioning_strength=audio_conditioning_strength,
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

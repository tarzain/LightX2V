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

# WebSocket endpoint URLs (GPU containers) - UI served separately from lightweight CPU container
WEBSOCKET_ENDPOINT_TURBO = "wss://tmalive--ltx2-official-distilled-loopy-officialltx2engin-9b95f7.modal.run/ws/stream"
WEBSOCKET_ENDPOINT_HQ = "wss://tmalive--ltx2-nondistilled-nondistilledltx2engine-streaming-app.modal.run/ws/stream"

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
        "transformers>=4.50.0,<5.0.0",  # 5.x breaks Gemma3TextConfig
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
        "google-genai",  # For Gemini Flash-Image API
        "aiohttp>=3.13.0",  # Required by google-genai (needs ClientConnectorDNSError added in 3.13+)
        "websockets",  # WebSocket client for LTX-2 connection
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

app = modal.App("ltx2-official-distilled-loopy", image=image)

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


@app.function(timeout=60)
@modal.asgi_app()
def streaming_ui():
    """
    Lightweight CPU-only function to serve the streaming UI.

    This serves the HTML page instantly without waiting for GPU/model loading.
    The page connects to the GPU container's WebSocket endpoint for actual streaming.
    Supports both Turbo (distilled) and HQ (non-distilled) modes.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    ui_app = FastAPI(title="LTX-2 Streaming UI")
    ui_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def inject_endpoints(html):
        """Inject both WebSocket endpoint URLs into the HTML."""
        html = html.replace("__WS_ENDPOINT_TURBO__", WEBSOCKET_ENDPOINT_TURBO)
        html = html.replace("__WS_ENDPOINT_HQ__", WEBSOCKET_ENDPOINT_HQ)
        return html

    @ui_app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(inject_endpoints(STREAMING_HTML))

    @ui_app.get("/stream", response_class=HTMLResponse)
    async def stream():
        return HTMLResponse(inject_endpoints(STREAMING_HTML))

    @ui_app.get("/health")
    async def health():
        return {"status": "healthy", "service": "streaming-ui"}

    return ui_app


@app.cls(
    gpu="H200",
    timeout=3600,
    scaledown_window=300,  # Keep warm for 5 minutes
    enable_memory_snapshot=False,  # TODO: re-enable after code is stable
    # experimental_options={"enable_gpu_snapshot": True},
    volumes={
        MODELS_DIR: model_volume,
        "/outputs": outputs_volume,
        "/vol_cache": cache_volume,
    },
    allow_concurrent_inputs=10,  # Handle multiple requests on same container
)
class OfficialLTX2Engine:
    """
    Run the official LTX-2 DistilledPipeline with GPU memory snapshotting.
    
    Optimized for lowest latency:
    - All models pre-loaded and kept in VRAM (~60GB)
    - torch.compile applied to transformer
    - GPU state snapshotted for instant cold starts
    """

    @modal.enter()  # snap=True disabled for faster iteration
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
        
        # Load transformer
        print(f"   Loading transformer (19B {'FP8' if self.use_fp8 else 'BF16'})...")
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
    
    def _encode_audio(self, audio_path: str, target_duration_seconds: float, return_waveform: bool = False):
        """
        Load and encode audio to latent representation.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            target_duration_seconds: Target duration in seconds
            return_waveform: If True, also return the waveform at 24kHz for pass-through

        Returns:
            Audio latent tensor for conditioning (and optionally waveform at 24kHz)
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
        output_sample_rate = 24000  # Vocoder output rate

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

        # Store original waveform resampled to output rate (24kHz) for pass-through
        waveform_24k = None
        if return_waveform:
            if sr != output_sample_rate:
                resampler_24k = torchaudio.transforms.Resample(sr, output_sample_rate)
                waveform_24k = resampler_24k(waveform)
            else:
                waveform_24k = waveform.clone()
            # Trim/pad to target duration at 24kHz
            target_samples_24k = int(target_duration_seconds * output_sample_rate)
            if waveform_24k.shape[1] > target_samples_24k:
                waveform_24k = waveform_24k[:, :target_samples_24k]
            elif waveform_24k.shape[1] < target_samples_24k:
                padding = torch.zeros(2, target_samples_24k - waveform_24k.shape[1])
                waveform_24k = torch.cat([waveform_24k, padding], dim=1)

        # Resample to encoder rate (16kHz) if needed
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

        if return_waveform:
            return audio_latent, waveform_24k
        return audio_latent

    def encode_audio_for_streaming(
        self,
        audio_b64: str,
        num_frames: int,
        frame_rate: float,
    ) -> "torch.Tensor":
        """
        Encode audio from base64 for streaming segment conditioning.

        Args:
            audio_b64: Base64-encoded audio data (any format torchaudio supports)
            num_frames: Number of frames in the segment
            frame_rate: Frame rate of the video

        Returns:
            Audio latent tensor for conditioning
        """
        import base64
        import tempfile

        # Calculate target duration based on segment parameters
        target_duration = num_frames / frame_rate

        # Decode and save to temp file (torchaudio needs a file path)
        audio_data = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            audio_path = tmp.name

        # Use existing _encode_audio method
        return self._encode_audio(audio_path, target_duration)

    def encode_audio_chunks_for_streaming(
        self,
        audio_b64: str,
        num_frames: int,
        frame_rate: float,
    ) -> list:
        """
        Encode audio from base64, automatically splitting into segment-sized chunks.

        If the audio is longer than one segment, it will be split into multiple
        chunks and each chunk encoded separately. This allows clients to send
        longer audio (e.g., music, speech) and have it automatically queued
        for multiple upcoming segments.

        Args:
            audio_b64: Base64-encoded audio data (any format torchaudio supports)
            num_frames: Number of frames per segment
            frame_rate: Frame rate of the video

        Returns:
            List of (latent, waveform_24k) tuples, one per segment chunk.
            waveform_24k is at 24kHz for direct pass-through output.
        """
        import base64
        import tempfile
        import torch
        import torchaudio

        # Calculate segment duration
        segment_duration = num_frames / frame_rate

        # Decode and load audio
        audio_data = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            audio_path = tmp.name

        # Load audio to get duration
        waveform, sr = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sr

        print(f"   Audio chunking: {audio_duration:.2f}s audio, {segment_duration:.2f}s per segment", flush=True)

        # If audio fits in one segment, just encode it directly (with waveform for pass-through)
        if audio_duration <= segment_duration * 1.1:  # 10% tolerance
            latent, waveform_24k = self._encode_audio(audio_path, segment_duration, return_waveform=True)
            return [(latent, waveform_24k)]

        # Split into chunks WITH OVERLAP to match video segment overlap
        # The overlap ensures that when the output audio is trimmed (to match video overlap),
        # we don't lose any of the original audio
        results = []  # List of (latent, waveform_24k) tuples

        # Audio parameters for chunking
        sample_rate = 16000  # Target sample rate for encoder
        output_sample_rate = 24000  # Vocoder output rate for pass-through

        # Resample to both rates
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform_16k = resampler(waveform)
        else:
            waveform_16k = waveform

        if sr != output_sample_rate:
            resampler_24k = torchaudio.transforms.Resample(sr, output_sample_rate)
            waveform_24k_full = resampler_24k(waveform)
        else:
            waveform_24k_full = waveform.clone()

        # Ensure stereo for both
        for wav in [waveform_16k, waveform_24k_full]:
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

        # Re-assign after ensuring stereo
        if waveform_16k.shape[0] == 1:
            waveform_16k = waveform_16k.repeat(2, 1)
        elif waveform_16k.shape[0] > 2:
            waveform_16k = waveform_16k[:2]

        if waveform_24k_full.shape[0] == 1:
            waveform_24k_full = waveform_24k_full.repeat(2, 1)
        elif waveform_24k_full.shape[0] > 2:
            waveform_24k_full = waveform_24k_full[:2]

        samples_per_chunk_16k = int(segment_duration * sample_rate)
        samples_per_chunk_24k = int(segment_duration * output_sample_rate)
        total_samples_16k = waveform_16k.shape[1]

        # Calculate overlap in samples (same ratio as video: 16 frames out of 49)
        # This matches the overlap_frames = 16 used in generate_streaming
        overlap_frames = 16
        overlap_ratio = overlap_frames / num_frames
        overlap_samples_16k = int(samples_per_chunk_16k * overlap_ratio)
        overlap_samples_24k = int(samples_per_chunk_24k * overlap_ratio)

        # Calculate how many chunks we need, accounting for overlap
        # Each chunk after the first starts overlap_samples earlier
        effective_chunk_advance_16k = samples_per_chunk_16k - overlap_samples_16k
        effective_chunk_advance_24k = samples_per_chunk_24k - overlap_samples_24k
        num_chunks = 1 + max(0, int((total_samples_16k - samples_per_chunk_16k) / effective_chunk_advance_16k) + 1)

        print(f"   Audio chunking: {num_chunks} chunks with overlap (16k: {overlap_samples_16k}, 24k: {overlap_samples_24k} samples)", flush=True)

        for i in range(num_chunks):
            if i == 0:
                start_sample_16k = 0
                start_sample_24k = 0
            else:
                # Start earlier to include overlap (this part will be trimmed at output)
                start_sample_16k = i * effective_chunk_advance_16k
                start_sample_24k = i * effective_chunk_advance_24k

            end_sample_16k = min(start_sample_16k + samples_per_chunk_16k, waveform_16k.shape[1])
            end_sample_24k = min(start_sample_24k + samples_per_chunk_24k, waveform_24k_full.shape[1])

            # Don't create chunks that are too short
            if end_sample_16k - start_sample_16k < samples_per_chunk_16k * 0.5:
                break

            # Extract chunks at both sample rates
            chunk_waveform_16k = waveform_16k[:, start_sample_16k:end_sample_16k]
            chunk_waveform_24k = waveform_24k_full[:, start_sample_24k:end_sample_24k]

            # Pad if needed (last chunk might be shorter)
            if chunk_waveform_16k.shape[1] < samples_per_chunk_16k:
                padding = torch.zeros(2, samples_per_chunk_16k - chunk_waveform_16k.shape[1])
                chunk_waveform_16k = torch.cat([chunk_waveform_16k, padding], dim=1)

            if chunk_waveform_24k.shape[1] < samples_per_chunk_24k:
                padding = torch.zeros(2, samples_per_chunk_24k - chunk_waveform_24k.shape[1])
                chunk_waveform_24k = torch.cat([chunk_waveform_24k, padding], dim=1)

            # Save chunk to temp file and encode
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                torchaudio.save(tmp.name, chunk_waveform_16k, sample_rate)
                chunk_path = tmp.name

            latent = self._encode_audio(chunk_path, segment_duration)
            results.append((latent, chunk_waveform_24k))
            print(f"   Audio chunk {i+1}/{num_chunks} encoded (16k samples {start_sample_16k}-{end_sample_16k})", flush=True)

        return results

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
        rolling_mode: bool = False,
        segment_seconds: float = 5.0,
    ) -> bytes:
        """
        Unified generation method supporting all conditioning combinations:
        - Text only (T2V)
        - Text + first frame (I2V)
        - Text + first + last frame (FL2V)
        - Text + audio (A2V)
        - Text + first frame + audio (I2V+A2V)
        - Video extension (extend existing video)
        - Rolling generation (autoregressive multi-segment)
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
        
        # If rolling mode is enabled, use autoregressive generation
        if rolling_mode:
            return self._generate_rolling(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                first_frame_image=Image.open(io.BytesIO(base64.b64decode(first_frame_b64))).convert("RGB") if first_frame_b64 else None,
                output_name=output_name,
                skip_upscaling=do_skip,
                segment_seconds=segment_seconds,
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

    def _generate_rolling(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        first_frame_image,  # PIL Image or None
        output_name: str,
        skip_upscaling: bool = True,
        segment_seconds: float = 5.0,
        overlap_frames: int = 8,  # Frames to overlap between segments for conditioning
    ) -> bytes:
        """
        Rolling/autoregressive video generation.
        
        Generates video in segments, using the last frames of each segment
        as conditioning for the next segment (similar to video extension).
        This allows generating arbitrarily long videos while maintaining
        temporal coherence.
        
        Args:
            prompt: Text prompt
            seed: Random seed
            height, width: Video dimensions
            num_frames: Total number of frames to generate
            frame_rate: Frame rate
            first_frame_image: Optional PIL Image for the first frame
            output_name: Output filename
            skip_upscaling: Skip 2-stage pipeline
            segment_seconds: Duration of each segment in seconds
            overlap_frames: Number of frames to use as conditioning overlap
        """
        import gc
        import torch
        import numpy as np
        import tempfile
        from PIL import Image
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByKeyframeIndex
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, AUDIO_SAMPLE_RATE
        from ltx_pipelines.utils import helpers as ltx_helpers
        from ltx_pipelines.utils.helpers import (
            euler_denoising_loop,
            noise_video_state,
            noise_audio_state,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import encode_video
        
        # Cap segment frames to avoid OOM - max ~97 frames (3.2s at 30fps) works reliably
        MAX_SEGMENT_FRAMES = 97
        segment_frames = min(int(segment_seconds * frame_rate), MAX_SEGMENT_FRAMES)
        actual_segment_seconds = segment_frames / frame_rate
        total_duration = num_frames / frame_rate
        
        # Calculate number of segments needed
        # Each segment after the first adds (segment_frames - overlap_frames) new frames
        first_segment_new = segment_frames
        subsequent_segment_new = segment_frames - overlap_frames
        
        if num_frames <= segment_frames:
            # Single segment - no rolling needed
            num_segments = 1
        else:
            # First segment contributes segment_frames, each subsequent contributes (segment_frames - overlap)
            remaining_after_first = num_frames - segment_frames
            num_segments = 1 + max(0, (remaining_after_first + subsequent_segment_new - 1) // subsequent_segment_new)
        
        print(f"   Rolling: Total {total_duration:.1f}s ({num_frames} frames) -> {num_segments} segments")
        print(f"   Rolling: Segment size: {actual_segment_seconds:.1f}s ({segment_frames} frames, capped at {MAX_SEGMENT_FRAMES})")
        print(f"   Rolling: Overlap: {overlap_frames} frames, new frames per segment: {subsequent_segment_new}")
        
        device = self.pipeline.device
        dtype = torch.bfloat16
        
        # Clear memory before starting
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.inference_mode():
            # Pre-encode text (shared across all segments)
            text_encoder = self._text_encoder
            video_encoder = self._video_encoder
            video_decoder = self._video_decoder
            transformer = self._transformer
            
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
            
            segment_video_paths = []  # Paths to segment video files
            
            previous_segment_latent = None
            
            for seg_idx in range(num_segments):
                seg_seed = seed + seg_idx  # Vary seed slightly for each segment
                generator = torch.Generator(device=device).manual_seed(seg_seed)
                noiser = GaussianNoiser(generator=generator)
                stepper = EulerDiffusionStep()
                
                # Determine how many frames this segment generates
                if seg_idx == 0:
                    # First segment: no overlap conditioning
                    this_segment_total_frames = segment_frames
                    conditioning_frames = 0
                else:
                    # Subsequent segments: use overlap_frames from previous as conditioning
                    this_segment_total_frames = segment_frames
                    conditioning_frames = overlap_frames
                
                print(f"   Rolling: Segment {seg_idx + 1}/{num_segments} - generating {this_segment_total_frames} frames (conditioning: {conditioning_frames})")
                
                output_shape = VideoPixelShape(
                    batch=1,
                    frames=this_segment_total_frames,
                    width=width,
                    height=height,
                    fps=frame_rate,
                )
                
                conditionings = []
                
                # First segment: use input image if provided
                if seg_idx == 0 and first_frame_image is not None:
                    # Save image to temp file for conditioning
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        first_frame_image.resize((width, height), Image.Resampling.LANCZOS).save(tmp.name, "PNG")
                        first_path = tmp.name
                    
                    # Use I2V conditioning for first frame
                    conditionings = ltx_helpers.image_conditionings_by_replacing_latent(
                        images=[(first_path, 0, 1.0)],
                        height=height,
                        width=width,
                        video_encoder=video_encoder,
                        dtype=dtype,
                        device=device,
                    )
                
                # Subsequent segments: condition on previous segment's last frames (as latent)
                if seg_idx > 0 and previous_segment_latent is not None:
                    # previous_segment_latent is [B, C, T, H, W]
                    # Take the last overlap_frames worth of latents
                    # Note: latent temporal dimension is frames / temporal_compression
                    # For LTX-2, temporal compression is typically 8
                    latent_temporal = previous_segment_latent.shape[2]
                    
                    # Calculate how many latent frames correspond to overlap_frames
                    # Roughly: latent_frames = ceil(pixel_frames / 8)
                    overlap_latent_frames = max(1, overlap_frames // 8)
                    
                    # Get the last N latent frames
                    conditioning_latent = previous_segment_latent[:, :, -overlap_latent_frames:, :, :]
                    
                    print(f"   Rolling: Using {overlap_latent_frames} latent frames (from {latent_temporal} total) as conditioning")
                    
                    # Create conditioning that freezes these frames at the start
                    video_conditioning = VideoConditionByKeyframeIndex(
                        keyframes=conditioning_latent,
                        frame_idx=0,
                        strength=1.0,  # Freeze conditioning frames
                    )
                    conditionings = [video_conditioning]
                
                # Initialize video state
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
                
                # Initialize audio state
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
                
                # Run denoising
                video_state, audio_state = denoising_loop(
                    stage_1_sigmas,
                    video_state,
                    audio_state,
                    stepper,
                )
                
                # Save the latent for next segment's conditioning (before unpatchify!)
                # We need the latent AFTER denoising but BEFORE clearing conditioning
                # Actually, we need to get it before clear_conditioning changes it
                # The latent shape here is still in patchified form, so we need to unpatchify first
                video_state = video_tools.clear_conditioning(video_state)
                video_state = video_tools.unpatchify(video_state)
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)
                
                # Store latent for next segment (unpatchified, [B, C, T, H, W])
                previous_segment_latent = video_state.latent.clone()
                
                print(f"   Rolling: Segment {seg_idx + 1} latent shape: {previous_segment_latent.shape}")
                
                # Decode video using encode_video (handles spatial tiling correctly)
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
                
                # Encode segment to temp file (encode_video handles tile merging!)
                segment_path = f"/tmp/rolling_segment_{seg_idx}.mp4"
                encode_video(
                    video=video_iterator,
                    fps=frame_rate,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=segment_path,
                    video_chunks_number=1,
                )
                
                segment_video_paths.append(segment_path)
                print(f"   Rolling: Segment {seg_idx + 1} saved to {segment_path}")
                
                # Aggressive memory cleanup between segments
                del video_state, audio_state, video_iterator, audio
                del video_tools, audio_tools
                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"   Rolling: All {num_segments} segments complete, concatenating...")
            
            # Free the last segment latent
            del previous_segment_latent
            gc.collect()
            torch.cuda.empty_cache()
            
            # Concatenate all segment videos using ffmpeg
            import subprocess
            import os
            
            out_path = f"/outputs/{output_name}"
            
            if len(segment_video_paths) == 1:
                # Single segment, just copy
                import shutil
                shutil.copy(segment_video_paths[0], out_path)
            else:
                # Create concat file for ffmpeg
                # First, trim each segment after the first to remove overlap
                trimmed_paths = []
                for i, path in enumerate(segment_video_paths):
                    if i == 0:
                        # First segment: use full video
                        trimmed_paths.append(path)
                    else:
                        # Subsequent segments: skip overlap frames at the start
                        trim_seconds = overlap_frames / frame_rate
                        trimmed_path = f"/tmp/rolling_segment_{i}_trimmed.mp4"
                        # Re-encode to ensure clean timestamps
                        result = subprocess.run([
                            "ffmpeg", "-y", "-i", path,
                            "-ss", str(trim_seconds),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            "-c:a", "aac", "-b:a", "128k",
                            "-r", str(int(frame_rate)),  # Force frame rate
                            trimmed_path
                        ], capture_output=True)
                        if result.returncode != 0:
                            print(f"   Rolling: Warning - trim failed: {result.stderr.decode()[:200]}")
                        trimmed_paths.append(trimmed_path)
                
                # Write concat file
                concat_file = "/tmp/rolling_concat.txt"
                with open(concat_file, "w") as f:
                    for path in trimmed_paths:
                        f.write(f"file '{path}'\n")
                
                # Concatenate with re-encoding to fix timestamps
                target_duration = num_frames / frame_rate
                result = subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file,
                    "-t", str(target_duration),  # Trim to exact duration
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "128k",
                    "-r", str(int(frame_rate)),  # Force frame rate
                    "-movflags", "+faststart",  # Enable streaming
                    out_path
                ], capture_output=True)
                if result.returncode != 0:
                    print(f"   Rolling: Warning - concat failed: {result.stderr.decode()[:500]}")
                
                # Cleanup trimmed files
                for path in trimmed_paths[1:]:  # Skip first (original)
                    try:
                        os.remove(path)
                    except:
                        pass
        
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        
        # Cleanup temp files
        import os
        for path in segment_video_paths:
            try:
                os.remove(path)
            except:
                pass
        
        gc.collect()
        torch.cuda.empty_cache()

        total_duration = num_frames / frame_rate
        print(f"   Rolling: Complete! Output: {out_path} ({num_frames} frames, {total_duration:.1f}s)")
        return video_bytes

    def encode_target_image(self, image_base64: str, target_height: int = 480, target_width: int = 832) -> "torch.Tensor":
        """
        Encode a base64 image to latent space for end-frame conditioning.

        Args:
            image_base64: Base64-encoded image data (with or without data URL prefix)
            target_height: Height to resize image to
            target_width: Width to resize image to

        Returns:
            Encoded latent tensor suitable for VideoConditionByKeyframeIndex
        """
        import torch
        import numpy as np
        import base64
        from io import BytesIO
        from PIL import Image

        # Strip data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_data)).convert('RGB')

        # Resize to match generation resolution
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Convert to tensor: [B, C, T, H, W] where T=1 for single frame
        # Normalize to [-1, 1] range (same as LTX's normalize_latent: x / 127.5 - 1.0)
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0  # [H, W, C] in [-1, 1] range
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
        img_tensor = img_tensor.to(dtype=torch.bfloat16, device=self.pipeline.device)

        # Encode to latent space
        with torch.inference_mode():
            encoded_latent = self._video_encoder(img_tensor)

        print(f"   Target image encoded. Latent shape: {encoded_latent.shape}", flush=True)
        return encoded_latent

    def generate_streaming(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        use_second_stage: bool = False,
        is_first_segment: bool = True,
        start_frame_latent: "torch.Tensor | None" = None,
        end_frame_latent: "torch.Tensor | None" = None,
        target_frame_position: float = 1.0,
        audio_latent: "torch.Tensor | None" = None,
        audio_conditioning_strength: float = 0.3,
        original_audio_waveform: "torch.Tensor | None" = None,
        skip_audio_output: bool = False,
    ):
        """
        Generator that yields frames for real-time streaming.

        Args:
            is_first_segment: If True, starts fresh. If False, conditions on previous segment.
            start_frame_latent: Optional latent for start-frame conditioning (first frame image).
            end_frame_latent: Optional latent for end-frame conditioning (target image).
            target_frame_position: Position for target frame (0.0=start, 0.5=middle, 1.0=end).
            audio_latent: Optional audio latent for audio-to-video conditioning.
            audio_conditioning_strength: Strength of audio conditioning (0.0-1.0).
            original_audio_waveform: Original audio waveform at 24kHz for pass-through output.

        Yields:
            dict with either:
            - {"type": "frame_chunk", "frames": [{"data": base64_jpeg, "index": int}, ...]}
            - {"type": "audio", "data": base64_wav, "sample_rate": int}
            - {"type": "segment_complete", "segment": int, "frames": int}
        """
        import gc
        import torch
        import numpy as np
        import base64
        from io import BytesIO
        from PIL import Image
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByKeyframeIndex
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            euler_denoising_loop,
            noise_video_state,
            noise_audio_state,
            simple_denoising_func,
        )

        device = self.pipeline.device
        dtype = torch.bfloat16

        # Use 49 frames per segment (like the streaming app)
        segment_frames = num_frames

        gc.collect()
        torch.cuda.empty_cache()

        with torch.inference_mode():
            # Pre-encode text
            text_encoder = self._text_encoder
            video_encoder = self._video_encoder
            video_decoder = self._video_decoder
            transformer = self._transformer

            print(f"   Streaming: Encoding prompt: {prompt[:60]}...", flush=True)
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            video_context, audio_context = context_p

            # Sigmas for denoising
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

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

            generator = torch.Generator(device=device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            stepper = EulerDiffusionStep()

            output_shape = VideoPixelShape(
                batch=1,
                frames=segment_frames,
                width=width,
                height=height,
                fps=frame_rate,
            )

            # Build conditionings - use previous segment's latent for continuity
            conditionings = []
            has_latent = hasattr(self, '_streaming_last_latent') and self._streaming_last_latent is not None
            print(f"   Streaming: is_first_segment={is_first_segment}, has_previous_latent={has_latent}", flush=True)
            if has_latent:
                print(f"   Streaming: Previous latent shape: {self._streaming_last_latent.shape}", flush=True)

            if not is_first_segment and has_latent:
                print(f"   Streaming: Conditioning on previous segment latent", flush=True)
                start_conditioning = VideoConditionByKeyframeIndex(
                    keyframes=self._streaming_last_latent,
                    frame_idx=0,
                    strength=1.0,
                )
                conditionings.append(start_conditioning)
            elif is_first_segment:
                # Clear any previous latent when starting fresh
                print(f"   Streaming: First segment - clearing previous latent", flush=True)
                self._streaming_last_latent = None
                # Use provided start image for first frame conditioning if available
                if start_frame_latent is not None:
                    print(f"   Streaming: Adding start-frame conditioning (start image), shape: {start_frame_latent.shape}", flush=True)
                    start_conditioning = VideoConditionByKeyframeIndex(
                        keyframes=start_frame_latent,
                        frame_idx=0,
                        strength=1.0,
                    )
                    conditionings.append(start_conditioning)

            # Add target-frame conditioning (target image) if provided
            if end_frame_latent is not None:
                # Calculate frame index from position (0.0=start, 0.5=middle, 1.0=end)
                target_frame_idx = int(target_frame_position * (segment_frames - 1))
                target_frame_idx = max(0, min(segment_frames - 1, target_frame_idx))  # Clamp
                print(f"   Streaming: Adding target-frame conditioning at frame {target_frame_idx}/{segment_frames-1} (pos={target_frame_position:.2f}), shape: {end_frame_latent.shape}", flush=True)
                end_conditioning = VideoConditionByKeyframeIndex(
                    keyframes=end_frame_latent,
                    frame_idx=target_frame_idx,
                    strength=1.0,
                )
                conditionings.append(end_conditioning)

            # Initialize video state
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

            # Initialize audio state - use provided audio or previous latent for continuity
            audio_initial_latent = None
            audio_noise_scale = 1.0  # Default: generate fresh audio

            # Get expected audio latent shape for this segment
            from ltx_core.types import AudioLatentShape
            expected_audio_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
            expected_frames = expected_audio_shape.frames

            # Skip audio conditioning setup when audio output is disabled
            if not skip_audio_output and audio_latent is not None:
                # External audio conditioning provided for this segment
                print(f"   Streaming: Using provided audio conditioning, shape: {audio_latent.shape}", flush=True)

                # Resize audio latent to match expected frames if needed
                actual_frames = audio_latent.shape[2]
                if actual_frames != expected_frames:
                    print(f"   Streaming: Resizing audio latent from {actual_frames} to {expected_frames} frames", flush=True)
                    B, C, T, H = audio_latent.shape
                    audio_flat = audio_latent.reshape(B * C, 1, T, H)
                    audio_resized = torch.nn.functional.interpolate(
                        audio_flat, size=(expected_frames, H), mode='bilinear', align_corners=False
                    )
                    audio_initial_latent = audio_resized.reshape(B, C, expected_frames, H)
                else:
                    audio_initial_latent = audio_latent

                audio_noise_scale = audio_conditioning_strength
                if original_audio_waveform is not None:
                    print(f"   Streaming: Audio with pass-through output, noise_scale={audio_noise_scale}", flush=True)
                else:
                    print(f"   Streaming: Audio conditioning strength: {audio_noise_scale}", flush=True)

            if is_first_segment:
                # Clear previous audio latent when starting fresh
                self._streaming_last_audio_latent = None

            # Track if we need to apply audio continuity blending after denoising
            apply_audio_continuity = False
            prev_audio_overlap = None
            if not skip_audio_output and not is_first_segment and audio_latent is None:
                has_prev_audio = hasattr(self, '_streaming_last_audio_latent') and self._streaming_last_audio_latent is not None
                if has_prev_audio:
                    prev_audio_overlap = self._streaming_last_audio_latent
                    apply_audio_continuity = True
                    print(f"   Streaming: Will apply audio continuity blending after denoising, overlap shape: {prev_audio_overlap.shape}", flush=True)

            # Create the noised audio state (always start fresh for now)
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

            # Run stage 1 denoising (8 steps)
            print(f"   Streaming: Stage 1 denoising ({len(stage_1_sigmas)} steps)...", flush=True)
            video_state, audio_state = denoising_loop(stage_1_sigmas, video_state, audio_state, stepper)

            # Clear conditioning and unpatchify
            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)

            # Apply audio continuity (keyframe-style conditioning)
            # Directly inject the previous segment's overlap at frame 0, like video keyframe conditioning
            # The overlap portion is trimmed from output, so no blending needed
            if apply_audio_continuity and prev_audio_overlap is not None:
                overlap_frame_count = prev_audio_overlap.shape[2]
                total_audio_frames = audio_state.latent.shape[2]
                if overlap_frame_count <= total_audio_frames:
                    # Direct replacement at frame 0 (same as video keyframe conditioning)
                    audio_state.latent[:, :, :overlap_frame_count, :] = prev_audio_overlap
                    print(f"   Streaming: Injected {overlap_frame_count} audio frames at position 0", flush=True)
                else:
                    print(f"   Streaming: Warning - overlap ({overlap_frame_count}) > total frames ({total_audio_frames}), skipping", flush=True)

            # Store LAST latent frames for next segment conditioning
            # Temporal compression is ~8x, so overlap_frames=16 -> 2 latent frames
            # More overlap frames = better continuity but more redundant frames to skip
            overlap_frames = 16  # ~0.5s at 30fps
            latent_overlap = max(1, overlap_frames // 8)  # 2 latent frames
            self._streaming_last_latent = video_state.latent[:, :, -latent_overlap:, :, :].clone()
            self._streaming_overlap_frames = overlap_frames  # Store for frame skipping
            print(f"   Streaming: Stored last {latent_overlap} latent frame(s) ({overlap_frames} video frames) for conditioning, shape: {self._streaming_last_latent.shape}", flush=True)

            # Store audio latent overlap for next segment conditioning (audio continuity)
            if not skip_audio_output:
                audio_latent_frames = audio_state.latent.shape[2]
                audio_overlap_frames = max(1, int(audio_latent_frames * overlap_frames / segment_frames))
                self._streaming_last_audio_latent = audio_state.latent[:, :, -audio_overlap_frames:, :].clone()
                print(f"   Streaming: Stored last {audio_overlap_frames}/{audio_latent_frames} audio latent frames for conditioning, shape: {self._streaming_last_audio_latent.shape}", flush=True)

            # Determine latent for decode
            if use_second_stage:
                print(f"   Streaming: Stage 2 upsampling and refinement...", flush=True)
                # Upsample with proper normalization
                upsampled_latent = upsample_video(
                    latent=video_state.latent[:1],
                    video_encoder=video_encoder,
                    upsampler=self._spatial_upsampler,
                )

                # Stage 2 output shape at 2x resolution
                stage_2_shape = VideoPixelShape(
                    batch=1,
                    frames=segment_frames,
                    width=width * 2,
                    height=height * 2,
                    fps=frame_rate,
                )

                # Re-initialize for stage 2
                video_state_2, video_tools_2 = noise_video_state(
                    output_shape=stage_2_shape,
                    noiser=noiser,
                    conditionings=[],
                    components=self.pipeline.pipeline_components,
                    dtype=dtype,
                    device=device,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=upsampled_latent,
                )

                audio_state_2, audio_tools_2 = noise_audio_state(
                    output_shape=stage_2_shape,
                    noiser=noiser,
                    conditionings=[],
                    components=self.pipeline.pipeline_components,
                    dtype=dtype,
                    device=device,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=audio_state.latent,
                )

                # Run stage 2 denoising (4 steps)
                video_state_2, audio_state_2 = denoising_loop(stage_2_sigmas, video_state_2, audio_state_2, stepper)

                # Clear and unpatchify
                video_state_2 = video_tools_2.clear_conditioning(video_state_2)
                video_state_2 = video_tools_2.unpatchify(video_state_2)
                audio_state_2 = audio_tools_2.clear_conditioning(audio_state_2)
                audio_state_2 = audio_tools_2.unpatchify(audio_state_2)

                video_latent_for_decode = video_state_2.latent
                final_audio_state = audio_state_2
                # Update stored audio latent overlap with refined stage 2 audio
                if not skip_audio_output:
                    audio_latent_frames_2 = audio_state_2.latent.shape[2]
                    audio_overlap_frames_2 = max(1, int(audio_latent_frames_2 * overlap_frames / segment_frames))
                    self._streaming_last_audio_latent = audio_state_2.latent[:, :, -audio_overlap_frames_2:, :].clone()
            else:
                video_latent_for_decode = video_state.latent
                final_audio_state = audio_state

            if not skip_audio_output:
                # Get audio output - either pass-through original or decode from latent
                audio_sample_rate = 24000  # Output rate (vocoder or pass-through)

                if original_audio_waveform is not None:
                    # Pass-through mode: use original waveform directly (no VAE round-trip)
                    print(f"   Streaming: Using original audio waveform (pass-through)", flush=True)
                    audio_np = original_audio_waveform.cpu().numpy()
                    audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                else:
                    # Traditional mode: decode audio from latent
                    print(f"   Streaming: Decoding audio...", flush=True)
                    audio_waveform = vae_decode_audio(
                        latent=final_audio_state.latent[:1],
                        audio_decoder=self._audio_decoder,
                        vocoder=self._vocoder,
                    )
                    audio_np = audio_waveform.cpu().numpy()
                    audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)

                # Trim audio to match skipped video frames (for non-first segments)
                if not is_first_segment and overlap_frames > 0:
                    overlap_duration = overlap_frames / frame_rate
                    samples_to_skip = int(overlap_duration * audio_sample_rate)
                    if audio_np.ndim > 1:
                        audio_np = audio_np[:, samples_to_skip:]
                    else:
                        audio_np = audio_np[samples_to_skip:]
                    print(f"   Streaming: Trimmed {samples_to_skip} audio samples ({overlap_duration:.2f}s) for overlap", flush=True)

                # Convert audio to base64 WAV
                import wave
                import struct
                audio_buffer = BytesIO()
                with wave.open(audio_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(2 if audio_np.ndim > 1 and audio_np.shape[0] == 2 else 1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(audio_sample_rate)
                    if audio_np.ndim > 1 and audio_np.shape[0] == 2:
                        audio_interleaved = audio_np.T.flatten()
                    else:
                        audio_interleaved = audio_np.flatten()
                    wav_file.writeframes(audio_interleaved.tobytes())

                audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
                yield {"type": "audio", "data": audio_base64, "sample_rate": audio_sample_rate}

            # Decode video - yields frame chunks
            print(f"   Streaming: Decoding video frames...", flush=True)
            video_iterator = vae_decode_video(
                video_decoder=video_decoder,
                latent=video_latent_for_decode[:1],
            )

            # Process and yield frames in chunks as the VAE decoder produces them
            skip_frames = 0 if is_first_segment else overlap_frames
            global_frame_idx = 0  # Counts all frames (including skipped)
            output_frame_idx = 0  # Counts only yielded frames

            for chunk in video_iterator:
                if isinstance(chunk, torch.Tensor):
                    chunk_np = chunk.cpu().numpy()
                else:
                    chunk_np = np.array(chunk)

                # Remove batch dimension if present
                while len(chunk_np.shape) > 4:
                    chunk_np = chunk_np.squeeze(0)

                # Handle tensor format
                if len(chunk_np.shape) == 4:
                    if chunk_np.shape[0] == 3:  # [C, T, H, W]
                        chunk_np = np.transpose(chunk_np, (1, 2, 3, 0))
                    elif chunk_np.shape[1] == 3:  # [T, C, H, W]
                        chunk_np = np.transpose(chunk_np, (0, 2, 3, 1))

                # Normalize to 0-255
                if chunk_np.max() <= 1.0:
                    chunk_np = chunk_np * 255
                chunk_np = np.clip(chunk_np, 0, 255).astype(np.uint8)

                # Encode non-skipped frames in this chunk as JPEG
                chunk_frames = []
                for i in range(chunk_np.shape[0]):
                    if global_frame_idx < skip_frames:
                        global_frame_idx += 1
                        continue
                    global_frame_idx += 1

                    frame = chunk_np[i]
                    img = Image.fromarray(frame)
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    chunk_frames.append({
                        "data": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                        "index": output_frame_idx,
                    })
                    output_frame_idx += 1

                if chunk_frames:
                    yield {"type": "frame_chunk", "frames": chunk_frames}

            print(f"   Streaming: Yielded {output_frame_idx} frames (skipped first {skip_frames} overlap frames)", flush=True)
            yield {"type": "segment_complete", "segment": 1, "frames": segment_frames}

            gc.collect()
            torch.cuda.empty_cache()

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

    @modal.asgi_app()
    def streaming_app(self):
        """
        ASGI app for real-time WebSocket streaming with access to preloaded models.

        Endpoints:
        - GET /stream - Streaming UI
        - WS /ws/stream - WebSocket for real-time frame streaming
        - GET /health - Health check
        """
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
        import asyncio
        import json

        app = FastAPI(title="LTX-2 Streaming API")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        engine = self  # Reference to the engine with preloaded models

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(STREAMING_HTML)

        @app.get("/stream", response_class=HTMLResponse)
        async def stream_ui():
            return HTMLResponse(STREAMING_HTML)

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": "LTX-2 Streaming", "streaming": True}

        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket endpoint for real-time video streaming."""
            print("GPU-WS: Connection request received", flush=True)
            await websocket.accept()
            print("GPU-WS: Accepted, ready for messages", flush=True)

            current_prompt = None
            should_stop = False
            segment_count = 0
            # Queues for conditioning - items are consumed in order per segment
            target_image_queue = []  # List of (latent, position) tuples
            audio_queue = []  # List of (latent, strength, waveform_24k) tuples for pass-through
            # Legacy single-item support (for backwards compatibility)
            target_image_latent = None  # For target-frame conditioning
            target_frame_position = 1.0  # Default: end of segment (0.0=start, 0.5=middle, 1.0=end)
            audio_latent = None  # For audio conditioning
            audio_conditioning_strength = 0.3  # Default strength
            original_audio_waveform = None  # For audio pass-through (24kHz)
            reset_next_segment = False  # Flag to treat next segment as first (history reset)
            
            # Start/end images for first segment (can be set via set_start_image before start)
            start_image_latent = None
            end_image_latent = None
            
            # Loopy mode: loop on current image until new target arrives
            loopy_mode = False  # Enabled via start message
            loop_image_latent = None  # Current image to loop on (both start and end)
            pending_loop_update = None  # Target latent to become new loop image after transition

            try:
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        msg = json.loads(data)

                        if msg.get("action") == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                        if msg.get("action") == "stop":
                            should_stop = True
                            await websocket.send_json({"type": "stopped"})
                            break

                        elif msg.get("action") == "update_prompt":
                            current_prompt = msg.get("prompt", current_prompt)
                            await websocket.send_json({"type": "prompt_updated", "prompt": current_prompt})

                        elif msg.get("action") == "set_start_image":
                            # Set start image for first segment (sent separately to avoid large message sizes)
                            image_data = msg.get("image")
                            if image_data:
                                try:
                                    img_height = msg.get("height", 480)
                                    img_width = msg.get("width", 832)
                                    start_image_latent = engine.encode_target_image(image_data, img_height, img_width)
                                    await websocket.send_json({"type": "start_image_set", "success": True})
                                    print(f"   WebSocket: Start image set (latent shape: {start_image_latent.shape})", flush=True)
                                    # In loopy mode, start image also becomes the loop image
                                    if loopy_mode:
                                        loop_image_latent = start_image_latent
                                        print(f"   WebSocket: Loopy mode - start image is now loop image", flush=True)
                                except Exception as e:
                                    print(f"   WebSocket: Failed to encode start image: {e}", flush=True)
                                    await websocket.send_json({"type": "start_image_set", "success": False, "error": str(e)})
                            else:
                                await websocket.send_json({"type": "start_image_set", "success": False, "error": "No image data"})

                        elif msg.get("action") == "set_target_image":
                            # Encode target image for target-frame conditioning (queued for upcoming segments)
                            image_data = msg.get("image")
                            if image_data:
                                try:
                                    target_height = msg.get("height", 480)
                                    target_width = msg.get("width", 832)
                                    latent = engine.encode_target_image(
                                        image_data, target_height, target_width
                                    )
                                    # Get target frame position (0.0=start, 0.5=middle, 1.0=end)
                                    position = float(msg.get("position", 1.0))
                                    position = max(0.0, min(1.0, position))
                                    # Add to queue for upcoming segments
                                    target_image_queue.append((latent, position))
                                    await websocket.send_json({
                                        "type": "target_image_set",
                                        "success": True,
                                        "position": position,
                                        "queue_length": len(target_image_queue)
                                    })
                                    print(f"   WebSocket: Target image queued at position {position:.2f} (queue: {len(target_image_queue)})", flush=True)
                                except Exception as e:
                                    print(f"   WebSocket: Failed to encode target image: {e}", flush=True)
                                    await websocket.send_json({"type": "target_image_set", "success": False, "error": str(e)})
                            else:
                                await websocket.send_json({"type": "target_image_set", "success": False, "error": "No image data"})

                        elif msg.get("action") == "clear_target_image":
                            target_image_latent = None
                            await websocket.send_json({"type": "target_image_cleared"})
                            print(f"   WebSocket: Target image cleared", flush=True)

                        elif msg.get("action") == "reset_history":
                            # Clear the model's conditioning state - next segment will be generated fresh
                            engine._streaming_last_latent = None
                            reset_next_segment = True
                            await websocket.send_json({"type": "history_reset"})
                            print(f"   WebSocket: History reset - next segment will start fresh", flush=True)

                        elif msg.get("action") == "set_audio":
                            # Encode audio for conditioning (queued for upcoming segments)
                            # Automatically splits longer audio into segment-sized chunks
                            audio_data = msg.get("audio")
                            if audio_data:
                                try:
                                    # Get segment parameters for duration calculation
                                    seg_num_frames = msg.get("num_frames", 49)
                                    seg_frame_rate = msg.get("frame_rate", 30.0)
                                    strength = float(msg.get("strength", 0.3))
                                    strength = max(0.0, min(1.0, strength))

                                    # Encode and auto-chunk if needed (returns list of (latent, waveform) tuples)
                                    chunks = engine.encode_audio_chunks_for_streaming(
                                        audio_b64=audio_data,
                                        num_frames=seg_num_frames,
                                        frame_rate=seg_frame_rate,
                                    )
                                    # Add all chunks to queue with waveform for pass-through
                                    for latent, waveform in chunks:
                                        audio_queue.append((latent, strength, waveform))
                                    await websocket.send_json({
                                        "type": "audio_set",
                                        "success": True,
                                        "strength": strength,
                                        "chunks_added": len(chunks),
                                        "queue_length": len(audio_queue)
                                    })
                                    print(f"   WebSocket: Audio queued ({len(chunks)} chunks) with strength {strength:.2f} (queue: {len(audio_queue)})", flush=True)
                                except Exception as e:
                                    print(f"   WebSocket: Failed to encode audio: {e}", flush=True)
                                    await websocket.send_json({"type": "audio_set", "success": False, "error": str(e)})
                            else:
                                await websocket.send_json({"type": "audio_set", "success": False, "error": "No audio data"})

                        elif msg.get("action") == "clear_audio":
                            # Clear the audio queue
                            audio_queue.clear()
                            audio_latent = None
                            await websocket.send_json({"type": "audio_cleared"})
                            print(f"   WebSocket: Audio queue cleared", flush=True)

                        elif msg.get("action") == "set_next_segment":
                            # Combined action to set prompt, target image, and/or audio for next segment
                            response = {"type": "next_segment_set"}
                            if "prompt" in msg:
                                current_prompt = msg.get("prompt")
                                response["prompt"] = current_prompt
                                print(f"   WebSocket: Next segment prompt: {current_prompt[:50]}...", flush=True)
                            if "target_image" in msg:
                                image_data = msg.get("target_image")
                                img_height = msg.get("height", 480)
                                img_width = msg.get("width", 832)
                                if image_data:
                                    try:
                                        latent = engine.encode_target_image(image_data, img_height, img_width)
                                        position = float(msg.get("position", 1.0))
                                        position = max(0.0, min(1.0, position))
                                        target_image_queue.append((latent, position))
                                        response["target_image_set"] = True
                                        response["target_image_queue_length"] = len(target_image_queue)
                                        print(f"   WebSocket: Next segment target image queued (queue: {len(target_image_queue)})", flush=True)
                                    except Exception as e:
                                        response["target_image_set"] = False
                                        response["target_image_error"] = str(e)
                                        print(f"   WebSocket: Failed to encode target image: {e}", flush=True)
                            # Update target frame position if provided (legacy single-item support)
                            if "position" in msg and "target_image" not in msg:
                                target_frame_position = float(msg.get("position", 1.0))
                                target_frame_position = max(0.0, min(1.0, target_frame_position))
                                response["position"] = target_frame_position
                            # Add audio for next segment(s) - auto-chunks longer audio
                            if "audio" in msg:
                                audio_data = msg.get("audio")
                                if audio_data:
                                    try:
                                        # Get segment parameters
                                        seg_frames = msg.get("num_frames", 49)
                                        seg_fps = msg.get("frame_rate", 30.0)
                                        strength = float(msg.get("audio_strength", 0.3))
                                        strength = max(0.0, min(1.0, strength))
                                        chunks = engine.encode_audio_chunks_for_streaming(
                                            audio_b64=audio_data,
                                            num_frames=seg_frames,
                                            frame_rate=seg_fps,
                                        )
                                        for latent, waveform in chunks:
                                            audio_queue.append((latent, strength, waveform))
                                        response["audio_set"] = True
                                        response["audio_chunks_added"] = len(chunks)
                                        response["audio_queue_length"] = len(audio_queue)
                                        print(f"   WebSocket: Next segment audio queued ({len(chunks)} chunks) (queue: {len(audio_queue)})", flush=True)
                                    except Exception as e:
                                        response["audio_set"] = False
                                        response["audio_error"] = str(e)
                                        print(f"   WebSocket: Failed to encode audio: {e}", flush=True)
                            await websocket.send_json(response)

                        elif msg.get("action") == "start":
                            current_prompt = msg.get("prompt", "A beautiful landscape")
                            seed = msg.get("seed", 42)
                            height = msg.get("height", 480)
                            width = msg.get("width", 832)
                            num_frames = msg.get("num_frames", 49)
                            frame_rate = msg.get("frame_rate", 24.0)
                            use_second_stage = msg.get("use_second_stage", False)
                            max_segments = msg.get("max_segments", 10)
                            skip_audio_output = msg.get("skip_audio", False)

                            # Loopy mode: loop on current image until target arrives
                            loopy_mode = msg.get("loopy_mode", False)

                            # Encode optional start/end images for the first segment
                            # Note: start_image_latent may already be set via set_start_image action
                            started_response = {"type": "started", "prompt": current_prompt, "loopy_mode": loopy_mode}
                            
                            # Check if start image was pre-set via set_start_image action
                            if start_image_latent is not None:
                                started_response["start_image_set"] = True
                                started_response["start_image_source"] = "pre_set"
                                print(f"   WebSocket: Using pre-set start image for first segment", flush=True)
                                if loopy_mode and loop_image_latent is None:
                                    loop_image_latent = start_image_latent
                                    started_response["loop_image_set"] = True
                                    print(f"   WebSocket: Loopy mode - using pre-set start image as loop image", flush=True)

                            # Encode start_image from message if provided and not already set
                            if msg.get("start_image") and start_image_latent is None:
                                try:
                                    start_image_latent = engine.encode_target_image(msg["start_image"], height, width)
                                    started_response["start_image_set"] = True
                                    started_response["start_image_source"] = "message"
                                    print(f"   WebSocket: Start image encoded for first segment", flush=True)
                                    
                                    # In loopy mode, start image becomes the loop image
                                    if loopy_mode:
                                        loop_image_latent = start_image_latent
                                        started_response["loop_image_set"] = True
                                        print(f"   WebSocket: Loopy mode enabled - start image is now loop image", flush=True)
                                except Exception as e:
                                    started_response["start_image_error"] = str(e)
                                    print(f"   WebSocket: Failed to encode start image: {e}", flush=True)

                            if msg.get("end_image"):
                                try:
                                    end_image_latent = engine.encode_target_image(msg["end_image"], height, width)
                                    target_image_latent = end_image_latent  # Use as target for first segment
                                    started_response["end_image_set"] = True
                                    print(f"   WebSocket: End image encoded for first segment", flush=True)
                                except Exception as e:
                                    started_response["end_image_error"] = str(e)
                                    print(f"   WebSocket: Failed to encode end image: {e}", flush=True)

                            # Handle start_audio - encode and queue for first segment(s)
                            if msg.get("start_audio"):
                                try:
                                    audio_data = msg["start_audio"]
                                    audio_strength = float(msg.get("audio_strength", 0.3))
                                    # Use auto-chunking to split long audio into segment-sized chunks
                                    chunks = engine.encode_audio_chunks_for_streaming(
                                        audio_b64=audio_data,
                                        num_frames=num_frames,
                                        frame_rate=frame_rate,
                                    )
                                    for latent, waveform in chunks:
                                        audio_queue.append((latent, audio_strength, waveform))
                                    started_response["start_audio_set"] = True
                                    started_response["audio_chunks"] = len(chunks)
                                    print(f"   WebSocket: Start audio encoded ({len(chunks)} chunks) for first segment(s)", flush=True)
                                except Exception as e:
                                    started_response["start_audio_error"] = str(e)
                                    print(f"   WebSocket: Failed to encode start audio: {e}", flush=True)

                            await websocket.send_json(started_response)

                            segment_count = 0
                            should_stop = False

                            while not should_stop and segment_count < max_segments:
                                segment_count += 1
                                seg_seed = seed + segment_count

                                # Check for messages with longer timeout to catch prompt/image updates
                                try:
                                    check_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                                    check_msg = json.loads(check_data)
                                    if check_msg.get("action") == "stop":
                                        should_stop = True
                                        break
                                    elif check_msg.get("action") == "update_prompt":
                                        current_prompt = check_msg.get("prompt", current_prompt)
                                        print(f"   WebSocket: Prompt updated to: {current_prompt[:50]}...", flush=True)
                                        await websocket.send_json({"type": "prompt_updated", "prompt": current_prompt})
                                    elif check_msg.get("action") == "set_target_image":
                                        image_data = check_msg.get("image")
                                        if image_data:
                                            try:
                                                latent = engine.encode_target_image(image_data, height, width)
                                                position = float(check_msg.get("position", 1.0))
                                                position = max(0.0, min(1.0, position))
                                                target_image_queue.append((latent, position))
                                                await websocket.send_json({
                                                    "type": "target_image_set",
                                                    "success": True,
                                                    "position": position,
                                                    "queue_length": len(target_image_queue)
                                                })
                                                print(f"   WebSocket: Target image queued (pre-segment) at position {position:.2f} (queue: {len(target_image_queue)})", flush=True)
                                            except Exception as e:
                                                print(f"   WebSocket: Failed to encode target image: {e}", flush=True)
                                                await websocket.send_json({"type": "target_image_set", "success": False, "error": str(e)})
                                    elif check_msg.get("action") == "clear_target_image":
                                        target_image_queue.clear()
                                        target_image_latent = None
                                        await websocket.send_json({"type": "target_image_cleared"})
                                    elif check_msg.get("action") == "set_audio":
                                        audio_data = check_msg.get("audio")
                                        if audio_data:
                                            try:
                                                strength = float(check_msg.get("strength", 0.3))
                                                strength = max(0.0, min(1.0, strength))
                                                chunks = engine.encode_audio_chunks_for_streaming(
                                                    audio_b64=audio_data,
                                                    num_frames=num_frames,
                                                    frame_rate=frame_rate,
                                                )
                                                for latent, waveform in chunks:
                                                    audio_queue.append((latent, strength, waveform))
                                                await websocket.send_json({
                                                    "type": "audio_set",
                                                    "success": True,
                                                    "strength": strength,
                                                    "chunks_added": len(chunks),
                                                    "queue_length": len(audio_queue)
                                                })
                                                print(f"   WebSocket: Audio queued (pre-segment) ({len(chunks)} chunks) with strength {strength:.2f} (queue: {len(audio_queue)})", flush=True)
                                            except Exception as e:
                                                print(f"   WebSocket: Failed to encode audio: {e}", flush=True)
                                                await websocket.send_json({"type": "audio_set", "success": False, "error": str(e)})
                                    elif check_msg.get("action") == "clear_audio":
                                        audio_queue.clear()
                                        audio_latent = None
                                        await websocket.send_json({"type": "audio_cleared"})
                                    elif check_msg.get("action") == "reset_history":
                                        # Clear the model's conditioning state - next segment will be generated fresh
                                        engine._streaming_last_latent = None
                                        reset_next_segment = True
                                        await websocket.send_json({"type": "history_reset"})
                                        print(f"   WebSocket: History reset (pre-segment) - next segment will start fresh", flush=True)
                                    elif check_msg.get("action") == "set_next_segment":
                                        # Combined action to set prompt, target image, and/or audio
                                        response = {"type": "next_segment_set"}
                                        if "prompt" in check_msg:
                                            current_prompt = check_msg.get("prompt")
                                            response["prompt"] = current_prompt
                                            print(f"   WebSocket: Next segment prompt (pre-segment): {current_prompt[:50]}...", flush=True)
                                        if "target_image" in check_msg:
                                            image_data = check_msg.get("target_image")
                                            if image_data:
                                                try:
                                                    latent = engine.encode_target_image(image_data, height, width)
                                                    position = float(check_msg.get("position", 1.0))
                                                    position = max(0.0, min(1.0, position))
                                                    target_image_queue.append((latent, position))
                                                    response["target_image_set"] = True
                                                    response["target_image_queue_length"] = len(target_image_queue)
                                                    print(f"   WebSocket: Next segment target image queued (pre-segment) (queue: {len(target_image_queue)})", flush=True)
                                                except Exception as e:
                                                    response["target_image_set"] = False
                                                    response["target_image_error"] = str(e)
                                        # Add audio for next segment(s) - auto-chunks longer audio
                                        if "audio" in check_msg:
                                            audio_data = check_msg.get("audio")
                                            if audio_data:
                                                try:
                                                    strength = float(check_msg.get("audio_strength", 0.3))
                                                    strength = max(0.0, min(1.0, strength))
                                                    chunks = engine.encode_audio_chunks_for_streaming(
                                                        audio_b64=audio_data,
                                                        num_frames=num_frames,
                                                        frame_rate=frame_rate,
                                                    )
                                                    for latent, waveform in chunks:
                                                        audio_queue.append((latent, strength, waveform))
                                                    response["audio_set"] = True
                                                    response["audio_chunks_added"] = len(chunks)
                                                    response["audio_queue_length"] = len(audio_queue)
                                                    print(f"   WebSocket: Next segment audio queued (pre-segment) ({len(chunks)} chunks) (queue: {len(audio_queue)})", flush=True)
                                                except Exception as e:
                                                    response["audio_set"] = False
                                                    response["audio_error"] = str(e)
                                        await websocket.send_json(response)
                                except asyncio.TimeoutError:
                                    pass

                                if should_stop:
                                    break

                                # Check if we have a target image for this segment
                                has_target = target_image_latent is not None
                                print(f"   WebSocket: Segment {segment_count} using prompt: {current_prompt[:50]}...{' (with target image)' if has_target else ''}", flush=True)
                                await websocket.send_json({
                                    "type": "segment_start",
                                    "segment": segment_count,
                                    "prompt": current_prompt,
                                    "has_target_image": has_target
                                })

                                # Run generation in thread pool
                                import concurrent.futures
                                loop = asyncio.get_running_loop()

                                # Handle loopy mode vs normal mode for target image selection
                                pending_loop_update = None  # Will be set if we need to update loop_image_latent after segment
                                
                                if loopy_mode and loop_image_latent is not None:
                                    # Loopy mode: either transition to new target or loop on current image
                                    if target_image_queue:
                                        # Transition mode: use target from queue, then it becomes the new loop image
                                        target_image_latent, target_frame_position = target_image_queue.pop(0)
                                        pending_loop_update = target_image_latent  # Update loop image after this segment
                                        print(f"   WebSocket: LOOPY TRANSITION - using target from queue, will become new loop image (remaining: {len(target_image_queue)})", flush=True)
                                    else:
                                        # Loop mode: use loop_image_latent as end frame to create seamless loop
                                        target_image_latent = loop_image_latent
                                        target_frame_position = 1.0  # Always at end for looping
                                        print(f"   WebSocket: LOOPY MODE - looping on current image (A->A)", flush=True)
                                else:
                                    # Normal mode: pop from queues if available
                                    if target_image_queue:
                                        target_image_latent, target_frame_position = target_image_queue.pop(0)
                                        print(f"   WebSocket: Popped target image from queue (remaining: {len(target_image_queue)})", flush=True)

                                if audio_queue:
                                    audio_latent, audio_conditioning_strength, original_audio_waveform = audio_queue.pop(0)
                                    print(f"   WebSocket: Popped audio from queue (remaining: {len(audio_queue)})", flush=True)
                                else:
                                    audio_latent = None
                                    audio_conditioning_strength = 0.3
                                    original_audio_waveform = None

                                # Capture variables for closure (including target image and audio)
                                _prompt = current_prompt
                                _seed = seg_seed
                                _height = height
                                _width = width
                                _frames = num_frames
                                _fps = frame_rate
                                _stage2 = use_second_stage
                                _is_first = (segment_count == 1) or reset_next_segment
                                if reset_next_segment:
                                    reset_next_segment = False  # Clear the flag after using it
                                    print(f"   WebSocket: History reset applied - treating as first segment", flush=True)
                                _target_latent = target_image_latent  # Capture for this segment
                                _target_position = target_frame_position  # Capture position for this segment
                                _audio_latent = audio_latent  # Capture for this segment
                                _audio_strength = audio_conditioning_strength  # Capture for this segment
                                _original_audio_waveform = original_audio_waveform  # Capture for pass-through
                                # Start image only applies to first segment
                                _start_latent = start_image_latent if _is_first else None
                                
                                # In loopy mode, log the mode we're in
                                if loopy_mode:
                                    mode_str = "TRANSITION" if pending_loop_update is not None else "LOOP"
                                    print(f"   WebSocket: Starting segment {segment_count} [{mode_str}], is_first={_is_first}, has_start_image={_start_latent is not None}, has_end_image={_target_latent is not None}, target_pos={_target_position:.2f}", flush=True)
                                else:
                                    print(f"   WebSocket: Starting segment {segment_count}, is_first={_is_first}, has_start_image={_start_latent is not None}, has_audio={_audio_latent is not None}, target_pos={_target_position:.2f}", flush=True)

                                # Clear target image after capturing (one-shot use) - only in non-loopy mode
                                # In loopy mode, target_image_latent is reused for looping
                                if not loopy_mode and target_image_latent is not None:
                                    target_image_latent = None
                                    await websocket.send_json({"type": "target_image_used"})

                                # Clear start image after first segment
                                if _is_first and start_image_latent is not None:
                                    start_image_latent = None
                                    await websocket.send_json({"type": "start_image_used"})

                                # Notify if audio was used (one-shot)
                                if _audio_latent is not None:
                                    await websocket.send_json({"type": "audio_used"})

                                frame_queue = asyncio.Queue()
                                keepalive_active = True

                                async def keepalive_sender():
                                    while keepalive_active and not should_stop:
                                        await asyncio.sleep(5)
                                        try:
                                            await websocket.send_json({"type": "keepalive"})
                                        except Exception:
                                            break

                                def run_generation_to_queue():
                                    try:
                                        for item in engine.generate_streaming(
                                            prompt=_prompt,
                                            seed=_seed,
                                            height=_height,
                                            width=_width,
                                            num_frames=_frames,
                                            frame_rate=_fps,
                                            use_second_stage=_stage2,
                                            is_first_segment=_is_first,
                                            start_frame_latent=_start_latent,
                                            end_frame_latent=_target_latent,
                                            target_frame_position=_target_position,
                                            audio_latent=_audio_latent,
                                            audio_conditioning_strength=_audio_strength,
                                            original_audio_waveform=_original_audio_waveform,
                                            skip_audio_output=skip_audio_output,
                                        ):
                                            asyncio.run_coroutine_threadsafe(frame_queue.put(item), loop)
                                    except Exception as e:
                                        asyncio.run_coroutine_threadsafe(
                                            frame_queue.put({"type": "error", "message": f"generation_error: {e}"}),
                                            loop,
                                        )
                                    finally:
                                        asyncio.run_coroutine_threadsafe(frame_queue.put(None), loop)

                                keepalive_task = asyncio.create_task(keepalive_sender())
                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    generation_future = loop.run_in_executor(pool, run_generation_to_queue)

                                    while True:
                                        item = await frame_queue.get()
                                        if item is None:
                                            break
                                        if should_stop:
                                            break
                                        # Add segment number to frame chunk messages for debugging
                                        if item.get("type") == "frame_chunk":
                                            item["segment"] = segment_count
                                        await websocket.send_json(item)

                                        try:
                                            check_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                                            check_msg = json.loads(check_data)
                                            if check_msg.get("action") == "stop":
                                                should_stop = True
                                                break
                                            elif check_msg.get("action") == "update_prompt":
                                                current_prompt = check_msg.get("prompt", current_prompt)
                                                print(f"   WebSocket: Prompt updated (mid-segment) to: {current_prompt[:50]}...", flush=True)
                                                await websocket.send_json({"type": "prompt_updated", "prompt": current_prompt})
                                            elif check_msg.get("action") == "set_target_image":
                                                image_data = check_msg.get("image")
                                                if image_data:
                                                    try:
                                                        latent = engine.encode_target_image(image_data, height, width)
                                                        position = float(check_msg.get("position", 1.0))
                                                        position = max(0.0, min(1.0, position))
                                                        target_image_queue.append((latent, position))
                                                        await websocket.send_json({
                                                            "type": "target_image_set",
                                                            "success": True,
                                                            "position": position,
                                                            "queue_length": len(target_image_queue)
                                                        })
                                                        print(f"   WebSocket: Target image queued (mid-segment) at position {position:.2f} (queue: {len(target_image_queue)})", flush=True)
                                                    except Exception as e:
                                                        print(f"   WebSocket: Failed to encode target image: {e}", flush=True)
                                                        await websocket.send_json({"type": "target_image_set", "success": False, "error": str(e)})
                                            elif check_msg.get("action") == "clear_target_image":
                                                target_image_queue.clear()
                                                target_image_latent = None
                                                await websocket.send_json({"type": "target_image_cleared"})
                                            elif check_msg.get("action") == "set_audio":
                                                audio_data = check_msg.get("audio")
                                                if audio_data:
                                                    try:
                                                        strength = float(check_msg.get("strength", 0.3))
                                                        strength = max(0.0, min(1.0, strength))
                                                        chunks = engine.encode_audio_chunks_for_streaming(
                                                            audio_b64=audio_data,
                                                            num_frames=num_frames,
                                                            frame_rate=frame_rate,
                                                        )
                                                        for latent, waveform in chunks:
                                                            audio_queue.append((latent, strength, waveform))
                                                        await websocket.send_json({
                                                            "type": "audio_set",
                                                            "success": True,
                                                            "strength": strength,
                                                            "chunks_added": len(chunks),
                                                            "queue_length": len(audio_queue)
                                                        })
                                                        print(f"   WebSocket: Audio queued (mid-segment) ({len(chunks)} chunks) with strength {strength:.2f} (queue: {len(audio_queue)})", flush=True)
                                                    except Exception as e:
                                                        print(f"   WebSocket: Failed to encode audio: {e}", flush=True)
                                                        await websocket.send_json({"type": "audio_set", "success": False, "error": str(e)})
                                            elif check_msg.get("action") == "clear_audio":
                                                audio_queue.clear()
                                                audio_latent = None
                                                await websocket.send_json({"type": "audio_cleared"})
                                            elif check_msg.get("action") == "reset_history":
                                                # Clear the model's conditioning state - next segment will be generated fresh
                                                engine._streaming_last_latent = None
                                                reset_next_segment = True
                                                await websocket.send_json({"type": "history_reset"})
                                                print(f"   WebSocket: History reset (mid-segment) - next segment will start fresh", flush=True)
                                            elif check_msg.get("action") == "set_next_segment":
                                                # Combined action to set prompt, target image, and/or audio
                                                response = {"type": "next_segment_set"}
                                                if "prompt" in check_msg:
                                                    current_prompt = check_msg.get("prompt")
                                                    response["prompt"] = current_prompt
                                                    print(f"   WebSocket: Next segment prompt (mid-segment): {current_prompt[:50]}...", flush=True)
                                                if "target_image" in check_msg:
                                                    image_data = check_msg.get("target_image")
                                                    if image_data:
                                                        try:
                                                            latent = engine.encode_target_image(image_data, height, width)
                                                            position = float(check_msg.get("position", 1.0))
                                                            position = max(0.0, min(1.0, position))
                                                            target_image_queue.append((latent, position))
                                                            response["target_image_set"] = True
                                                            response["target_image_queue_length"] = len(target_image_queue)
                                                            print(f"   WebSocket: Next segment target image queued (mid-segment) (queue: {len(target_image_queue)})", flush=True)
                                                        except Exception as e:
                                                            response["target_image_set"] = False
                                                            response["target_image_error"] = str(e)
                                                # Add audio for next segment(s) - auto-chunks longer audio
                                                if "audio" in check_msg:
                                                    audio_data = check_msg.get("audio")
                                                    if audio_data:
                                                        try:
                                                            strength = float(check_msg.get("audio_strength", 0.3))
                                                            strength = max(0.0, min(1.0, strength))
                                                            chunks = engine.encode_audio_chunks_for_streaming(
                                                                audio_b64=audio_data,
                                                                num_frames=num_frames,
                                                                frame_rate=frame_rate,
                                                            )
                                                            for latent, waveform in chunks:
                                                                audio_queue.append((latent, strength, waveform))
                                                            response["audio_set"] = True
                                                            response["audio_chunks_added"] = len(chunks)
                                                            response["audio_queue_length"] = len(audio_queue)
                                                            print(f"   WebSocket: Next segment audio queued (mid-segment) ({len(chunks)} chunks) (queue: {len(audio_queue)})", flush=True)
                                                        except Exception as e:
                                                            response["audio_set"] = False
                                                            response["audio_error"] = str(e)
                                                await websocket.send_json(response)
                                        except asyncio.TimeoutError:
                                            pass

                                    try:
                                        await generation_future
                                    except Exception:
                                        pass

                                keepalive_active = False
                                try:
                                    await keepalive_task
                                except Exception:
                                    pass

                                # After segment completes: update loop image if we just did a transition
                                if pending_loop_update is not None:
                                    loop_image_latent = pending_loop_update
                                    print(f"   WebSocket: LOOPY MODE - updated loop image after transition (now looping on new image)", flush=True)
                                    await websocket.send_json({"type": "loop_image_updated"})
                                    pending_loop_update = None

                            if should_stop:
                                await websocket.send_json({"type": "stopped"})
                            else:
                                await websocket.send_json({"type": "complete", "segments": segment_count})

                    except asyncio.TimeoutError:
                        continue

            except WebSocketDisconnect:
                print("WebSocket disconnected (client closed connection)", flush=True)
            except Exception as e:
                print(f"WebSocket error: {type(e).__name__}: {e}", flush=True)
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except:
                    pass

        return app


# ============================================================================
# Web API
# ============================================================================

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

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
                
                <label style="display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; cursor: pointer;">
                    <input type="checkbox" id="rolling-mode" style="width: auto;" />
                    <span>🔄 Rolling mode (autoregressive long video)</span>
                </label>
                <div class="slider-hint">Generate in segments for videos longer than ~3s with better coherence</div>
                
                <div id="rolling-options" style="display: none; margin-top: 0.5rem; padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <label>Segment Duration (seconds)</label>
                    <input type="range" id="segment-duration" min="2" max="5" step="0.5" value="3" />
                    <div class="slider-hint">Each segment: <span id="segment-value">3.0</span>s (capped at ~3.2s for memory)</div>
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
        
        // Rolling mode toggle
        $('rolling-mode').addEventListener('change', () => {
            $('rolling-options').style.display = $('rolling-mode').checked ? 'block' : 'none';
        });
        
        // Segment duration slider
        $('segment-duration').addEventListener('input', () => {
            $('segment-value').textContent = parseFloat($('segment-duration').value).toFixed(1);
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
                fd.append('rolling_mode', $('rolling-mode').checked ? 'true' : 'false');
                fd.append('segment_seconds', $('segment-duration').value);
                
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
        rolling_mode: str = Form("false"),
        segment_seconds: float = Form(3.0),
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
            do_rolling_mode = rolling_mode.lower() == "true"
            
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
                rolling_mode=do_rolling_mode,
                segment_seconds=segment_seconds,
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

    @web_app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for real-time video streaming.

        Client sends:
        - {"action": "start", "prompt": str, "seed": int, "height": int, "width": int,
           "num_frames": int, "frame_rate": float, "use_second_stage": bool}
        - {"action": "update_prompt", "prompt": str}
        - {"action": "stop"}

        Server sends:
        - {"type": "frame_chunk", "frames": [{"data": base64_jpeg, "index": int}, ...]}
        - {"type": "audio", "data": base64_wav, "sample_rate": int}
        - {"type": "segment_complete", "segment": int, "frames": int}
        - {"type": "error", "message": str}
        - {"type": "stopped"}
        """
        await websocket.accept()

        engine = OfficialLTX2Engine()
        current_prompt = None
        should_stop = False
        segment_count = 0
        last_segment_latent = None

        try:
            while True:
                # Wait for a message from the client
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    msg = json.loads(data)

                    if msg.get("action") == "stop":
                        should_stop = True
                        await websocket.send_json({"type": "stopped"})
                        break

                    elif msg.get("action") == "update_prompt":
                        current_prompt = msg.get("prompt", current_prompt)
                        await websocket.send_json({"type": "prompt_updated", "prompt": current_prompt})

                    elif msg.get("action") == "start":
                        current_prompt = msg.get("prompt", "A beautiful landscape")
                        seed = msg.get("seed", 42)
                        height = msg.get("height", 480)
                        width = msg.get("width", 832)
                        num_frames = msg.get("num_frames", 49)
                        frame_rate = msg.get("frame_rate", 24.0)
                        use_second_stage = msg.get("use_second_stage", False)
                        max_segments = msg.get("max_segments", 10)  # Limit segments

                        await websocket.send_json({"type": "started", "prompt": current_prompt})

                        # Generate segments continuously
                        segment_count = 0
                        while not should_stop and segment_count < max_segments:
                            segment_count += 1
                            seg_seed = seed + segment_count

                            # Check for stop/update messages during generation
                            try:
                                check_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                                check_msg = json.loads(check_data)
                                if check_msg.get("action") == "stop":
                                    should_stop = True
                                    break
                                elif check_msg.get("action") == "update_prompt":
                                    current_prompt = check_msg.get("prompt", current_prompt)
                                    await websocket.send_json({"type": "prompt_updated", "prompt": current_prompt})
                            except asyncio.TimeoutError:
                                pass

                            if should_stop:
                                break

                            # Generate segment
                            await websocket.send_json({
                                "type": "segment_start",
                                "segment": segment_count,
                                "prompt": current_prompt
                            })

                            # Run generation in thread pool to not block
                            import concurrent.futures
                            loop = asyncio.get_event_loop()

                            def run_generation():
                                return list(engine.generate_streaming(
                                    prompt=current_prompt,
                                    seed=seg_seed,
                                    height=height,
                                    width=width,
                                    num_frames=num_frames,
                                    frame_rate=frame_rate,
                                    use_second_stage=use_second_stage,
                                ))

                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                results = await loop.run_in_executor(pool, run_generation)

                            # Send all results
                            for item in results:
                                if should_stop:
                                    break
                                await websocket.send_json(item)

                                # Check for stop between frames
                                try:
                                    check_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                                    check_msg = json.loads(check_data)
                                    if check_msg.get("action") == "stop":
                                        should_stop = True
                                        break
                                    elif check_msg.get("action") == "update_prompt":
                                        current_prompt = check_msg.get("prompt", current_prompt)
                                except asyncio.TimeoutError:
                                    pass

                        if should_stop:
                            await websocket.send_json({"type": "stopped"})
                        else:
                            await websocket.send_json({"type": "complete", "segments": segment_count})

                except asyncio.TimeoutError:
                    # No message, continue waiting
                    continue

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except:
                pass
            raise

    @web_app.get("/stream", response_class=HTMLResponse)
    async def stream_ui():
        """Serve the streaming web UI."""
        return HTMLResponse(STREAMING_HTML)

    return web_app


# Streaming UI HTML with buffered playback
STREAMING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LTX-2 Streaming</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 50%, #0f0f1a 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #e94560);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .layout { display: grid; grid-template-columns: 350px 1fr; gap: 20px; }
        @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

        .controls {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
        }
        label { display: block; color: #888; font-size: 0.85rem; margin: 15px 0 5px; }
        label:first-child { margin-top: 0; }
        input, textarea, select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #333;
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 0.95rem;
        }
        textarea { min-height: 80px; resize: vertical; }
        input:focus, textarea:focus { outline: none; border-color: #e94560; }

        .row { display: flex; gap: 10px; }
        .row > * { flex: 1; }

        .checkbox-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
        }
        .checkbox-row input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #e94560;
        }
        .checkbox-row label { margin: 0; color: #fff; }

        .slider-container {
            margin-top: 15px;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            color: #888;
            font-size: 0.85rem;
        }
        .slider-value { color: #e94560; font-weight: 600; }
        input[type="range"] {
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: #333;
            border-radius: 3px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: #e94560;
            border-radius: 50%;
            cursor: pointer;
        }

        .btn {
            width: 100%;
            padding: 12px;
            margin-top: 20px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-start {
            color: white;
            flex: 1;
        }
        .btn-turbo {
            background: linear-gradient(135deg, #e94560, #ff6b35);
        }
        .btn-turbo:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(233,69,96,0.4); }
        .btn-hq {
            background: linear-gradient(135deg, #7b2fff, #00d4ff);
        }
        .btn-hq:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(123,47,255,0.4); }
        .btn-stop {
            background: #ff4444;
            color: white;
        }
        .btn-update {
            background: #333;
            color: white;
            margin-top: 10px;
        }

        .video-container {
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        #videoCanvas {
            width: 100%;
            height: auto;
            display: block;
            background: #000;
        }

        /* Drag and drop overlay */
        .drop-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(233, 69, 96, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 100;
            pointer-events: none;
        }
        .drop-overlay.active {
            display: flex;
        }
        .drop-overlay-text {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
        }

        /* Target image indicator */
        .target-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(74, 222, 128, 0.9);
            color: #000;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            display: none;
            z-index: 50;
        }
        .target-indicator.active {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .target-indicator img {
            width: 40px;
            height: 30px;
            object-fit: cover;
            border-radius: 4px;
        }
        .target-indicator .clear-btn {
            cursor: pointer;
            padding: 2px 6px;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
        }
        .target-indicator .clear-btn:hover {
            background: rgba(0,0,0,0.5);
        }
        .status-bar {
            padding: 10px 15px;
            background: rgba(0,0,0,0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            flex-wrap: wrap;
            gap: 10px;
        }
        .status { color: #888; }
        .status.connected { color: #00ff88; }
        .status.generating { color: #e94560; }
        .status.buffering { color: #ffaa00; }
        .stats { color: #666; }
        .buffer-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
        }
        .buffer-bar {
            width: 100px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }
        .buffer-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #00d4ff);
            transition: width 0.1s;
        }

        .log {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.8rem;
            max-height: 150px;
            overflow-y: auto;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 LTX-2 Real-Time Streaming</h1>
        <div class="layout">
            <div class="controls">
                <label>Prompt</label>
                <textarea id="prompt">A serene mountain landscape with flowing rivers and dramatic clouds, cinematic lighting, high quality</textarea>

                <div class="row">
                    <div>
                        <label>Width</label>
                        <input type="number" id="width" value="832" step="32">
                    </div>
                    <div>
                        <label>Height</label>
                        <input type="number" id="height" value="480" step="32">
                    </div>
                </div>

                <div class="row">
                    <div>
                        <label>Frames/Segment</label>
                        <input type="number" id="numFrames" value="49" min="17" max="97">
                    </div>
                    <div>
                        <label>Seed</label>
                        <input type="number" id="seed" value="42">
                    </div>
                </div>

                <div class="row">
                    <div>
                        <label>Max Segments</label>
                        <input type="number" id="maxSegments" value="10" min="1" max="50">
                    </div>
                </div>

                <div class="slider-container">
                    <div class="slider-label">
                        <span>Playback FPS</span>
                        <span class="slider-value" id="fpsValue">18</span>
                    </div>
                    <input type="range" id="playbackFps" min="6" max="30" value="12" oninput="updateFps(this.value)">
                </div>

                <div class="checkbox-row">
                    <input type="checkbox" id="useSecondStage">
                    <label for="useSecondStage">2x Upsampling (960x1664)</label>
                </div>

                <!-- First Segment Image Conditioning -->
                <div class="image-inputs" style="margin: 12px 0;">
                    <label style="font-size: 0.85rem; color: #aaa; margin-bottom: 8px; display: block;">First Segment Images (optional)</label>
                    <div style="display: flex; gap: 10px;">
                        <div class="image-input-box" style="flex: 1;">
                            <input type="file" id="startImageInput" accept="image/*" style="display: none;" onchange="handleStartImage(this)">
                            <div id="startImageBox" onclick="document.getElementById('startImageInput').click()"
                                 style="border: 2px dashed #444; border-radius: 8px; padding: 8px; text-align: center; cursor: pointer; min-height: 60px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative;">
                                <img id="startImagePreview" src="" style="max-width: 100%; max-height: 50px; display: none; border-radius: 4px;">
                                <span id="startImageLabel" style="font-size: 0.75rem; color: #888;">Start Frame</span>
                                <button id="clearStartImage" onclick="event.stopPropagation(); clearStartImage();" style="display: none; position: absolute; top: 2px; right: 2px; background: rgba(255,0,0,0.7); border: none; color: white; border-radius: 50%; width: 18px; height: 18px; cursor: pointer; font-size: 10px;">✕</button>
                            </div>
                        </div>
                        <div class="image-input-box" style="flex: 1;">
                            <input type="file" id="endImageInput" accept="image/*" style="display: none;" onchange="handleEndImage(this)">
                            <div id="endImageBox" onclick="document.getElementById('endImageInput').click()"
                                 style="border: 2px dashed #444; border-radius: 8px; padding: 8px; text-align: center; cursor: pointer; min-height: 60px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative;">
                                <img id="endImagePreview" src="" style="max-width: 100%; max-height: 50px; display: none; border-radius: 4px;">
                                <span id="endImageLabel" style="font-size: 0.75rem; color: #888;">End Frame</span>
                                <button id="clearEndImage" onclick="event.stopPropagation(); clearEndImage();" style="display: none; position: absolute; top: 2px; right: 2px; background: rgba(255,0,0,0.7); border: none; color: white; border-radius: 50%; width: 18px; height: 18px; cursor: pointer; font-size: 10px;">✕</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Target Frame Position -->
                <div class="slider-group" style="margin: 12px 0;">
                    <label style="font-size: 0.85rem; color: #aaa; margin-bottom: 4px; display: block;">
                        Target Frame Position: <span id="targetPositionLabel">End</span>
                    </label>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 0.7rem; color: #666;">Start</span>
                        <input type="range" id="targetFramePosition" min="0" max="1" step="0.1" value="1"
                               style="flex: 1; accent-color: #6366f1;"
                               oninput="updateTargetPositionLabel(this.value)">
                        <span style="font-size: 0.7rem; color: #666;">End</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #666; text-align: center; margin-top: 2px;">
                        Where in the next segment the dropped target image appears
                    </div>
                </div>

                <div class="btn-row" style="display: flex; gap: 10px;">
                    <button class="btn btn-start btn-turbo" id="startTurboBtn" onclick="startStream('turbo')">⚡ Start Turbo</button>
                    <button class="btn btn-start btn-hq" id="startHQBtn" onclick="startStream('hq')">✨ Start HQ</button>
                </div>
                <button class="btn btn-stop" id="stopBtn" onclick="stopStream()" style="display:none;">⏹ Stop</button>
                <button class="btn btn-update" id="updateBtn" onclick="updatePrompt()" style="display:none;">🔄 Update Prompt</button>

                <div class="log" id="log"></div>
            </div>

            <div class="video-container" id="videoContainer">
                <canvas id="videoCanvas" width="832" height="480"></canvas>
                <div class="drop-overlay" id="dropOverlay">
                    <div class="drop-overlay-text">Drop image to set as target frame</div>
                </div>
                <div class="target-indicator" id="targetIndicator">
                    <img id="targetPreview" src="" alt="Target">
                    <span>Target Set</span>
                    <span class="clear-btn" onclick="clearTargetImage()">✕</span>
                </div>
                <div class="status-bar">
                    <span class="status" id="status">Disconnected</span>
                    <div class="buffer-indicator">
                        <span>Buffer:</span>
                        <div class="buffer-bar"><div class="buffer-fill" id="bufferFill" style="width: 0%"></div></div>
                        <span id="bufferCount">0</span>
                    </div>
                    <span class="stats" id="stats">Frames: 0 | Played: 0</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket endpoints (injected by server)
        const WS_ENDPOINT_TURBO = "__WS_ENDPOINT_TURBO__";
        const WS_ENDPOINT_HQ = "__WS_ENDPOINT_HQ__";

        // WebSocket and state
        let ws = null;
        let isStreaming = false;

        // Frame buffer for smooth playback
        let frameBuffer = [];
        let playedFrames = 0;
        let receivedFrames = 0;
        let segmentCount = 0;

        // Playback control
        let playbackFps = 12;
        let generationFps = 24;  // Server-side generation rate
        let playbackInterval = null;
        let isPlaying = false;

        // Audio
        let audioContext = null;
        let audioQueue = [];
        let audioSources = [];  // Track active audio sources for rate changes
        let nextAudioTime = 0;

        // Target image (drag-drop during streaming)
        let targetImageData = null;

        // First segment images (set before starting)
        let startImageData = null;
        let endImageData = null;

        // Canvas
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');

        // UI elements
        const statusEl = document.getElementById('status');
        const statsEl = document.getElementById('stats');
        const logEl = document.getElementById('log');
        const bufferFillEl = document.getElementById('bufferFill');
        const bufferCountEl = document.getElementById('bufferCount');
        const videoContainer = document.getElementById('videoContainer');
        const dropOverlay = document.getElementById('dropOverlay');
        const targetIndicator = document.getElementById('targetIndicator');
        const targetPreview = document.getElementById('targetPreview');

        function log(msg) {
            const time = new Date().toLocaleTimeString();
            logEl.innerHTML = `[${time}] ${msg}<br>` + logEl.innerHTML;
            if (logEl.innerHTML.length > 5000) {
                logEl.innerHTML = logEl.innerHTML.substring(0, 5000);
            }
        }

        function updateFps(value) {
            playbackFps = parseInt(value);
            document.getElementById('fpsValue').textContent = value;
            if (playbackInterval) {
                clearInterval(playbackInterval);
                playbackInterval = setInterval(playFrame, 1000 / playbackFps);
            }
            // Note: Audio is pre-stretched for pitch preservation, so FPS changes
            // only affect new audio chunks. Currently playing audio continues at
            // its original stretched rate. This is a trade-off for pitch preservation.
            log(`FPS changed to ${value} - new audio will be time-stretched accordingly`);
        }

        function updateStatus(text, className) {
            statusEl.textContent = text;
            statusEl.className = 'status ' + (className || '');
        }

        function updateStats() {
            statsEl.textContent = `Frames: ${receivedFrames} | Played: ${playedFrames}`;
            const bufferSize = frameBuffer.length;
            bufferCountEl.textContent = bufferSize;
            // Buffer bar: 0-100 frames mapped to 0-100%
            const bufferPercent = Math.min(100, (bufferSize / 100) * 100);
            bufferFillEl.style.width = bufferPercent + '%';
        }

        function playFrame() {
            if (frameBuffer.length > 0) {
                const frame = frameBuffer.shift();
                const frameData = frame.data;
                const segment = frame.segment;
                const frameIndex = frame.index;

                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    // Draw debug overlay with segment/frame info
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                    ctx.fillRect(10, 10, 180, 30);
                    ctx.fillStyle = '#00ff00';
                    ctx.font = 'bold 16px monospace';
                    ctx.fillText(`Seg: ${segment} | Frame: ${frameIndex}`, 20, 30);
                };
                img.src = 'data:image/jpeg;base64,' + frameData;
                playedFrames++;
                updateStats();

                if (frameBuffer.length > 0) {
                    updateStatus('Playing', 'generating');
                }
            } else if (isStreaming) {
                updateStatus('Buffering...', 'buffering');
            }
        }

        function startPlayback() {
            if (!isPlaying) {
                isPlaying = true;
                playbackInterval = setInterval(playFrame, 1000 / playbackFps);
                log('Playback started at ' + playbackFps + ' FPS');
            }
        }

        function stopPlayback() {
            if (playbackInterval) {
                clearInterval(playbackInterval);
                playbackInterval = null;
            }
            isPlaying = false;
        }

        // Time-stretch audio buffer using granular synthesis (preserves pitch)
        function timeStretchBuffer(ctx, buffer, rate) {
            if (Math.abs(rate - 1.0) < 0.01) {
                return buffer; // No stretching needed
            }

            const grainSize = 0.03; // 30ms grains
            const overlap = 0.6; // 60% overlap for smoother output

            const inputLength = buffer.length;
            const outputLength = Math.floor(inputLength / rate);
            const numChannels = buffer.numberOfChannels;
            const sampleRate = buffer.sampleRate;

            const outputBuffer = ctx.createBuffer(numChannels, outputLength, sampleRate);

            const grainSamples = Math.floor(grainSize * sampleRate);
            const hopIn = Math.floor(grainSamples * (1 - overlap));
            const hopOut = Math.floor(hopIn / rate);

            for (let ch = 0; ch < numChannels; ch++) {
                const input = buffer.getChannelData(ch);
                const output = outputBuffer.getChannelData(ch);

                // Fill with zeros first
                output.fill(0);

                let inPos = 0;
                let outPos = 0;

                while (inPos < inputLength - grainSamples && outPos < outputLength - grainSamples) {
                    // Copy grain with Hann window for smooth crossfade
                    for (let i = 0; i < grainSamples && outPos + i < outputLength; i++) {
                        const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / grainSamples));
                        output[outPos + i] += input[inPos + i] * window;
                    }

                    inPos += hopIn;
                    outPos += hopOut;
                }
            }

            // Normalize to prevent clipping
            for (let ch = 0; ch < numChannels; ch++) {
                const output = outputBuffer.getChannelData(ch);
                let maxVal = 0;
                for (let i = 0; i < output.length; i++) {
                    maxVal = Math.max(maxVal, Math.abs(output[i]));
                }
                if (maxVal > 1.0) {
                    const scale = 0.95 / maxVal;
                    for (let i = 0; i < output.length; i++) {
                        output[i] *= scale;
                    }
                }
            }

            return outputBuffer;
        }

        function playAudioChunk(base64Audio) {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                nextAudioTime = audioContext.currentTime;
            }

            // Decode base64 to ArrayBuffer
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            audioContext.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
                // Calculate playback rate
                const audioRate = playbackFps / generationFps;

                // Time-stretch the buffer to preserve pitch
                const stretchedBuffer = timeStretchBuffer(audioContext, buffer, audioRate);

                const source = audioContext.createBufferSource();
                source.buffer = stretchedBuffer;
                source.connect(audioContext.destination);

                // Play at normal rate (stretching already adjusted duration)
                source.playbackRate.value = 1.0;

                // Track this source
                audioSources.push(source);
                source.onended = () => {
                    const idx = audioSources.indexOf(source);
                    if (idx > -1) audioSources.splice(idx, 1);
                };

                // Schedule audio to play at the right time
                const startTime = Math.max(audioContext.currentTime, nextAudioTime);
                source.start(startTime);
                // Duration is now the stretched buffer duration
                nextAudioTime = startTime + stretchedBuffer.duration;

                log(`Audio: time-stretched ${audioRate.toFixed(2)}x (pitch preserved)`);
            }, (err) => {
                console.error('Audio decode error:', err);
            });
        }

        // ============ Target Image / Drag-and-Drop ============

        // Drag and drop handlers
        videoContainer.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropOverlay.classList.add('active');
        });

        videoContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        videoContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Only hide if leaving the container entirely
            if (!videoContainer.contains(e.relatedTarget)) {
                dropOverlay.classList.remove('active');
            }
        });

        videoContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropOverlay.classList.remove('active');

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                handleTargetImage(files[0]);
            }
        });

        function handleTargetImage(file) {
            // Resize image on client side to avoid large WebSocket messages
            const targetWidth = parseInt(document.getElementById('width').value) || 832;
            const targetHeight = parseInt(document.getElementById('height').value) || 480;

            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    // Create canvas at target resolution
                    const canvas = document.createElement('canvas');
                    canvas.width = targetWidth;
                    canvas.height = targetHeight;
                    const ctx = canvas.getContext('2d');

                    // Draw image scaled to fit
                    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

                    // Export as JPEG with good quality (smaller than PNG)
                    const resizedImageData = canvas.toDataURL('image/jpeg', 0.9);

                    targetImageData = resizedImageData;

                    // Show preview
                    targetPreview.src = resizedImageData;
                    targetIndicator.classList.add('active');

                    const sizeKB = Math.round(resizedImageData.length / 1024);
                    log(`Image resized to ${targetWidth}x${targetHeight} (${sizeKB}KB)`);

                    // Send to server if connected
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        sendTargetImage(resizedImageData);
                    } else {
                        log('Target image staged (will send when streaming starts)');
                    }
                };
                img.onerror = () => {
                    log('Failed to load image');
                };
                img.src = e.target.result;
            };
            reader.onerror = () => {
                log('Failed to read image file');
            };
            reader.readAsDataURL(file);
        }

        function sendTargetImage(imageData) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                try {
                    const height = parseInt(document.getElementById('height').value);
                    const width = parseInt(document.getElementById('width').value);
                    const position = parseFloat(document.getElementById('targetFramePosition').value);
                    const message = JSON.stringify({
                        action: 'set_target_image',
                        image: imageData,
                        height: height,
                        width: width,
                        position: position
                    });
                    const sizeMB = (message.length / (1024 * 1024)).toFixed(2);
                    const posLabel = position <= 0.33 ? 'start' : (position <= 0.66 ? 'middle' : 'end');
                    log(`Sending target image (${sizeMB}MB) at ${posLabel} of segment...`);
                    ws.send(message);
                } catch (err) {
                    log('Error sending target image: ' + err.message);
                    console.error('Send error:', err);
                }
            }
        }

        function updateTargetPositionLabel(value) {
            const val = parseFloat(value);
            let label;
            if (val <= 0.15) {
                label = 'Start (0%)';
            } else if (val <= 0.35) {
                label = 'Early (25%)';
            } else if (val <= 0.65) {
                label = 'Middle (50%)';
            } else if (val <= 0.85) {
                label = 'Late (75%)';
            } else {
                label = 'End (100%)';
            }
            document.getElementById('targetPositionLabel').textContent = label;
        }

        function clearTargetImage() {
            targetImageData = null;
            targetPreview.src = '';
            targetIndicator.classList.remove('active');

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'clear_target_image' }));
                log('Target image cleared');
            }
        }

        // ============ Start/End Image Functions ============

        function handleStartImage(input) {
            if (input.files && input.files[0]) {
                const file = input.files[0];
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = new Image();
                    img.onload = () => {
                        // Resize to match generation resolution
                        const targetWidth = parseInt(document.getElementById('width').value);
                        const targetHeight = parseInt(document.getElementById('height').value);
                        const canvas = document.createElement('canvas');
                        canvas.width = targetWidth;
                        canvas.height = targetHeight;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
                        startImageData = canvas.toDataURL('image/jpeg', 0.9);

                        // Show preview
                        document.getElementById('startImagePreview').src = startImageData;
                        document.getElementById('startImagePreview').style.display = 'block';
                        document.getElementById('startImageLabel').style.display = 'none';
                        document.getElementById('clearStartImage').style.display = 'block';
                        document.getElementById('startImageBox').style.borderColor = '#4a9eff';
                        log('Start image set');
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        }

        function clearStartImage() {
            startImageData = null;
            document.getElementById('startImageInput').value = '';
            document.getElementById('startImagePreview').src = '';
            document.getElementById('startImagePreview').style.display = 'none';
            document.getElementById('startImageLabel').style.display = 'block';
            document.getElementById('clearStartImage').style.display = 'none';
            document.getElementById('startImageBox').style.borderColor = '#444';
            log('Start image cleared');
        }

        function handleEndImage(input) {
            if (input.files && input.files[0]) {
                const file = input.files[0];
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = new Image();
                    img.onload = () => {
                        // Resize to match generation resolution
                        const targetWidth = parseInt(document.getElementById('width').value);
                        const targetHeight = parseInt(document.getElementById('height').value);
                        const canvas = document.createElement('canvas');
                        canvas.width = targetWidth;
                        canvas.height = targetHeight;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
                        endImageData = canvas.toDataURL('image/jpeg', 0.9);

                        // Show preview
                        document.getElementById('endImagePreview').src = endImageData;
                        document.getElementById('endImagePreview').style.display = 'block';
                        document.getElementById('endImageLabel').style.display = 'none';
                        document.getElementById('clearEndImage').style.display = 'block';
                        document.getElementById('endImageBox').style.borderColor = '#4a9eff';
                        log('End image set');
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        }

        function clearEndImage() {
            endImageData = null;
            document.getElementById('endImageInput').value = '';
            document.getElementById('endImagePreview').src = '';
            document.getElementById('endImagePreview').style.display = 'none';
            document.getElementById('endImageLabel').style.display = 'block';
            document.getElementById('clearEndImage').style.display = 'none';
            document.getElementById('endImageBox').style.borderColor = '#444';
            log('End image cleared');
        }

        // ============ Streaming Functions ============

        let currentMode = 'turbo';  // Track current streaming mode

        function startStream(mode = 'turbo') {
            currentMode = mode;
            // Select WebSocket URL based on mode
            const wsUrl = mode === 'hq' ? WS_ENDPOINT_HQ : WS_ENDPOINT_TURBO;

            log(`Connecting to ${mode.toUpperCase()} endpoint...`);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                log('Connected!');
                updateStatus('Connected', 'connected');

                // Reset state
                frameBuffer = [];
                playedFrames = 0;
                receivedFrames = 0;
                segmentCount = 0;
                isStreaming = true;

                // Reset audio state
                audioSources = [];
                nextAudioTime = 0;
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }

                // Send start command
                generationFps = 24.0;  // Server-side generation rate
                const config = {
                    action: 'start',
                    prompt: document.getElementById('prompt').value,
                    seed: parseInt(document.getElementById('seed').value),
                    height: parseInt(document.getElementById('height').value),
                    width: parseInt(document.getElementById('width').value),
                    num_frames: parseInt(document.getElementById('numFrames').value),
                    frame_rate: generationFps,
                    use_second_stage: document.getElementById('useSecondStage').checked,
                    max_segments: parseInt(document.getElementById('maxSegments').value),
                };

                // Add start/end images for first segment if set
                if (startImageData) {
                    config.start_image = startImageData;
                    log('Including start image for first segment');
                }
                if (endImageData) {
                    config.end_image = endImageData;
                    log('Including end image for first segment');
                }

                // Update canvas size
                canvas.width = config.use_second_stage ? config.width * 2 : config.width;
                canvas.height = config.use_second_stage ? config.height * 2 : config.height;

                ws.send(JSON.stringify(config));
                log('Generation started: ' + config.prompt.substring(0, 40) + '...');

                // Send staged target image if any (for subsequent segments)
                if (targetImageData) {
                    sendTargetImage(targetImageData);
                }

                document.getElementById('startTurboBtn').style.display = 'none';
                document.getElementById('startHQBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'block';
                document.getElementById('updateBtn').style.display = 'block';

                // Start playback
                startPlayback();
                updateStats();
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);

                if (msg.type === 'frame_chunk') {
                    // Add all frames from chunk to buffer
                    for (const frame of msg.frames) {
                        frameBuffer.push({
                            data: frame.data,
                            segment: msg.segment || 0,
                            index: frame.index || 0
                        });
                        receivedFrames++;
                    }
                    updateStats();
                }
                else if (msg.type === 'audio') {
                    log('Audio chunk received');
                    playAudioChunk(msg.data);
                }
                else if (msg.type === 'started') {
                    if (msg.start_image_set) log('Start image encoded successfully');
                    if (msg.start_image_error) log('Start image error: ' + msg.start_image_error);
                    if (msg.end_image_set) log('End image encoded successfully');
                    if (msg.end_image_error) log('End image error: ' + msg.end_image_error);
                }
                else if (msg.type === 'segment_start') {
                    log(`Segment ${msg.segment}: ${msg.prompt.substring(0, 30)}...`);
                }
                else if (msg.type === 'segment_complete') {
                    segmentCount = msg.segment;
                    log(`Segment ${msg.segment} done (${msg.frames} frames)`);
                }
                else if (msg.type === 'prompt_updated') {
                    log('Prompt updated!');
                }
                else if (msg.type === 'target_image_set') {
                    if (msg.success) {
                        log('Target image set - will apply to next segment');
                    } else {
                        log('Failed to set target image: ' + (msg.error || 'Unknown error'));
                        clearTargetImage();
                    }
                }
                else if (msg.type === 'target_image_used') {
                    log('Target image applied to segment');
                    // Clear the indicator after use
                    targetImageData = null;
                    targetPreview.src = '';
                    targetIndicator.classList.remove('active');
                }
                else if (msg.type === 'target_image_cleared') {
                    log('Target image cleared');
                }
                else if (msg.type === 'start_image_used') {
                    log('Start image applied to first segment');
                    // Clear the start image UI
                    clearStartImage();
                }
                else if (msg.type === 'complete') {
                    log(`Complete! ${msg.segments} segments`);
                    isStreaming = false;
                    // Keep playing until buffer is empty
                }
                else if (msg.type === 'stopped') {
                    log('Stopped');
                    isStreaming = false;
                    stopPlayback();
                    updateStatus('Stopped', '');
                    resetButtons();
                }
                else if (msg.type === 'error') {
                    log('Error: ' + msg.message);
                    isStreaming = false;
                    stopPlayback();
                    updateStatus('Error', '');
                    resetButtons();
                }
            };

            ws.onerror = (error) => {
                log('WebSocket error');
                isStreaming = false;
                stopPlayback();
                updateStatus('Error', '');
                resetButtons();
            };

            ws.onclose = () => {
                log('Disconnected');
                isStreaming = false;
                // Don't stop playback immediately - let buffer drain
                setTimeout(() => {
                    if (frameBuffer.length === 0) {
                        stopPlayback();
                        updateStatus('Disconnected', '');
                        resetButtons();
                    }
                }, 1000);
            };
        }

        function stopStream() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'stop'}));
                log('Stop requested');
            }
            isStreaming = false;
        }

        function updatePrompt() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const newPrompt = document.getElementById('prompt').value;
                ws.send(JSON.stringify({action: 'update_prompt', prompt: newPrompt}));
                log('Updating prompt...');
            }
        }

        function resetButtons() {
            document.getElementById('startTurboBtn').style.display = 'block';
            document.getElementById('startHQBtn').style.display = 'block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('updateBtn').style.display = 'none';
            if (ws) {
                ws.close();
                ws = null;
            }
        }
    </script>
</body>
</html>
"""


# ============================================================================
# Minimal Streaming UI
# ============================================================================

MINIMAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LTX-2 Video Stream</title>
    <style>
        :root {
            --bg-primary: #000;
            --bg-secondary: rgba(30, 30, 30, 0.9);
            --bg-tertiary: rgba(255, 255, 255, 0.1);
            --bg-hover: rgba(255, 255, 255, 0.2);
            --text-primary: #fff;
            --text-secondary: #888;
            --text-muted: #666;
            --border-color: rgba(255, 255, 255, 0.1);
            --accent: #6366f1;
            --canvas-bg: #000;
        }

        body.light-mode {
            --bg-primary: #f5f5f5;
            --bg-secondary: rgba(255, 255, 255, 0.95);
            --bg-tertiary: rgba(0, 0, 0, 0.05);
            --bg-hover: rgba(0, 0, 0, 0.1);
            --text-primary: #1a1a1a;
            --text-secondary: #666;
            --text-muted: #999;
            --border-color: rgba(0, 0, 0, 0.1);
            --canvas-bg: #e0e0e0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }

        /* Full-screen video container */
        .video-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--canvas-bg);
        }

        #videoCanvas {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        /* Drop overlay for drag-and-drop */
        .drop-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(99, 102, 241, 0.3);
            border: 4px dashed #6366f1;
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 100;
            pointer-events: none;
        }

        .drop-overlay.active {
            display: flex;
        }

        .drop-overlay-text {
            font-size: 2rem;
            color: #fff;
            text-shadow: 0 2px 8px rgba(0,0,0,0.5);
        }

        /* Video controls bar */
        .video-controls {
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 20px;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            border-radius: 30px;
            z-index: 50;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .video-wrapper:hover .video-controls,
        .video-controls:hover,
        .video-controls.visible {
            opacity: 1;
        }

        .ctrl-btn {
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 50%;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }

        .ctrl-btn:hover {
            background: var(--bg-hover);
        }

        .ctrl-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .ctrl-btn.active {
            background: #6366f1;
        }

        /* Seek bar */
        .seek-container {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
            min-width: 200px;
            max-width: 400px;
        }

        .seek-bar {
            flex: 1;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: var(--bg-hover);
            border-radius: 2px;
            cursor: pointer;
        }

        .seek-bar::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--text-primary);
            cursor: pointer;
        }

        .seek-bar::-moz-range-track {
            background: var(--bg-hover);
            border-radius: 2px;
            height: 4px;
        }

        .seek-bar::-moz-range-thumb {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--text-primary);
            border: none;
            cursor: pointer;
        }

        .time-display {
            font-size: 12px;
            color: rgba(255,255,255,0.7);
            min-width: 80px;
            text-align: center;
        }

        /* Live indicator */
        .live-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(239, 68, 68, 0.8);
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .live-indicator.paused {
            background: rgba(100, 100, 100, 0.8);
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background: #fff;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        .live-indicator.paused .live-dot {
            animation: none;
            background: #888;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Bottom prompt bar */
        .prompt-bar {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 700px;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 8px 8px 16px;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            border-radius: 28px;
            border: 1px solid var(--border-color);
            z-index: 60;
        }

        .prompt-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 15px;
            outline: none;
            padding: 8px 0;
        }

        .prompt-input::placeholder {
            color: var(--text-secondary);
        }

        .prompt-btn {
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 50%;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
            flex-shrink: 0;
        }

        .prompt-btn:hover {
            background: var(--bg-hover);
        }

        .prompt-btn.primary {
            background: #6366f1;
        }

        .prompt-btn.primary:hover {
            background: #5558e3;
        }

        .prompt-btn.primary:disabled {
            background: #444;
            cursor: not-allowed;
        }

        /* Image preview in prompt bar */
        .attached-image {
            position: relative;
            width: 44px;
            height: 44px;
            border-radius: 8px;
            overflow: hidden;
            flex-shrink: 0;
        }

        .attached-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .attached-image .remove-btn {
            position: absolute;
            top: -4px;
            right: -4px;
            width: 18px;
            height: 18px;
            background: rgba(239, 68, 68, 0.9);
            border: none;
            border-radius: 50%;
            color: #fff;
            font-size: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Settings panel */
        .settings-panel {
            position: fixed;
            top: 70px;
            right: 20px;
            width: 280px;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            padding: 16px;
            z-index: 70;
            display: none;
        }

        .settings-panel.open {
            display: block;
        }

        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }

        .settings-title {
            font-size: 14px;
            font-weight: 600;
        }

        .settings-close {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 18px;
            cursor: pointer;
        }

        .setting-group {
            margin-bottom: 14px;
        }

        .setting-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
        }

        .setting-value {
            color: var(--text-primary);
        }

        .setting-input {
            width: 100%;
            padding: 8px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
        }

        .setting-input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .setting-slider {
            width: 100%;
            -webkit-appearance: none;
            appearance: none;
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
        }

        .setting-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #6366f1;
            cursor: pointer;
        }

        .setting-row {
            display: flex;
            gap: 10px;
        }

        .setting-row .setting-group {
            flex: 1;
        }

        /* Settings gear button */
        /* Top right controls container */
        .top-right-controls {
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 65;
        }

        .settings-btn {
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 50%;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            color: var(--text-primary);
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border-color);
            transition: background 0.2s;
        }

        .settings-btn:hover {
            background: var(--bg-hover);
        }

        .settings-btn.hidden {
            display: none;
        }

        /* Theme toggle */
        .theme-toggle {
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 50%;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            color: var(--text-primary);
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border-color);
            transition: background 0.2s;
        }

        .theme-toggle:hover {
            background: var(--bg-hover);
        }

        /* Status indicator */
        .status-indicator {
            position: fixed;
            top: 20px;
            left: 20px;
            padding: 8px 14px;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-secondary);
            z-index: 50;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
        }

        .status-dot.connected {
            background: #22c55e;
        }

        .status-dot.connecting {
            background: #f59e0b;
            animation: pulse 1s infinite;
        }

        /* Hidden file input */
        #imageInput {
            display: none;
        }

        /* Toast notifications */
        .toast {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 20px;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            font-size: 13px;
            z-index: 200;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }

        .toast.visible {
            opacity: 1;
        }

        /* Placeholder when no video */
        .placeholder {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            z-index: 5;
            pointer-events: none;
        }

        .placeholder.hidden {
            display: none;
        }

        .placeholder-title {
            font-size: 72px;
            font-weight: 300;
            letter-spacing: -2px;
            margin-bottom: 8px;
            color: var(--text-primary);
            opacity: 0.9;
        }

        .placeholder-subtitle {
            font-size: 16px;
            margin-bottom: 32px;
            opacity: 0.6;
        }

        .placeholder-text {
            font-size: 16px;
            margin-bottom: 8px;
        }

        .placeholder-hint {
            font-size: 13px;
            color: #555;
        }
    </style>
</head>
<body>
    <!-- Video display -->
    <div class="video-wrapper" id="videoWrapper">
        <canvas id="videoCanvas" width="832" height="480"></canvas>
        <div class="drop-overlay" id="dropOverlay">
            <div class="drop-overlay-text">Drop image or audio to condition video</div>
        </div>
    </div>

    <!-- Placeholder -->
    <div class="placeholder" id="placeholder">
        <div class="placeholder-title">vidi</div>
        <div class="placeholder-subtitle">a realtime interactive video stream</div>
        <div class="placeholder-text">Enter a prompt to start generating</div>
        <div class="placeholder-hint">Or drop image/audio to condition the video</div>
    </div>

    <!-- Status indicator -->
    <div class="status-indicator" id="statusIndicator">
        <div class="status-dot" id="statusDot"></div>
        <span id="statusText">Ready</span>
    </div>

    <!-- Top right controls -->
    <div class="top-right-controls">
        <button class="settings-btn" id="settingsBtn" title="Settings">⚙</button>
        <button class="theme-toggle" id="themeToggle" title="Toggle light/dark mode">🌙</button>
    </div>

    <!-- Video controls -->
    <div class="video-controls" id="videoControls">
        <button class="ctrl-btn" id="playPauseBtn" title="Play/Pause" disabled>▶</button>
        <button class="ctrl-btn" id="stopBtn" title="Stop" disabled>⏹</button>
        <div class="seek-container">
            <input type="range" class="seek-bar" id="seekBar" min="0" max="100" value="100" disabled>
            <div class="time-display" id="timeDisplay">0 / 0</div>
        </div>
        <button class="ctrl-btn" id="liveBtn" title="Jump to Live" disabled>⏭</button>
        <button class="ctrl-btn" id="resetBtn" title="Reset History" disabled>🔄</button>
        <button class="ctrl-btn" id="muteBtn" title="Mute/Unmute Audio">🔊</button>
        <div class="live-indicator" id="liveIndicator">
            <div class="live-dot"></div>
            <span>LIVE</span>
        </div>
    </div>

    <!-- Prompt bar -->
    <div class="prompt-bar" id="promptBar">
        <input type="file" id="imageInput" accept="image/*">
        <button class="prompt-btn" id="attachBtn" title="Attach image">📎</button>
        <div class="attached-image" id="attachedImage" style="display: none;">
            <img id="attachedPreview" src="">
            <button class="remove-btn" onclick="removeAttachment()">✕</button>
        </div>
        <input type="text" class="prompt-input" id="promptInput" placeholder="Describe what you want to see...">
        <button class="prompt-btn primary" id="sendBtn" title="Send">➤</button>
    </div>

    <!-- Settings panel -->
    <div class="settings-panel" id="settingsPanel">
        <div class="settings-header">
            <span class="settings-title">Settings</span>
            <button class="settings-close" onclick="toggleSettings()">✕</button>
        </div>

        <div class="setting-row">
            <div class="setting-group">
                <label class="setting-label">Width</label>
                <input type="number" class="setting-input" id="widthInput" value="832" step="32" min="256" max="1280">
            </div>
            <div class="setting-group">
                <label class="setting-label">Height</label>
                <input type="number" class="setting-input" id="heightInput" value="480" step="32" min="256" max="720">
            </div>
        </div>

        <div class="setting-row">
            <div class="setting-group">
                <label class="setting-label">Seed</label>
                <input type="number" class="setting-input" id="seedInput" value="42" min="0">
            </div>
            <div class="setting-group">
                <label class="setting-label">Frames/Segment</label>
                <input type="number" class="setting-input" id="numFramesInput" value="49" min="49" max="196" step="8">
            </div>
        </div>

        <div class="setting-group">
            <label class="setting-label">
                <span>Playback FPS</span>
                <span class="setting-value" id="fpsValue">12</span>
            </label>
            <input type="range" class="setting-slider" id="fpsSlider" min="1" max="30" value="12">
        </div>

        <div class="setting-group">
            <label class="setting-label">
                <span>Target Frame Position</span>
                <span class="setting-value" id="targetPosValue">End</span>
            </label>
            <input type="range" class="setting-slider" id="targetPosSlider" min="0" max="1" step="0.1" value="1">
            <div style="font-size: 10px; color: #666; margin-top: 4px;">Where dropped images appear in segment</div>
        </div>

        <div class="setting-group">
            <label class="setting-label">
                <span>Audio Noise Scale</span>
                <span class="setting-value" id="audioNoiseScaleValue">0.0</span>
            </label>
            <input type="range" class="setting-slider" id="audioNoiseScaleSlider" min="0" max="1" step="0.1" value="0">
            <div style="font-size: 10px; color: #666; margin-top: 4px;">0=pass-through (clean), 1=full denoise (more artifacts)</div>
        </div>
    </div>

    <!-- Toast -->
    <div class="toast" id="toast"></div>

    <script>
        // WebSocket endpoint
        const WS_ENDPOINT = "__WS_ENDPOINT_TURBO__";

        // State
        let ws = null;
        let isStreaming = false;
        let isPaused = false;
        let isLive = true;

        // Frame buffer and playback
        let frameHistory = [];      // All received frames (for seeking)
        let frameBuffer = [];       // Queue of frames waiting to be played
        let currentFrameIndex = 0;  // Current position in history
        let playbackInterval = null;
        let playbackFps = 12;
        let lastFrameTime = 0;

        // Audio playback state
        let audioContext = null;
        let audioQueue = [];          // Queue of audio buffers to play
        let nextAudioTime = 0;        // When the next audio chunk should start
        let audioSources = [];        // Active audio source nodes
        let isAudioMuted = false;
        const generationFps = 24;     // Server generates at 24fps

        // Audio history for playback
        let audioHistory = [];        // Array of {startFrame, endFrame, buffer} for each segment
        let currentAudioSegment = -1; // Currently playing audio segment index
        let segmentFrameCount = 0;    // Frames received in current segment

        // Pending data
        let pendingImage = null;
        let pendingAudio = null;
        let pendingAudioName = null;
        let currentPrompt = "";

        // Elements
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        const promptInput = document.getElementById('promptInput');
        const sendBtn = document.getElementById('sendBtn');
        const attachBtn = document.getElementById('attachBtn');
        const imageInput = document.getElementById('imageInput');
        const attachedImage = document.getElementById('attachedImage');
        const attachedPreview = document.getElementById('attachedPreview');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const stopBtn = document.getElementById('stopBtn');
        const seekBar = document.getElementById('seekBar');
        const timeDisplay = document.getElementById('timeDisplay');
        const liveBtn = document.getElementById('liveBtn');
        const resetBtn = document.getElementById('resetBtn');
        const liveIndicator = document.getElementById('liveIndicator');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const placeholder = document.getElementById('placeholder');
        const videoControls = document.getElementById('videoControls');
        const settingsPanel = document.getElementById('settingsPanel');
        const settingsBtn = document.getElementById('settingsBtn');
        const fpsSlider = document.getElementById('fpsSlider');
        const fpsValue = document.getElementById('fpsValue');
        const targetPosSlider = document.getElementById('targetPosSlider');
        const targetPosValue = document.getElementById('targetPosValue');
        const audioNoiseScaleSlider = document.getElementById('audioNoiseScaleSlider');
        const audioNoiseScaleValue = document.getElementById('audioNoiseScaleValue');
        const dropOverlay = document.getElementById('dropOverlay');
        const themeToggle = document.getElementById('themeToggle');
        const muteBtn = document.getElementById('muteBtn');

        // Initialize
        function init() {
            // Load saved theme
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'light') {
                document.body.classList.add('light-mode');
                themeToggle.textContent = '☀️';
            }

            // Theme toggle
            themeToggle.addEventListener('click', () => {
                document.body.classList.toggle('light-mode');
                const isLight = document.body.classList.contains('light-mode');
                themeToggle.textContent = isLight ? '☀️' : '🌙';
                localStorage.setItem('theme', isLight ? 'light' : 'dark');
            });

            // Event listeners
            sendBtn.addEventListener('click', handleSend);
            promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                }
            });

            attachBtn.addEventListener('click', () => imageInput.click());
            imageInput.addEventListener('change', handleImageSelect);

            playPauseBtn.addEventListener('click', togglePlayPause);
            stopBtn.addEventListener('click', stopStream);
            liveBtn.addEventListener('click', jumpToLive);
            resetBtn.addEventListener('click', resetHistory);
            seekBar.addEventListener('input', handleSeek);
            muteBtn.addEventListener('click', toggleMute);

            settingsBtn.addEventListener('click', toggleSettings);

            fpsSlider.addEventListener('input', (e) => {
                playbackFps = parseInt(e.target.value);
                fpsValue.textContent = playbackFps;
                // Restart playback at new FPS
                if (playbackInterval) {
                    startPlayback();
                }
            });

            targetPosSlider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                if (val <= 0.15) targetPosValue.textContent = 'Start';
                else if (val <= 0.35) targetPosValue.textContent = 'Early';
                else if (val <= 0.65) targetPosValue.textContent = 'Middle';
                else if (val <= 0.85) targetPosValue.textContent = 'Late';
                else targetPosValue.textContent = 'End';
            });

            audioNoiseScaleSlider.addEventListener('input', (e) => {
                audioNoiseScaleValue.textContent = parseFloat(e.target.value).toFixed(1);
            });

            // Drag and drop
            setupDragDrop();

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.target === promptInput) return;
                if (e.code === 'Space') {
                    e.preventDefault();
                    togglePlayPause();
                }
            });
        }

        // Drag and drop setup
        function setupDragDrop() {
            const wrapper = document.getElementById('videoWrapper');

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
                wrapper.addEventListener(event, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });

            wrapper.addEventListener('dragenter', () => {
                dropOverlay.classList.add('active');
            });

            wrapper.addEventListener('dragleave', (e) => {
                if (!wrapper.contains(e.relatedTarget)) {
                    dropOverlay.classList.remove('active');
                }
            });

            wrapper.addEventListener('drop', (e) => {
                dropOverlay.classList.remove('active');
                const file = e.dataTransfer.files[0];
                if (file) {
                    if (file.type.startsWith('image/')) {
                        processImageFile(file);
                    } else if (file.type.startsWith('audio/') || file.name.match(/\.(mp3|wav|ogg|m4a|flac|aac)$/i)) {
                        processAudioFile(file);
                    }
                }
            });
        }

        // Image handling
        function handleImageSelect(e) {
            const file = e.target.files[0];
            if (file) {
                processImageFile(file);
            }
        }

        function processImageFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    // Resize if needed
                    const maxDim = 1024;
                    let w = img.width, h = img.height;
                    if (w > maxDim || h > maxDim) {
                        if (w > h) { h = h * maxDim / w; w = maxDim; }
                        else { w = w * maxDim / h; h = maxDim; }
                    }

                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = w;
                    tempCanvas.height = h;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(img, 0, 0, w, h);

                    pendingImage = tempCanvas.toDataURL('image/jpeg', 0.9);
                    attachedPreview.src = pendingImage;
                    attachedImage.style.display = 'block';

                    if (isStreaming && ws && ws.readyState === WebSocket.OPEN) {
                        // If streaming, send as target image for next segment
                        sendTargetImage(pendingImage);
                        showToast('Target image set for next segment');
                    } else {
                        // If not streaming, render to canvas as start image preview
                        renderStartImagePreview(img);
                        showToast('Start image set - enter a prompt to begin');
                    }
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function removeAttachment() {
            pendingImage = null;
            attachedImage.style.display = 'none';
            attachedPreview.src = '';
            imageInput.value = '';
        }

        // Audio handling
        function processAudioFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                // Store as base64 (remove data URL prefix)
                const base64 = e.target.result.split(',')[1];
                pendingAudio = base64;
                pendingAudioName = file.name;

                if (isStreaming && ws && ws.readyState === WebSocket.OPEN) {
                    // If streaming, send as audio conditioning for upcoming segments
                    sendAudio(pendingAudio);
                    showToast('Audio queued for upcoming segments');
                    pendingAudio = null;
                    pendingAudioName = null;
                } else {
                    // If not streaming, store for start message
                    showToast('Audio set: ' + file.name.substring(0, 20) + ' - will sync when streaming starts');
                    updateAudioIndicator();
                }
            };
            reader.readAsDataURL(file);
        }

        function updateAudioIndicator() {
            // Show audio indicator in the placeholder if audio is pending
            const placeholder = document.getElementById('placeholder');
            let indicator = document.getElementById('audioIndicator');
            if (pendingAudio && !isStreaming) {
                if (!indicator) {
                    indicator = document.createElement('div');
                    indicator.id = 'audioIndicator';
                    indicator.style.cssText = 'margin-top: 10px; padding: 8px 16px; background: var(--bg-tertiary); border-radius: 8px; font-size: 12px; display: flex; align-items: center; gap: 8px;';
                    indicator.innerHTML = '<span style="font-size: 16px;">🎵</span><span id="audioIndicatorText"></span><button onclick="clearPendingAudio()" style="background: none; border: none; color: var(--text-secondary); cursor: pointer; margin-left: 8px;">✕</button>';
                    placeholder.appendChild(indicator);
                }
                document.getElementById('audioIndicatorText').textContent = pendingAudioName || 'Audio ready';
            } else if (indicator) {
                indicator.remove();
            }
        }

        function clearPendingAudio() {
            pendingAudio = null;
            pendingAudioName = null;
            updateAudioIndicator();
            showToast('Audio cleared');
        }

        function sendAudio(audioData) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;

            const numFrames = parseInt(document.getElementById('numFramesInput').value);
            const frameRate = 24;
            const audioStrength = parseFloat(audioNoiseScaleSlider.value);

            ws.send(JSON.stringify({
                action: 'set_audio',
                audio: audioData,
                strength: audioStrength,
                num_frames: numFrames,
                frame_rate: frameRate
            }));
        }

        // Audio playback functions
        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                nextAudioTime = audioContext.currentTime;
            }
            // Resume if suspended (browsers require user interaction)
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }

        function decodeBase64Audio(base64Audio) {
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return bytes.buffer.slice(0);
        }

        function playAudioChunk(base64Audio) {
            initAudioContext();

            // Store audio in history with current frame position
            const startFrame = frameHistory.length;
            const audioData = decodeBase64Audio(base64Audio);

            // Decode and store for history playback
            audioContext.decodeAudioData(audioData.slice(0), (buffer) => {
                // Store in history
                audioHistory.push({
                    startFrame: startFrame,
                    buffer: buffer,
                    base64: base64Audio // Keep for re-decoding if needed
                });

                // Update end frame of previous segment
                if (audioHistory.length > 1) {
                    audioHistory[audioHistory.length - 2].endFrame = startFrame - 1;
                }

                // Play immediately if live and not muted
                if (isLive && !isAudioMuted && !isPaused) {
                    playBuffer(buffer);
                }
            }, (err) => {
                console.error('Audio decode error:', err);
            });
        }

        function playBuffer(buffer) {
            if (!audioContext || isAudioMuted) return;

            // Calculate playback rate adjustment for FPS difference
            const audioRate = playbackFps / generationFps;

            // Time-stretch the buffer to match playback FPS while preserving pitch
            const stretchedBuffer = timeStretchBuffer(buffer, audioRate);

            const source = audioContext.createBufferSource();
            source.buffer = stretchedBuffer;
            source.connect(audioContext.destination);
            source.playbackRate.value = 1.0;

            // Track this source for pause/stop
            audioSources.push(source);
            source.onended = () => {
                const idx = audioSources.indexOf(source);
                if (idx > -1) audioSources.splice(idx, 1);
            };

            // Schedule audio to play at the right time
            const startTime = Math.max(audioContext.currentTime, nextAudioTime);
            source.start(startTime);
            nextAudioTime = startTime + stretchedBuffer.duration;
        }

        function playAudioForFrame(frameIndex) {
            if (!audioContext || isAudioMuted || audioHistory.length === 0) return;

            // Find which audio segment this frame belongs to
            for (let i = 0; i < audioHistory.length; i++) {
                const segment = audioHistory[i];
                const endFrame = segment.endFrame !== undefined ? segment.endFrame :
                    (i < audioHistory.length - 1 ? audioHistory[i + 1].startFrame - 1 : frameHistory.length);

                if (frameIndex >= segment.startFrame && frameIndex <= endFrame) {
                    // Only play if this is a different segment than currently playing
                    if (currentAudioSegment !== i) {
                        currentAudioSegment = i;
                        stopAllAudio();
                        nextAudioTime = audioContext.currentTime;
                        playBuffer(segment.buffer);
                    }
                    return;
                }
            }
        }

        // Time-stretch audio buffer using granular synthesis (preserves pitch)
        function timeStretchBuffer(buffer, rate) {
            if (Math.abs(rate - 1.0) < 0.01) {
                return buffer; // No stretching needed
            }

            const grainSize = 0.03; // 30ms grains
            const overlap = 0.6; // 60% overlap for smoother output

            const inputLength = buffer.length;
            const outputLength = Math.floor(inputLength / rate);
            const numChannels = buffer.numberOfChannels;
            const sampleRate = buffer.sampleRate;

            const outputBuffer = audioContext.createBuffer(numChannels, outputLength, sampleRate);

            const grainSamples = Math.floor(grainSize * sampleRate);
            const hopIn = Math.floor(grainSamples * (1 - overlap));
            const hopOut = Math.floor(hopIn / rate);

            for (let ch = 0; ch < numChannels; ch++) {
                const input = buffer.getChannelData(ch);
                const output = outputBuffer.getChannelData(ch);

                // Fill with zeros first
                output.fill(0);

                let inPos = 0;
                let outPos = 0;

                while (inPos < inputLength - grainSamples && outPos < outputLength - grainSamples) {
                    // Copy grain with Hann window for smooth crossfade
                    for (let i = 0; i < grainSamples && outPos + i < outputLength; i++) {
                        const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / grainSamples));
                        output[outPos + i] += input[inPos + i] * window;
                    }

                    inPos += hopIn;
                    outPos += hopOut;
                }
            }

            // Normalize to prevent clipping
            for (let ch = 0; ch < numChannels; ch++) {
                const output = outputBuffer.getChannelData(ch);
                let maxVal = 0;
                for (let i = 0; i < output.length; i++) {
                    maxVal = Math.max(maxVal, Math.abs(output[i]));
                }
                if (maxVal > 1.0) {
                    const scale = 0.95 / maxVal;
                    for (let i = 0; i < output.length; i++) {
                        output[i] *= scale;
                    }
                }
            }

            return outputBuffer;
        }

        function stopAllAudio() {
            audioSources.forEach(source => {
                try { source.stop(); } catch(e) {}
            });
            audioSources = [];
            nextAudioTime = 0;
            if (audioContext) {
                nextAudioTime = audioContext.currentTime;
            }
        }

        function pauseAudio() {
            if (audioContext && audioContext.state === 'running') {
                audioContext.suspend();
            }
        }

        function resumeAudio() {
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }

        function toggleMute() {
            isAudioMuted = !isAudioMuted;
            muteBtn.textContent = isAudioMuted ? '🔇' : '🔊';
            if (isAudioMuted) {
                stopAllAudio();
            }
        }

        // Send handling
        function handleSend() {
            const prompt = promptInput.value.trim();
            if (!prompt && !pendingImage) return;

            if (!isStreaming) {
                // Start new stream
                startStream(prompt, pendingImage);
            } else {
                // Update existing stream
                if (prompt && prompt !== currentPrompt) {
                    updatePrompt(prompt);
                }
                if (pendingImage) {
                    sendTargetImage(pendingImage);
                    showToast('Target image set for next segment');
                }
            }

            currentPrompt = prompt || currentPrompt;
            promptInput.value = '';
            removeAttachment();
        }

        // WebSocket connection
        function connect() {
            return new Promise((resolve, reject) => {
                setStatus('connecting', 'Connecting...');

                ws = new WebSocket(WS_ENDPOINT);

                ws.onopen = () => {
                    setStatus('connected', 'Connected');
                    resolve();
                };

                ws.onmessage = handleMessage;

                ws.onerror = (err) => {
                    console.error('WebSocket error:', err);
                    reject(err);
                };

                ws.onclose = () => {
                    setStatus('disconnected', 'Disconnected');
                    isStreaming = false;
                    updateControlsState();
                };
            });
        }

        function handleMessage(event) {
            const msg = JSON.parse(event.data);

            switch (msg.type) {
                case 'frame_chunk':
                    for (const frame of msg.frames) {
                        handleFrame(frame);
                    }
                    break;
                case 'audio':
                    // Play audio chunk from segment
                    playAudioChunk(msg.data);
                    break;
                case 'started':
                    isStreaming = true;
                    isPaused = false;
                    playPauseBtn.textContent = '⏸'; // Show pause icon since we're playing
                    placeholder.classList.add('hidden');
                    videoControls.classList.add('visible');
                    updateControlsState();
                    // Show audio status if included in start
                    if (msg.start_audio_set) {
                        showToast(`Audio ready (${msg.audio_chunks} chunk${msg.audio_chunks > 1 ? 's' : ''})`);
                    } else if (msg.start_audio_error) {
                        showToast('Audio encoding failed');
                    }
                    break;
                case 'stopped':
                case 'complete':
                    // Keep frames but stop receiving new ones
                    break;
                case 'prompt_updated':
                    showToast('Prompt updated');
                    break;
                case 'target_image_set':
                    if (msg.success) {
                        showToast('Target image ready');
                    }
                    break;
                case 'history_reset':
                    showToast('History reset - next segment starts fresh');
                    setStatus('success', 'History reset');
                    break;
                case 'audio_set':
                    if (msg.success) {
                        const chunks = msg.chunks_added || 1;
                        showToast(`Audio queued (${chunks} chunk${chunks > 1 ? 's' : ''})`);
                    }
                    break;
                case 'audio_used':
                    // Audio was consumed for a segment - could show indicator if desired
                    break;
            }
        }

        function handleFrame(msg) {
            const img = new Image();
            img.onload = () => {
                // Store in history for seeking
                frameHistory.push(img);

                // Add to playback buffer if we're live
                if (isLive) {
                    frameBuffer.push(img);
                }

                // Update seek bar max
                seekBar.max = frameHistory.length - 1;
                updateTimeDisplay();
                updateBufferDisplay();

                // Enable controls once we have frames
                if (frameHistory.length === 1) {
                    updateControlsState();
                }
            };
            img.src = 'data:image/jpeg;base64,' + msg.data;
        }

        function displayFrame(index) {
            if (index >= 0 && index < frameHistory.length) {
                const img = frameHistory[index];

                // Resize canvas if needed
                if (canvas.width !== img.width || canvas.height !== img.height) {
                    canvas.width = img.width;
                    canvas.height = img.height;
                }

                ctx.drawImage(img, 0, 0);
            }
        }

        function renderStartImagePreview(img) {
            // Render the start image to canvas immediately (before stream starts)
            const height = parseInt(document.getElementById('heightInput').value);
            const width = parseInt(document.getElementById('widthInput').value);

            canvas.width = width;
            canvas.height = height;

            // Draw image to fit canvas while maintaining aspect ratio
            const scale = Math.min(width / img.width, height / img.height);
            const scaledW = img.width * scale;
            const scaledH = img.height * scale;
            const offsetX = (width - scaledW) / 2;
            const offsetY = (height - scaledH) / 2;

            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, width, height);
            ctx.drawImage(img, offsetX, offsetY, scaledW, scaledH);

            // Hide placeholder since we have an image
            placeholder.classList.add('hidden');
        }

        function updateBufferDisplay() {
            updateTimeDisplay();
        }

        // Playback controls - FPS-controlled frame consumption
        function startPlayback() {
            if (playbackInterval) clearInterval(playbackInterval);

            const frameInterval = 1000 / playbackFps;

            playbackInterval = setInterval(() => {
                if (isPaused) return;

                if (isLive) {
                    // Live mode: consume from buffer at FPS rate
                    if (frameBuffer.length > 0) {
                        const img = frameBuffer.shift();
                        currentFrameIndex = frameHistory.indexOf(img);
                        if (currentFrameIndex === -1) currentFrameIndex = frameHistory.length - 1;

                        // Resize canvas if needed
                        if (canvas.width !== img.width || canvas.height !== img.height) {
                            canvas.width = img.width;
                            canvas.height = img.height;
                        }
                        ctx.drawImage(img, 0, 0);
                        seekBar.value = currentFrameIndex;
                        updateBufferDisplay();
                    }
                } else {
                    // Seeking mode: play through history at FPS rate
                    if (currentFrameIndex < frameHistory.length - 1) {
                        currentFrameIndex++;
                        displayFrame(currentFrameIndex);
                        seekBar.value = currentFrameIndex;
                        updateTimeDisplay();

                        // Play audio for this frame from history
                        playAudioForFrame(currentFrameIndex);

                        // Auto-switch to live when caught up
                        if (currentFrameIndex >= frameHistory.length - 1) {
                            isLive = true;
                            currentAudioSegment = -1; // Reset so live audio plays fresh
                            // Refill buffer with any frames we missed
                            frameBuffer = [];
                            updateLiveIndicator();
                        }
                    }
                }
            }, frameInterval);
        }

        function togglePlayPause() {
            isPaused = !isPaused;
            playPauseBtn.textContent = isPaused ? '▶' : '⏸';
            liveIndicator.classList.toggle('paused', isPaused);

            if (isPaused) {
                // When pausing, exit live mode so user can seek
                isLive = false;
                frameBuffer = []; // Clear buffer
                updateLiveIndicator();
                pauseAudio();
            } else {
                resumeAudio();
            }
        }

        function handleSeek() {
            isLive = false;
            frameBuffer = []; // Clear buffer when seeking
            currentFrameIndex = parseInt(seekBar.value);
            displayFrame(currentFrameIndex);
            updateTimeDisplay();
            updateLiveIndicator();
            // Stop current audio and reset segment tracker
            stopAllAudio();
            currentAudioSegment = -1;
            // If not paused, start playing audio for this position
            if (!isPaused) {
                playAudioForFrame(currentFrameIndex);
            }
        }

        function jumpToLive() {
            isLive = true;
            frameBuffer = []; // Clear old buffer, will refill with new frames
            currentFrameIndex = frameHistory.length - 1;
            seekBar.value = currentFrameIndex;
            displayFrame(currentFrameIndex);
            updateBufferDisplay();
            updateLiveIndicator();
            // Reset audio segment so live audio continues from current position
            currentAudioSegment = -1;
            stopAllAudio();
        }

        function resetHistory() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            ws.send(JSON.stringify({ action: 'reset_history' }));
            setStatus('Resetting history...', 'warning');
        }

        function updateTimeDisplay() {
            const pct = frameHistory.length > 0 ? Math.round((currentFrameIndex + 1) / frameHistory.length * 100) : 0;
            timeDisplay.textContent = `${pct}%`;
        }

        function updateLiveIndicator() {
            liveIndicator.style.opacity = isLive ? '1' : '0.5';
            liveBtn.classList.toggle('active', !isLive);
        }

        function updateControlsState() {
            const hasFrames = frameHistory.length > 0;
            playPauseBtn.disabled = !hasFrames;
            stopBtn.disabled = !isStreaming;
            seekBar.disabled = !hasFrames;
            liveBtn.disabled = !hasFrames;
            resetBtn.disabled = !isStreaming;
        }

        // Stream control
        async function startStream(prompt, imageData) {
            try {
                await connect();

                const height = parseInt(document.getElementById('heightInput').value);
                const width = parseInt(document.getElementById('widthInput').value);
                const seed = parseInt(document.getElementById('seedInput').value);
                const numFrames = parseInt(document.getElementById('numFramesInput').value);

                const message = {
                    action: 'start',
                    prompt: prompt || 'A beautiful cinematic scene',
                    height: height,
                    width: width,
                    seed: seed,
                    num_frames: numFrames,
                    frame_rate: 24,
                    max_segments: 100,
                    use_second_stage: false
                };

                // Add start image if provided (first frame conditioning)
                if (imageData) {
                    message.start_image = imageData;
                }

                // Add start audio if provided (will be chunked and queued for first segment(s))
                if (pendingAudio) {
                    message.start_audio = pendingAudio;
                    message.audio_strength = parseFloat(audioNoiseScaleSlider.value);
                    pendingAudio = null;
                    pendingAudioName = null;
                    updateAudioIndicator();
                }

                ws.send(JSON.stringify(message));

                // Clear old frames and buffers
                frameHistory = [];
                frameBuffer = [];
                currentFrameIndex = 0;
                seekBar.value = 0;
                seekBar.max = 0;
                isLive = true;
                isPaused = false;
                playPauseBtn.textContent = '⏸';

                // Clear audio history and initialize context
                audioHistory = [];
                currentAudioSegment = -1;
                initAudioContext();
                stopAllAudio();

                startPlayback();

            } catch (err) {
                showToast('Failed to connect');
                console.error(err);
            }
        }

        function stopStream() {
            // Send stop command
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'stop' }));
            }

            // Disconnect WebSocket
            disconnect();

            // Stop audio playback
            stopAllAudio();

            // Reset state for fresh start
            isStreaming = false;
            currentPrompt = "";

            updateControlsState();
            showToast('Stream stopped - enter a new prompt to restart');
        }

        function disconnect() {
            if (ws) {
                ws.onclose = null; // Prevent onclose handler
                ws.close();
                ws = null;
            }
            setStatus('disconnected', 'Ready');
        }

        function updatePrompt(prompt) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    action: 'update_prompt',
                    prompt: prompt
                }));
            }
        }

        function sendTargetImage(imageData) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const height = parseInt(document.getElementById('heightInput').value);
                const width = parseInt(document.getElementById('widthInput').value);
                const position = parseFloat(targetPosSlider.value);

                ws.send(JSON.stringify({
                    action: 'set_target_image',
                    image: imageData,
                    height: height,
                    width: width,
                    position: position
                }));
            }
        }

        // UI helpers
        function setStatus(state, text) {
            statusDot.className = 'status-dot ' + state;
            statusText.textContent = text;
        }

        function toggleSettings() {
            settingsPanel.classList.toggle('open');
        }

        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('visible');
            setTimeout(() => toast.classList.remove('visible'), 2000);
        }

        // Start
        aspectRatioSelect.addEventListener('change', updateAspectRatio);
        init();
    </script>
</body>
</html>
"""


# ============================================================================
# Image Director UI - Text-to-image controlled video generation
# ============================================================================

IMAGE_DIRECTOR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Director × LTX-2</title>
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: rgba(20, 20, 30, 0.95);
            --bg-tertiary: rgba(255, 255, 255, 0.1);
            --text-primary: #fff;
            --text-secondary: #888;
            --accent: #8b5cf6;
            --accent-glow: rgba(139, 92, 246, 0.5);
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
        }

        /* Light mode - optimized for e-ink and readability */
        body.light-mode {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --bg-tertiary: rgba(0, 0, 0, 0.08);
            --text-primary: #000000;
            --text-secondary: #444444;
            --accent: #6d28d9;
            --accent-glow: rgba(109, 40, 217, 0.3);
            --success: #16a34a;
            --warning: #d97706;
            --error: #dc2626;
        }

        body.light-mode .video-area,
        body.light-mode .video-wrapper {
            background: #f0f0f0;
        }

        body.light-mode .video-section {
            background: #f5f5f5;
        }
        
        /* Don't change canvas backgrounds - video frames draw over them anyway */
        /* Only the wrapper/section backgrounds matter for the overall look */

        body.light-mode .btn {
            border: 1px solid rgba(0, 0, 0, 0.2);
        }

        body.light-mode .input-group input,
        body.light-mode .input-group textarea,
        body.light-mode .input-group select {
            background: #ffffff;
            border: 1px solid #cccccc;
            color: #000000;
        }

        body.light-mode .activity-log {
            background: #f0f0f0;
        }

        body.light-mode .activity-item {
            border-bottom: 1px solid #dddddd;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            transition: background-color 0.3s, color 0.3s;
        }

        /* Main layout */
        .main-container {
            display: flex;
            flex: 1;
            gap: 20px;
            padding: 20px;
            max-height: 100vh;
            overflow: hidden;
        }

        /* Video section */
        .video-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .video-wrapper {
            flex: 1;
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #videoCanvas {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        #drawingCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: crosshair;
            touch-action: none;
            background: transparent !important;
        }

        .drawing-controls {
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 8px;
            align-items: center;
            background: var(--bg-secondary);
            padding: 8px 12px;
            border-radius: 20px;
            z-index: 10;
        }

        .drawing-controls input[type="color"] {
            width: 28px;
            height: 28px;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            padding: 0;
        }

        .drawing-controls input[type="range"] {
            width: 50px;
        }

        .drawing-controls button {
            padding: 5px 10px;
            border: none;
            border-radius: 8px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.75rem;
        }

        .drawing-controls button:hover {
            background: var(--accent);
        }

        .drawing-status {
            font-size: 0.7rem;
            color: var(--warning);
        }

        /* Generated image preview */
        .generated-preview {
            position: absolute;
            top: 10px;
            right: 10px;
            background: var(--bg-secondary);
            padding: 8px;
            border-radius: 8px;
            display: none;
            z-index: 10;
        }

        .generated-preview.show { display: block; }

        .generated-preview img {
            width: 120px;
            height: auto;
            border-radius: 4px;
        }

        .generated-preview .label {
            font-size: 0.7rem;
            color: var(--accent);
            margin-top: 4px;
            text-align: center;
        }

        /* Reference image preview (outgoing to Gemini) */
        .reference-preview {
            position: absolute;
            top: 10px;
            left: 10px;
            background: var(--bg-secondary);
            padding: 8px;
            border-radius: 8px;
            display: none;
            z-index: 10;
            border: 2px solid var(--warning);
        }

        .reference-preview.show { display: block; }

        .reference-preview img {
            width: 120px;
            height: auto;
            border-radius: 4px;
        }

        .reference-preview .label {
            font-size: 0.7rem;
            color: var(--warning);
            margin-top: 4px;
            text-align: center;
        }

        /* Control panel */
        .control-panel {
            width: 380px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
        }

        .panel-section {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 15px;
        }

        .panel-section h3 {
            font-size: 0.9rem;
            color: var(--accent);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Input groups */
        .input-group {
            margin-bottom: 12px;
        }

        .input-group:last-child {
            margin-bottom: 0;
        }

        .input-group label {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .input-group input[type="text"],
        .input-group input[type="password"],
        .input-group input[type="number"],
        .input-group textarea {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--bg-tertiary);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
            color: var(--text-primary);
            font-size: 0.9rem;
            outline: none;
        }

        .input-group textarea {
            min-height: 60px;
            resize: vertical;
        }

        .input-group input:focus,
        .input-group textarea:focus {
            border-color: var(--accent);
        }

        .input-row {
            display: flex;
            gap: 10px;
        }

        .input-row .input-group {
            flex: 1;
        }

        /* Buttons */
        .btn {
            padding: 10px 16px;
            border: none;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent), #6366f1);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .btn-secondary:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.15);
        }

        .btn-success {
            background: var(--success);
            color: white;
        }

        .btn-warning {
            background: var(--warning);
            color: white;
        }

        .btn-danger {
            background: var(--error);
            color: white;
        }

        .btn-row {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }

        .btn-row .btn {
            flex: 1;
        }

        /* Status bar */
        .status-bar {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            font-size: 0.8rem;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #666;
        }

        .status-dot.connected { background: var(--success); }
        .status-dot.generating { background: var(--warning); animation: pulse 1s infinite; }
        .status-dot.error { background: var(--error); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Activity log */
        .activity-log {
            max-height: 200px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .activity-item {
            padding: 6px 10px;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.3);
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .activity-item.status { border-left: 3px solid var(--warning); }
        .activity-item.success { border-left: 3px solid var(--success); }
        .activity-item.error { border-left: 3px solid var(--error); }
        .activity-item.image { border-left: 3px solid var(--accent); }

        .activity-icon { font-size: 0.9rem; }
        .activity-text { flex: 1; }
        .activity-time { color: var(--text-secondary); font-size: 0.65rem; }

        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            background: var(--bg-secondary);
            padding: 12px 24px;
            border-radius: 8px;
            opacity: 0;
            transition: all 0.3s ease;
            z-index: 1000;
        }

        .toast.show {
            transform: translateX(-50%) translateY(0);
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Video Section -->
        <div class="video-section">
            <div class="video-wrapper">
                <canvas id="videoCanvas" width="832" height="480"></canvas>
                <canvas id="drawingCanvas" width="832" height="480"></canvas>
                
                <div class="drawing-controls">
                    <input type="color" id="brushColor" value="#ff0000" title="Brush color">
                    <input type="range" id="brushSize" min="2" max="20" value="5" title="Brush size">
                    <button onclick="clearDrawing()">Clear</button>
                    <button onclick="undoDrawing()">Undo</button>
                    <span class="drawing-status" id="drawingStatus"></span>
                </div>
                
                <div class="reference-preview" id="referencePreview">
                    <img id="referenceImage" src="" alt="Reference">
                    <div class="label" id="referenceLabel">Sending to Gemini...</div>
                </div>
                
                <div class="generated-preview" id="generatedPreview">
                    <img id="previewImage" src="" alt="Generated">
                    <div class="label" id="previewLabel">Start Image</div>
                </div>
            </div>
            
            <div class="status-bar" style="margin-top: 10px;">
                <div class="status-item">
                    <span class="status-dot" id="ltxStatus"></span>
                    <span>LTX-2</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" id="geminiStatus"></span>
                    <span>Gemini</span>
                </div>
                <div class="status-item">
                    <span>Frames:</span>
                    <span id="frameCount">0</span>
                </div>
                <div class="status-item" id="statusMessage" style="flex: 1; text-align: right; color: var(--text-secondary);"></div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="control-panel">
            <!-- Settings -->
            <div class="panel-section">
                <h3>⚙️ Settings</h3>
                <div class="input-group">
                    <label>Gemini API Key</label>
                    <input type="password" id="apiKey" placeholder="Enter your Gemini API key">
                </div>
                <div class="input-row">
                    <div class="input-group">
                        <label>LTX-2 Resolution</label>
                        <input type="text" id="resolutionDisplay" value="1344x768" readonly>
                    </div>
                    <div class="input-group">
                        <label>Seed</label>
                        <input type="number" id="seed" value="42">
                    </div>
                </div>
                <div class="input-row">
                    <div class="input-group">
                        <label>Chunk Frames</label>
                        <input type="number" id="numFrames" value="49" min="8" max="97" step="1">
                    </div>
                    <div class="input-group">
                        <label>Playback FPS</label>
                        <input type="number" id="playbackFps" value="18" min="1" max="60" step="1" oninput="updatePlaybackFps()">
                    </div>
                </div>
                <div class="input-group" style="margin-top: 10px;">
                    <label>Gemini Model</label>
                    <select id="geminiModel">
                        <option value="gemini-2.5-flash-image" selected>Gemini 2.5 Flash Image (fast)</option>
                        <option value="gemini-3-pro-image-preview">Gemini 3 Pro Image (higher quality)</option>
                    </select>
                </div>
                <div class="input-group" style="margin-top: 10px;">
                    <label>Gemini Aspect Ratio</label>
                    <select id="geminiAspectRatio">
                        <option value="1:1">1:1 (1024x1024)</option>
                        <option value="2:3">2:3 (832x1216)</option>
                        <option value="3:2">3:2 (1216x832)</option>
                        <option value="3:4">3:4 (832x1152)</option>
                        <option value="4:3">4:3 (1152x832)</option>
                        <option value="4:5">4:5 (896x1152)</option>
                        <option value="5:4">5:4 (1152x896)</option>
                        <option value="9:16">9:16 (768x1344)</option>
                        <option value="16:9" selected>16:9 (1344x768)</option>
                        <option value="21:9">21:9 (1536x640)</option>
                    </select>
                </div>
                <button class="btn btn-secondary" style="width: 100%; margin-top: 8px;" onclick="initializeGemini()">Initialize Gemini</button>
            </div>

            <!-- Image Generation -->
            <div class="panel-section">
                <h3>🎨 Image Generation (Gemini)</h3>
                <div class="input-group">
                    <label>Image Description</label>
                    <textarea id="imageDescription" placeholder="Describe the image you want to generate...">A majestic mountain landscape at sunset with dramatic clouds and a serene lake reflection</textarea>
                </div>
                <div class="btn-row">
                    <button class="btn btn-success" id="btnGenerateStart" onclick="generateStartImage()" disabled>Generate Start</button>
                    <button class="btn btn-primary" id="btnGenerateTarget" onclick="generateTargetImage()" disabled>Generate Target</button>
                </div>
                <div class="input-group" style="margin-top: 10px;">
                    <label>Target Position: <span id="positionLabel">End (1.0)</span></label>
                    <input type="range" id="targetPosition" min="0" max="1" step="0.1" value="1" oninput="updatePositionLabel()">
                </div>
            </div>

            <!-- LTX-2 Prompt -->
            <div class="panel-section">
                <h3>🎬 Video Prompt (LTX-2)</h3>
                <div class="input-group">
                    <label>Video Generation Prompt</label>
                    <textarea id="ltxPrompt" placeholder="Describe the video motion and style...">Cinematic slow motion, dramatic lighting, smooth camera movement, high quality</textarea>
                </div>
                <button class="btn btn-secondary" style="width: 100%;" onclick="updatePrompt()">Update Prompt</button>
            </div>

            <!-- Controls -->
            <div class="panel-section">
                <h3>🎮 Controls</h3>
                <div class="btn-row">
                    <button class="btn btn-warning" onclick="resetHistory()">Reset History</button>
                    <button class="btn btn-danger" onclick="stopStream()">Stop</button>
                </div>
                <div class="btn-row" style="margin-top: 10px;">
                    <button class="btn btn-secondary" id="themeToggle" onclick="toggleTheme()" style="width: 50%;">
                        <span id="themeIcon">☀️</span> Light
                    </button>
                    <button class="btn btn-secondary" id="fullscreenToggle" onclick="toggleFullscreen()" style="width: 50%;">
                        <span id="fullscreenIcon">⛶</span> Fullscreen
                    </button>
                </div>
            </div>

            <!-- Activity Log -->
            <div class="panel-section">
                <h3>📋 Activity</h3>
                <div class="activity-log" id="activityLog">
                    <div class="activity-item status">
                        <span class="activity-icon">⏳</span>
                        <span class="activity-text">Enter API key and click Initialize</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        const WS_ENDPOINT = "__WS_ENDPOINT__";

        // State
        let ws = null;
        let isConnected = false;
        let isGeminiReady = false;
        let isStreaming = false;
        let frameBuffer = [];
        let frameCount = 0;
        let playbackInterval = null;
        let playbackFps = 18;

        // Drawing state
        let isDrawing = false;
        let isPaused = false;
        let pausedFrame = null;
        let drawingHistory = [];
        let currentPath = [];

        // DOM Elements
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        const drawingCanvas = document.getElementById('drawingCanvas');
        const drawCtx = drawingCanvas.getContext('2d');
        const ltxStatus = document.getElementById('ltxStatus');
        const geminiStatus = document.getElementById('geminiStatus');
        const activityLog = document.getElementById('activityLog');
        const toast = document.getElementById('toast');
        const playbackFpsInput = document.getElementById('playbackFps');
        const aspectRatioSelect = document.getElementById('geminiAspectRatio');
        const resolutionDisplay = document.getElementById('resolutionDisplay');

        const ASPECT_RATIO_RESOLUTIONS = {
            "1:1": { gemini: { width: 1024, height: 1024 }, ltx: { width: 1024, height: 1024 } },
            "2:3": { gemini: { width: 832, height: 1248 }, ltx: { width: 832, height: 1216 } },
            "3:2": { gemini: { width: 1248, height: 832 }, ltx: { width: 1216, height: 832 } },
            "3:4": { gemini: { width: 864, height: 1184 }, ltx: { width: 832, height: 1152 } },
            "4:3": { gemini: { width: 1184, height: 864 }, ltx: { width: 1152, height: 832 } },
            "4:5": { gemini: { width: 896, height: 1152 }, ltx: { width: 896, height: 1152 } },
            "5:4": { gemini: { width: 1152, height: 896 }, ltx: { width: 1152, height: 896 } },
            "9:16": { gemini: { width: 768, height: 1344 }, ltx: { width: 768, height: 1344 } },
            "16:9": { gemini: { width: 1344, height: 768 }, ltx: { width: 1344, height: 768 } },
            "21:9": { gemini: { width: 1536, height: 672 }, ltx: { width: 1536, height: 640 } }
        };

        function getSelectedResolution() {
            const ratio = aspectRatioSelect.value;
            const mapping = ASPECT_RATIO_RESOLUTIONS[ratio] || ASPECT_RATIO_RESOLUTIONS["16:9"];
            return mapping.ltx;
        }

        // ===== WebSocket Connection =====
        async function connectServer() {
            if (ws && ws.readyState === WebSocket.OPEN) return true;

            return new Promise((resolve, reject) => {
                ws = new WebSocket(WS_ENDPOINT);

                ws.onopen = () => {
                    isConnected = true;
                    ltxStatus.classList.add('connected');
                    logActivity('success', 'Connected to server');
                    resolve(true);
                };

                ws.onmessage = (event) => {
                    const msg = JSON.parse(event.data);
                    handleMessage(msg);
                };

                ws.onclose = () => {
                    isConnected = false;
                    isGeminiReady = false;
                    isStreaming = false;
                    ltxStatus.classList.remove('connected');
                    geminiStatus.classList.remove('connected');
                    logActivity('error', 'Disconnected from server');
                    updateButtons();
                };

                ws.onerror = (err) => {
                    logActivity('error', 'Connection error');
                    reject(err);
                };

                setTimeout(() => {
                    if (!isConnected) reject(new Error('Connection timeout'));
                }, 15000);
            });
        }

        function handleMessage(msg) {
            console.log('Message:', msg.type, msg);
            
            switch (msg.type) {
                case 'initialized':
                    isGeminiReady = true;
                    geminiStatus.classList.add('connected');
                    logActivity('success', 'Gemini client ready');
                    updateButtons();
                    break;

                case 'status':
                    setStatus(msg.message);
                    logActivity('status', msg.message);
                    break;

                case 'generated_image':
                    showGeneratedPreview(msg.data, msg.role, msg.position);
                    logActivity('image', `Generated ${msg.role} image`);
                    break;

                case 'streaming_started':
                    isStreaming = true;
                    startPlayback();
                    logActivity('success', 'Video streaming started');
                    updateButtons();
                    break;

                case 'frame_chunk':
                    for (const frame of msg.frames) {
                        const img = new Image();
                        img.onload = () => {
                            frameBuffer.push(img);
                            if (!isPaused) pausedFrame = img;
                        };
                        img.src = 'data:image/jpeg;base64,' + frame.data;
                        frameCount++;
                    }
                    document.getElementById('frameCount').textContent = frameCount;
                    break;

                case 'target_image_set':
                    logActivity('success', `Target image set at position ${msg.position}`);
                    showToast('Target image set');
                    break;

                case 'prompt_updated':
                    logActivity('success', `Prompt updated`);
                    showToast('Prompt updated');
                    break;

                case 'history_reset':
                    logActivity('success', 'History reset');
                    showToast('History reset');
                    break;

                case 'stopped':
                    isStreaming = false;
                    stopPlayback();
                    logActivity('status', 'Streaming stopped');
                    updateButtons();
                    break;

                case 'error':
                    logActivity('error', msg.message);
                    showToast('Error: ' + msg.message);
                    setStatus(msg.message, true);
                    break;
            }
        }

        // ===== Gemini Initialization =====
        async function initializeGemini() {
            const apiKey = document.getElementById('apiKey').value.trim();
            if (!apiKey) {
                showToast('Please enter your Gemini API key');
                return;
            }

            try {
                if (!isConnected) {
                    await connectServer();
                }

                ws.send(JSON.stringify({
                    action: 'init',
                    api_key: apiKey,
                    model: document.getElementById('geminiModel').value
                }));
                
                // Save API key
                localStorage.setItem('gemini_api_key', apiKey);
                logActivity('status', 'Initializing Gemini...');
            } catch (err) {
                logActivity('error', 'Failed to connect: ' + err.message);
            }
        }

        // ===== Image Generation =====
        async function generateStartImage() {
            const description = document.getElementById('imageDescription').value.trim();
            const ltxPrompt = document.getElementById('ltxPrompt').value.trim();
            const numFrames = parseInt(document.getElementById('numFrames').value, 10) || 49;
            const aspectRatio = document.getElementById('geminiAspectRatio').value;
            const { width, height } = getSelectedResolution();
            
            if (!description) {
                showToast('Please enter an image description');
                return;
            }

            const hasDrawings = hasDrawing();
            const drawnImage = hasDrawings ? getCombinedImage() : null;
            
            // Show reference preview if we're sending an image to Gemini
            if (drawnImage) {
                showReferencePreview(drawnImage, true);
            }

            ws.send(JSON.stringify({
                action: 'generate_start_image',
                description: description,
                ltx_prompt: ltxPrompt || description,
                height: height,
                width: width,
                seed: parseInt(document.getElementById('seed').value),
                num_frames: numFrames,
                aspect_ratio: aspectRatio,
                drawn_image: drawnImage,
                model: document.getElementById('geminiModel').value
            }));

            logActivity('status', hasDrawings ? 'Generating start image (with drawing)...' : 'Generating start image...');
            clearDrawing();
        }

        async function generateTargetImage() {
            const description = document.getElementById('imageDescription').value.trim();
            const position = parseFloat(document.getElementById('targetPosition').value);
            const aspectRatio = document.getElementById('geminiAspectRatio').value;
            const { width, height } = getSelectedResolution();
            
            if (!description) {
                showToast('Please enter an image description');
                return;
            }

            const hasDrawings = hasDrawing();
            const drawnImage = hasDrawings ? getCombinedImage() : null;
            
            // Show reference preview - either the drawn image or the last frame
            if (drawnImage) {
                showReferencePreview(drawnImage, true);
            } else if (frameBuffer.length > 0) {
                // Show the last frame as reference (even without drawings)
                const lastFrame = frameBuffer[frameBuffer.length - 1];
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = lastFrame.width;
                tempCanvas.height = lastFrame.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(lastFrame, 0, 0);
                const frameB64 = tempCanvas.toDataURL('image/jpeg', 0.9).split(',')[1];
                showReferencePreview(frameB64, false);
            }

            ws.send(JSON.stringify({
                action: 'generate_target_image',
                description: description,
                position: position,
                height: height,
                width: width,
                aspect_ratio: aspectRatio,
                drawn_image: drawnImage,
                model: document.getElementById('geminiModel').value
            }));

            logActivity('status', hasDrawings ? 'Generating target image (with drawing)...' : 'Generating target image...');
            clearDrawing();
        }

        function updatePrompt() {
            const prompt = document.getElementById('ltxPrompt').value.trim();
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    action: 'update_prompt',
                    prompt: prompt
                }));
            }
        }

        function resetHistory() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'reset_history' }));
            }
        }

        function stopStream() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'stop' }));
            }
        }

        // ===== Video Playback =====
        function startPlayback() {
            if (playbackInterval) return;
            playbackInterval = setInterval(() => {
                if (isPaused) return;
                if (frameBuffer.length > 0) {
                    const frame = frameBuffer.shift();
                    ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
                    pausedFrame = frame;
                }
            }, 1000 / playbackFps);
        }

        function updatePlaybackFps() {
            const nextFps = parseFloat(playbackFpsInput.value);
            if (!Number.isFinite(nextFps) || nextFps <= 0) {
                return;
            }
            playbackFps = nextFps;
            if (playbackInterval) {
                stopPlayback();
                startPlayback();
            }
        }

        function updateAspectRatio() {
            const { width, height } = getSelectedResolution();
            resolutionDisplay.value = `${width}x${height}`;
            canvas.width = width;
            canvas.height = height;
            drawingCanvas.width = width;
            drawingCanvas.height = height;
            drawCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        function stopPlayback() {
            if (playbackInterval) {
                clearInterval(playbackInterval);
                playbackInterval = null;
            }
        }

        // ===== Drawing Functions =====
        function initDrawing() {
            drawingCanvas.addEventListener('mousedown', startDrawing);
            drawingCanvas.addEventListener('mousemove', draw);
            drawingCanvas.addEventListener('mouseup', stopDrawingStroke);
            drawingCanvas.addEventListener('mouseleave', stopDrawingStroke);

            drawingCanvas.addEventListener('touchstart', (e) => {
                e.preventDefault();
                const touch = e.touches[0];
                startDrawing({ clientX: touch.clientX, clientY: touch.clientY });
            });
            drawingCanvas.addEventListener('touchmove', (e) => {
                e.preventDefault();
                const touch = e.touches[0];
                draw({ clientX: touch.clientX, clientY: touch.clientY });
            });
            drawingCanvas.addEventListener('touchend', stopDrawingStroke);
        }

        function getDrawingCoords(e) {
            // Account for object-fit: contain letterboxing
            const rect = drawingCanvas.getBoundingClientRect();
            const canvasAspect = drawingCanvas.width / drawingCanvas.height;
            const rectAspect = rect.width / rect.height;

            let renderW, renderH, offsetX, offsetY;
            if (rectAspect > canvasAspect) {
                // Letterboxed on sides (pillarboxing)
                renderH = rect.height;
                renderW = rect.height * canvasAspect;
                offsetX = (rect.width - renderW) / 2;
                offsetY = 0;
            } else {
                // Letterboxed on top/bottom
                renderW = rect.width;
                renderH = rect.width / canvasAspect;
                offsetX = 0;
                offsetY = (rect.height - renderH) / 2;
            }

            return {
                x: ((e.clientX - rect.left - offsetX) / renderW) * drawingCanvas.width,
                y: ((e.clientY - rect.top - offsetY) / renderH) * drawingCanvas.height
            };
        }

        function startDrawing(e) {
            isDrawing = true;
            if (!isPaused && pausedFrame) {
                isPaused = true;
                document.getElementById('drawingStatus').textContent = 'Paused - Drawing';
            }

            const coords = getDrawingCoords(e);
            currentPath = [coords];

            drawCtx.strokeStyle = document.getElementById('brushColor').value;
            drawCtx.lineWidth = document.getElementById('brushSize').value;
            drawCtx.lineCap = 'round';
            drawCtx.lineJoin = 'round';
            drawCtx.beginPath();
            drawCtx.moveTo(coords.x, coords.y);
        }

        function draw(e) {
            if (!isDrawing) return;
            const coords = getDrawingCoords(e);
            currentPath.push(coords);
            drawCtx.lineTo(coords.x, coords.y);
            drawCtx.stroke();
        }

        function stopDrawingStroke() {
            if (isDrawing && currentPath.length > 0) {
                drawingHistory.push({
                    path: currentPath,
                    color: document.getElementById('brushColor').value,
                    size: document.getElementById('brushSize').value
                });
            }
            isDrawing = false;
            currentPath = [];
        }

        function clearDrawing() {
            drawCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            drawingHistory = [];
            isPaused = false;
            document.getElementById('drawingStatus').textContent = '';
        }

        function undoDrawing() {
            if (drawingHistory.length === 0) return;
            drawingHistory.pop();
            redrawAll();
        }

        function redrawAll() {
            drawCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            for (const stroke of drawingHistory) {
                if (stroke.path.length < 2) continue;
                drawCtx.strokeStyle = stroke.color;
                drawCtx.lineWidth = stroke.size;
                drawCtx.lineCap = 'round';
                drawCtx.lineJoin = 'round';
                drawCtx.beginPath();
                drawCtx.moveTo(stroke.path[0].x, stroke.path[0].y);
                for (let i = 1; i < stroke.path.length; i++) {
                    drawCtx.lineTo(stroke.path[i].x, stroke.path[i].y);
                }
                drawCtx.stroke();
            }
        }

        function getCombinedImage() {
            const combined = document.createElement('canvas');
            combined.width = drawingCanvas.width;
            combined.height = drawingCanvas.height;
            const combCtx = combined.getContext('2d');

            if (pausedFrame) {
                combCtx.drawImage(pausedFrame, 0, 0, combined.width, combined.height);
            }
            combCtx.drawImage(drawingCanvas, 0, 0, combined.width, combined.height);

            return combined.toDataURL('image/jpeg', 0.9).split(',')[1];
        }

        function hasDrawing() {
            return drawingHistory.length > 0;
        }

        // ===== UI Helpers =====
        function showReferencePreview(imageData, hasDrawing) {
            const preview = document.getElementById('referencePreview');
            const img = document.getElementById('referenceImage');
            const label = document.getElementById('referenceLabel');
            
            img.src = 'data:image/jpeg;base64,' + imageData;
            label.textContent = hasDrawing ? '📤 With Drawings → Gemini' : '📤 Frame → Gemini';
            preview.classList.add('show');
            
            // Hide when we get the generated image back (or after 10s max)
            setTimeout(() => preview.classList.remove('show'), 10000);
        }
        
        function hideReferencePreview() {
            document.getElementById('referencePreview').classList.remove('show');
        }
        
        function showGeneratedPreview(imageData, role, position) {
            // Hide reference preview when we get the result
            hideReferencePreview();
            
            const preview = document.getElementById('generatedPreview');
            const img = document.getElementById('previewImage');
            const label = document.getElementById('previewLabel');
            
            img.src = 'data:image/jpeg;base64,' + imageData;
            label.textContent = role === 'start' ? 'Start Image' : `Target @ ${position}`;
            preview.classList.add('show');
            
            setTimeout(() => preview.classList.remove('show'), 5000);
        }

        function updatePositionLabel() {
            const val = parseFloat(document.getElementById('targetPosition').value);
            let label;
            if (val <= 0.2) label = 'Start';
            else if (val <= 0.4) label = 'Early';
            else if (val <= 0.6) label = 'Middle';
            else if (val <= 0.8) label = 'Late';
            else label = 'End';
            document.getElementById('positionLabel').textContent = `${label} (${val})`;
        }

        function updateButtons() {
            document.getElementById('btnGenerateStart').disabled = !isGeminiReady || isStreaming;
            document.getElementById('btnGenerateTarget').disabled = !isGeminiReady || !isStreaming;
        }

        function setStatus(message, isError = false) {
            const el = document.getElementById('statusMessage');
            el.textContent = message;
            el.style.color = isError ? 'var(--error)' : 'var(--text-secondary)';
        }

        function logActivity(type, message) {
            const time = new Date().toLocaleTimeString();
            const icons = { status: '📢', success: '✅', error: '❌', image: '🖼️' };
            const item = document.createElement('div');
            item.className = `activity-item ${type}`;
            item.innerHTML = `
                <span class="activity-icon">${icons[type] || '📝'}</span>
                <span class="activity-text">${message}</span>
                <span class="activity-time">${time}</span>
            `;
            activityLog.insertBefore(item, activityLog.firstChild);
            while (activityLog.children.length > 30) {
                activityLog.removeChild(activityLog.lastChild);
            }
        }

        function showToast(message, duration = 3000) {
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), duration);
        }

        // ===== Theme Toggle =====
        function toggleTheme() {
            document.body.classList.toggle('light-mode');
            const isLight = document.body.classList.contains('light-mode');
            const btn = document.getElementById('themeToggle');
            const icon = document.getElementById('themeIcon');
            
            icon.textContent = isLight ? '🌙' : '☀️';
            btn.innerHTML = icon.outerHTML + (isLight ? ' Dark' : ' Light');
            
            localStorage.setItem('image_director_theme', isLight ? 'light' : 'dark');
        }
        
        function loadSavedTheme() {
            const savedTheme = localStorage.getItem('image_director_theme');
            if (savedTheme === 'light') {
                document.body.classList.add('light-mode');
                const btn = document.getElementById('themeToggle');
                const icon = document.getElementById('themeIcon');
                icon.textContent = '🌙';
                btn.innerHTML = icon.outerHTML + ' Dark';
            }
        }

        // ===== Fullscreen Toggle =====
        function toggleFullscreen() {
            const icon = document.getElementById('fullscreenIcon');
            const btn = document.getElementById('fullscreenToggle');
            
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().then(() => {
                    icon.textContent = '⛶';
                    btn.innerHTML = icon.outerHTML + ' Exit';
                }).catch(err => {
                    showToast('Fullscreen failed: ' + err.message, 3000);
                });
            } else {
                document.exitFullscreen().then(() => {
                    icon.textContent = '⛶';
                    btn.innerHTML = icon.outerHTML + ' Fullscreen';
                }).catch(err => {
                    showToast('Exit fullscreen failed: ' + err.message, 3000);
                });
            }
        }
        
        // Listen for fullscreen changes (e.g., user presses Escape)
        document.addEventListener('fullscreenchange', () => {
            const icon = document.getElementById('fullscreenIcon');
            const btn = document.getElementById('fullscreenToggle');
            if (document.fullscreenElement) {
                icon.textContent = '⛶';
                btn.innerHTML = icon.outerHTML + ' Exit';
            } else {
                icon.textContent = '⛶';
                btn.innerHTML = icon.outerHTML + ' Fullscreen';
            }
        });

        // ===== Initialize =====
        function init() {
            loadSavedTheme();
            initDrawing();
            updateButtons();
            updatePositionLabel();
            updateAspectRatio();

            // Load saved API key
            const savedKey = localStorage.getItem('gemini_api_key');
            if (savedKey) {
                document.getElementById('apiKey').value = savedKey;
            }

            // Connect on load
            setTimeout(connectServer, 500);
        }

        init();
    </script>
</body>
</html>
"""


@app.function(timeout=3600)
@modal.asgi_app()
def image_director_ui():
    """
    Image Director UI - Generate images with Gemini Flash-Image, stream video with LTX-2.
    
    No voice/audio - just text-to-image with optional drawing annotations.
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import asyncio
    import json
    import base64
    from google import genai

    ui_app = FastAPI(title="Image Director × LTX-2")
    ui_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    LTX_WS_URL = WEBSOCKET_ENDPOINT_TURBO

    @ui_app.get("/", response_class=HTMLResponse)
    async def index():
        html = IMAGE_DIRECTOR_HTML.replace(
            "__WS_ENDPOINT__",
            "wss://tmalive--ltx2-official-distilled-loopy-image-director-ui.modal.run/ws/image-director"
        )
        return HTMLResponse(html)

    @ui_app.get("/health")
    async def health():
        return {"status": "healthy", "service": "image-director-ui"}

    @ui_app.websocket("/ws/image-director")
    async def image_director_bridge(websocket: WebSocket):
        """Bridge: Client (Gemini Flash-Image (CPU) (LTX-2 (GPU)"""
        await websocket.accept()
        print("Client connected", flush=True)

        ltx_ws = None
        genai_client = None
        gemini_model = "gemini-2.5-flash-image"
        should_stop = False
        state = {"last_frame_b64": None, "is_streaming": False}
        tasks = []

        def is_ws_closed(ws):
            """Check if websocket is closed (compatible with different websockets versions)."""
            if ws is None:
                return True
            # Try different APIs for compatibility
            try:
                # Newer websockets (>=13.0) uses state
                from websockets.protocol import State
                return ws.state == State.CLOSED or ws.state == State.CLOSING
            except (ImportError, AttributeError):
                pass
            try:
                # Some versions have .closed property
                return ws.closed
            except AttributeError:
                pass
            try:
                # Check close_code (None if open)
                return ws.close_code is not None
            except AttributeError:
                pass
            # Assume open if we can't determine
            return False

        async def connect_to_ltx():
            """Connect to LTX-2 GPU endpoint using websockets library."""
            nonlocal ltx_ws
            import websockets
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"CPU-WS: Connecting to {LTX_WS_URL} (attempt {attempt + 1})...", flush=True)
                    # Use websockets library with LONG open_timeout for cold start
                    # The open_timeout is critical - it's the websockets library's internal timeout
                    # for the opening handshake, NOT covered by asyncio.wait_for
                    ltx_ws = await websockets.connect(
                        LTX_WS_URL,
                        open_timeout=660,      # 11 min for cold start - THIS IS THE KEY!
                        ping_interval=20,      # Send ping every 20s
                        ping_timeout=60,       # Wait 60s for pong
                        close_timeout=10,
                        max_size=50 * 1024 * 1024,  # 50MB max message size for receiving
                    )
                    print(f"CPU-WS: Connected, verifying with ping...", flush=True)
                    
                    # Send a ping to verify bidirectional communication before returning
                    try:
                        await ltx_ws.send(json.dumps({"action": "ping"}))
                        pong = await asyncio.wait_for(ltx_ws.recv(), timeout=5.0)
                        pong_data = json.loads(pong)
                        if pong_data.get("type") != "pong":
                            raise RuntimeError(f"Expected pong, got: {pong_data}")
                        print(f"CPU-WS: Connection verified with ping/pong!", flush=True)
                    except Exception as e:
                        print(f"CPU-WS: Ping/pong failed: {e}", flush=True)
                        await ltx_ws.close()
                        ltx_ws = None
                        raise
                    
                    return True
                except Exception as e:
                    print(f"CPU-WS: Connection failed (attempt {attempt + 1}): {type(e).__name__}: {e}", flush=True)
                    if ltx_ws:
                        try:
                            await ltx_ws.close()
                        except Exception:
                            pass
                        ltx_ws = None
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0)
            
            return False

        async def send_to_ltx(payload, purpose: str):
            """Send a message to LTX-2, reconnecting if needed."""
            nonlocal ltx_ws
            
            max_retries = 5
            for attempt in range(max_retries):
                # Connect if needed
                if ltx_ws is None:
                    print(f"CPU-WS: ltx_ws is None, need to connect", flush=True)
                elif is_ws_closed(ltx_ws):
                    print(f"CPU-WS: Connection closed, need to reconnect", flush=True)
                
                if ltx_ws is None or is_ws_closed(ltx_ws):
                    if ltx_ws:
                        try:
                            await ltx_ws.close()
                        except Exception:
                            pass
                        ltx_ws = None
                    if not await connect_to_ltx():
                        if attempt < max_retries - 1:
                            print(f"CPU-WS: Connection failed, retry {attempt + 2}/{max_retries}...", flush=True)
                            await asyncio.sleep(1.0)
                            continue
                        return False, f"LTX-2 connection failed ({purpose})"
                
                # Try to send
                try:
                    payload_str = json.dumps(payload)
                    payload_size = len(payload_str)
                    print(f"CPU-WS: Sending {purpose} ({payload_size} bytes / {payload_size/1024/1024:.2f} MB)...", flush=True)
                    await ltx_ws.send(payload_str)
                    print(f"CPU-WS: Sent {purpose} successfully!", flush=True)
                    return True, None
                except Exception as e:
                    print(f"CPU-WS: Send {purpose} failed (attempt {attempt + 1}): {type(e).__name__}: {e}", flush=True)
                    try:
                        await ltx_ws.close()
                    except Exception:
                        pass
                    ltx_ws = None
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
            
            return False, f"LTX-2 send failed after {max_retries} attempts ({purpose})"

        async def receive_from_ltx():
            """Receive frames from LTX-2 and forward to client."""
            nonlocal ltx_ws
            import websockets
            
            # Log initial state
            ws_state = f"ltx_ws={ltx_ws is not None}, closed={is_ws_closed(ltx_ws)}"
            print(f"CPU-WS: Starting receive loop ({ws_state})", flush=True)
            
            frame_count = 0
            try:
                loop_iteration = 0
                while not should_stop and ltx_ws and not is_ws_closed(ltx_ws):
                    loop_iteration += 1
                    if loop_iteration == 1:
                        print(f"CPU-WS: Entered recv loop, waiting for first message...", flush=True)
                    try:
                        msg = await ltx_ws.recv()
                        parsed = json.loads(msg)
                        msg_type = parsed.get("type", "unknown")
                        
                        if frame_count == 0:
                            print(f"CPU-WS: First message received: type={msg_type}", flush=True)
                        
                        await websocket.send_json(parsed)
                        # Store latest frame for image generation conditioning
                        if msg_type == "frame_chunk" and parsed.get("frames"):
                            last_frame = parsed["frames"][-1]
                            state["last_frame_b64"] = last_frame["data"]
                            frame_count += len(parsed["frames"])
                            if frame_count % 100 < len(parsed["frames"]):
                                print(f"CPU-WS: Forwarded {frame_count} frames", flush=True)
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"CPU-WS: Connection closed: {e}", flush=True)
                        break
                    except Exception as e:
                        if not should_stop:
                            print(f"CPU-WS: Recv error: {type(e).__name__}: {e}", flush=True)
                        break
                
                # Log why we exited the loop
                exit_reason = []
                if should_stop:
                    exit_reason.append("should_stop=True")
                if not ltx_ws:
                    exit_reason.append("ltx_ws=None")
                elif is_ws_closed(ltx_ws):
                    exit_reason.append("ws_closed=True")
                print(f"CPU-WS: Exited recv loop after {loop_iteration} iterations. Reason: {', '.join(exit_reason) or 'unknown'}", flush=True)
                
            finally:
                print(f"CPU-WS: Receive loop ended, forwarded {frame_count} frames total", flush=True)
                # Clean up the connection so next generate_start_image creates a fresh one
                if ltx_ws:
                    try:
                        await ltx_ws.close()
                    except Exception:
                        pass
                    ltx_ws = None
                state["is_streaming"] = False
                print("CPU-WS: Connection cleaned up, ready for new connection", flush=True)

        async def generate_image(description: str, reference_image_b64: str = None, aspect_ratio: str = "16:9", model_override: str = None):
            """Generate an image using the selected Gemini model."""
            if not genai_client:
                return None, "Gemini client not initialized"

            model = model_override or gemini_model

            try:
                from google.genai import types as genai_types

                # Build contents with optional reference image
                if reference_image_b64:
                    image_bytes = base64.b64decode(reference_image_b64)
                    contents = [
                        genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        f"Based on this image, generate a new image: {description}"
                    ]
                    print(f"[{model}] Generating image with reference: {description[:50]}...", flush=True)
                else:
                    contents = [description]
                    print(f"[{model}] Generating image from text: {description[:50]}...", flush=True)

                # Build config based on model
                if model == "gemini-3-pro-image-preview":
                    config = genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        image_config=genai_types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size="1K",
                        ),
                        system_instruction=[
                            genai_types.Part.from_text(text="do not think just produce the image immediately"),
                        ],
                    )
                else:
                    config = genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        image_config=genai_types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                        ),
                    )

                response = await genai_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                # Extract image data
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        image_b64 = base64.b64encode(part.inline_data.data).decode()
                        print(f"[{model}] Image generated successfully", flush=True)
                        return image_b64, None

                return None, "No image in response"
            except Exception as e:
                print(f"[{model}] Image generation error: {e}", flush=True)
                return None, str(e)

        def encode_image_as_jpeg(image_b64: str, target_width: int, target_height: int) -> str:
            """Resize image to target dimensions and encode as JPEG for efficient transfer."""
            from PIL import Image
            import io

            # Decode base64 to image
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes))

            original_size = len(image_bytes)
            print(f"Resizing image from {img.size} to ({target_width}, {target_height}) and encoding as JPEG", flush=True)

            # Resize to target dimensions (match LTX-2 generation resolution)
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Encode as JPEG - lower quality adds compression artifacts that give
            # the LTX-2 encoder texture/noise to work with for animation
            buffer = io.BytesIO()
            img_resized.save(buffer, format="JPEG", quality=97)
            new_bytes = buffer.getvalue()
            new_size = len(new_bytes)

            print(f"Image size: {original_size/1024:.1f}KB -> {new_size/1024:.1f}KB", flush=True)

            return base64.b64encode(new_bytes).decode()

        # Full-scale LTX-2 resolutions (all divisible by 64)
        ASPECT_RATIO_RESOLUTIONS = {
            "1:1": (1024, 1024),
            "2:3": (832, 1216),
            "3:2": (1216, 832),
            "3:4": (832, 1152),
            "4:3": (1152, 832),
            "4:5": (896, 1152),
            "5:4": (1152, 896),
            "9:16": (768, 1344),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
        }

        def resolve_dimensions(aspect_ratio: str, fallback_w: int, fallback_h: int):
            size = ASPECT_RATIO_RESOLUTIONS.get(aspect_ratio)
            if size:
                return size
            return fallback_w, fallback_h

        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    msg = json.loads(data)
                    action = msg.get("action")

                    if action == "init":
                        # Initialize Gemini client with API key and model selection
                        api_key = msg.get("api_key")
                        if not api_key:
                            await websocket.send_json({"type": "error", "message": "API key required"})
                            continue

                        gemini_model = msg.get("model", "gemini-2.5-flash-image")

                        try:
                            genai_client = genai.Client(api_key=api_key)
                            await websocket.send_json({"type": "initialized", "message": f"Gemini client ready ({gemini_model})"})
                            print(f"Gemini client initialized with model: {gemini_model}", flush=True)
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": f"Failed to initialize Gemini: {e}"})

                    elif action == "generate_start_image":
                        # Generate initial image and start LTX-2 stream
                        
                        # Check if already streaming - if so, user should use generate_target_image instead
                        if state["is_streaming"]:
                            await websocket.send_json({
                                "type": "error", 
                                "message": "Already streaming! Use 'Generate Target' to add new images, or 'Stop' first."
                            })
                            continue
                        
                        description = msg.get("description", "")
                        ltx_prompt = msg.get("ltx_prompt", description)
                        height = msg.get("height", 480)
                        width = msg.get("width", 832)
                        aspect_ratio = msg.get("aspect_ratio", "16:9")
                        seed = msg.get("seed", 42)
                        drawn_image = msg.get("drawn_image")  # Optional drawing
                        width, height = resolve_dimensions(aspect_ratio, width, height)
                        
                        if not description:
                            await websocket.send_json({"type": "error", "message": "Image description required"})
                            continue
                        
                        await websocket.send_json({"type": "status", "message": "Generating start image..."})
                        
                        # Generate image (use drawn image as reference if provided)
                        model = msg.get("model")
                        image_b64, error = await generate_image(description, drawn_image, aspect_ratio, model_override=model)
                        if error:
                            await websocket.send_json({"type": "error", "message": f"Image generation failed: {error}"})
                            continue

                        # Resize to LTX-2 resolution and encode as JPEG
                        image_b64 = encode_image_as_jpeg(image_b64, width, height)

                        # Send generated image preview to client
                        await websocket.send_json({"type": "generated_image", "data": image_b64, "role": "start"})
                        
                        # Send start image BEFORE start so it's ready when generation begins
                        await websocket.send_json({"type": "status", "message": "Connecting to LTX-2 and sending start image..."})

                        ok, err = await send_to_ltx({
                            "action": "set_start_image",
                            "image": image_b64,
                            "height": height,
                            "width": width,
                        }, "set_start_image")
                        if not ok:
                            await websocket.send_json({"type": "error", "message": f"Failed to send start image: {err}"})
                            # Continue anyway, streaming will work without start image

                        # Now start the stream - start_image_latent is already set on GPU
                        await websocket.send_json({"type": "status", "message": "Starting LTX-2 stream..."})
                        ok, err = await send_to_ltx({
                            "action": "start",
                            "prompt": ltx_prompt,
                            "height": height,
                            "width": width,
                            "seed": seed,
                            "num_frames": 49,
                            "frame_rate": 24,
                            "max_segments": 1000,
                            "loopy_mode": True,
                            "skip_audio": True,
                        }, "start")
                        if not ok:
                            await websocket.send_json({"type": "error", "message": err})
                            continue
                        
                        state["is_streaming"] = True
                        tasks.append(asyncio.create_task(receive_from_ltx()))
                        await websocket.send_json({"type": "streaming_started"})

                    elif action == "generate_target_image":
                        # Generate target image during streaming
                        if not state["is_streaming"]:
                            await websocket.send_json({"type": "error", "message": "Not streaming - generate start image first"})
                            continue
                        
                        description = msg.get("description", "")
                        position = float(msg.get("position", 1.0))
                        position = max(0.0, min(1.0, position))
                        drawn_image = msg.get("drawn_image")  # Optional drawing
                        aspect_ratio = msg.get("aspect_ratio", "16:9")
                        height = msg.get("height", 480)
                        width = msg.get("width", 832)
                        width, height = resolve_dimensions(aspect_ratio, width, height)
                        
                        if not description:
                            await websocket.send_json({"type": "error", "message": "Image description required"})
                            continue
                        
                        await websocket.send_json({"type": "status", "message": "Generating target image..."})
                        
                        # Use drawn image or current frame as reference
                        reference = drawn_image or state.get("last_frame_b64")
                        model = msg.get("model")
                        image_b64, error = await generate_image(description, reference, aspect_ratio, model_override=model)
                        if error:
                            await websocket.send_json({"type": "error", "message": f"Image generation failed: {error}"})
                            continue
                        
                        # Resize to LTX-2 resolution and encode as JPEG
                        image_b64 = encode_image_as_jpeg(image_b64, width, height)

                        # Send generated image preview to client
                        await websocket.send_json({"type": "generated_image", "data": image_b64, "role": "target", "position": position})
                        
                        # Send to LTX-2 as target image
                        ok, err = await send_to_ltx({
                            "action": "set_target_image",
                            "image": image_b64,
                            "position": position,
                            "height": height,
                            "width": width
                        }, "set_target_image")
                        if ok:
                            await websocket.send_json({"type": "target_image_set", "position": position})
                        else:
                            await websocket.send_json({"type": "error", "message": err})
                            print("Warning: Failed to set target image", flush=True)

                    elif action == "update_prompt":
                        # Update LTX-2 prompt directly
                        prompt = msg.get("prompt", "")
                        ok, err = await send_to_ltx({"action": "update_prompt", "prompt": prompt}, "update_prompt")
                        if ok:
                            await websocket.send_json({"type": "prompt_updated", "prompt": prompt})
                        else:
                            await websocket.send_json({"type": "error", "message": err})

                    elif action == "reset_history":
                        # Reset LTX-2 generation history
                        ok, err = await send_to_ltx({"action": "reset_history"}, "reset_history")
                        if ok:
                            await websocket.send_json({"type": "history_reset"})
                        else:
                            await websocket.send_json({"type": "error", "message": err})

                    elif action == "stop":
                        ok, err = await send_to_ltx({"action": "stop"}, "stop")
                        if not ok:
                            await websocket.send_json({"type": "error", "message": err})
                        state["is_streaming"] = False
                        await websocket.send_json({"type": "stopped"})
                        break

                except asyncio.TimeoutError:
                    # Check if LTX connection died while we were waiting
                    if state["is_streaming"] and (not ltx_ws or is_ws_closed(ltx_ws)):
                        print("LTX connection lost during streaming, notifying client", flush=True)
                        await websocket.send_json({"type": "error", "message": "LTX-2 connection lost"})
                        await websocket.send_json({"type": "ltx_disconnected"})
                        state["is_streaming"] = False
                    continue

        except WebSocketDisconnect:
            print("CLIENT DISCONNECTED - browser closed or refreshed", flush=True)
        except asyncio.CancelledError:
            print("BRIDGE CANCELLED - task was cancelled", flush=True)
        except Exception as e:
            import traceback
            print(f"BRIDGE ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            print("CLEANUP STARTING...", flush=True)
            should_stop = True
            for t in tasks:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            if ltx_ws:
                try:
                    await ltx_ws.close()
                except Exception:
                    pass
            print("Bridge cleanup complete", flush=True)

    return ui_app


@app.function(timeout=60)
@modal.asgi_app()
def minimal_ui():
    """
    Minimalistic video-focused streaming UI.

    Features:
    - Full-screen video canvas with frame history
    - Chat-style prompt input with image attachment
    - Video controls: play/pause, seek, jump to live
    - Settings panel for resolution, seed, FPS, target position
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    ui_app = FastAPI(title="LTX-2 Minimal UI")
    ui_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def inject_endpoint(html):
        """Inject WebSocket endpoint URL into the HTML."""
        return html.replace("__WS_ENDPOINT_TURBO__", WEBSOCKET_ENDPOINT_TURBO)

    @ui_app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(inject_endpoint(MINIMAL_HTML))

    @ui_app.get("/health")
    async def health():
        return {"status": "healthy", "service": "minimal-ui"}

    return ui_app


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

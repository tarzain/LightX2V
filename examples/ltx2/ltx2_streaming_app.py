#!/usr/bin/env python3
"""
LTX-2 Streaming Video Generation with Rolling Segments.
Generates video continuously using autoregressive segment generation,
streaming frames to the browser in real-time.
"""
import os
import sys
import base64
import queue
import threading
import time
import gc
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO

# Configuration
LTX2_MODELS_DIR = "/workspace/models/LTX-2"
GEMMA_DIR = "/workspace/models/gemma"
USE_FP8 = False  # True = FP8 checkpoint (~27GB), False = BF16 (~43GB)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
engine = None
is_generating = False
current_prompt = "A beautiful landscape with mountains and flowing water, cinematic, high quality"
prompt_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=256)
audio_queue = queue.Queue(maxsize=32)  # Audio chunks queue
hard_reset_requested = False

# Image adjustment parameters
brightness = 1.0
contrast = 1.0
gamma = 1.0

# Audio sample rate - vocoder outputs at 24kHz (not the decoder's 16kHz mel rate)
audio_sample_rate = 24000


class LTX2StreamingEngine:
    """LTX-2 streaming video generator with rolling segment generation."""

    def __init__(self):
        print("Initializing LTX-2 Streaming Engine...", flush=True)

        # Select checkpoint
        if USE_FP8:
            self.ckpt = f"{LTX2_MODELS_DIR}/ltx-2-19b-distilled-fp8.safetensors"
            print(f"Using FP8 checkpoint (~27GB VRAM)", flush=True)
        else:
            self.ckpt = f"{LTX2_MODELS_DIR}/ltx-2-19b-distilled.safetensors"
            print(f"Using BF16 checkpoint (~43GB VRAM)", flush=True)

        self.spatial_upsampler = f"{LTX2_MODELS_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"

        # Verify files exist
        if not os.path.exists(self.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt}")
        if not os.path.exists(GEMMA_DIR):
            raise FileNotFoundError(f"Gemma model not found: {GEMMA_DIR}")

        # Load the distilled pipeline
        print("Loading DistilledPipeline...", flush=True)
        from ltx_pipelines.distilled import DistilledPipeline

        self.pipeline = DistilledPipeline(
            checkpoint_path=self.ckpt,
            spatial_upsampler_path=self.spatial_upsampler,
            gemma_root=GEMMA_DIR,
            loras=[],
            fp8transformer=USE_FP8,
        )

        # Pre-load models into VRAM
        print("Pre-loading models into VRAM...", flush=True)
        self._preload_models()

        # Run warmup
        print("Running warmup...", flush=True)
        self._warmup()

        print("LTX-2 Streaming Engine ready!", flush=True)
        self._print_memory_usage()

    def _preload_models(self):
        """Pre-load all models and patch to keep them in VRAM."""
        ledger = self.pipeline.model_ledger

        print("   Loading text encoder (Gemma)...", flush=True)
        self._text_encoder = ledger.text_encoder()

        print("   Loading transformer (19B)...", flush=True)
        self._transformer = ledger.transformer()

        # Apply torch.compile for faster inference
        # Note: Using "default" mode instead of "reduce-overhead" because CUDA graphs
        # require static shapes, but conditioning changes between segments
        print("   Compiling transformer with torch.compile (default mode)...", flush=True)
        self._transformer = torch.compile(
            self._transformer,
            mode="default",  # Kernel fusion without strict CUDA graph requirements
            fullgraph=False,  # Allow graph breaks for compatibility
        )

        print("   Loading VAE encoder...", flush=True)
        self._video_encoder = ledger.video_encoder()

        print("   Loading VAE decoder...", flush=True)
        self._video_decoder = ledger.video_decoder()

        print("   Loading audio decoder...", flush=True)
        self._audio_decoder = ledger.audio_decoder()

        print("   Loading vocoder...", flush=True)
        self._vocoder = ledger.vocoder()

        torch.cuda.synchronize()

        # Patch ledger to return cached models
        self._patch_model_ledger(ledger)

    def _patch_model_ledger(self, ledger):
        """Patch ModelLedger to return cached models instead of reloading."""
        cached_text_encoder = self._text_encoder
        cached_transformer = self._transformer
        cached_video_encoder = self._video_encoder
        cached_video_decoder = self._video_decoder
        cached_audio_decoder = self._audio_decoder
        cached_vocoder = self._vocoder

        ledger.text_encoder = lambda: cached_text_encoder
        ledger.transformer = lambda: cached_transformer
        ledger.video_encoder = lambda: cached_video_encoder
        ledger.video_decoder = lambda: cached_video_decoder
        ledger.audio_decoder = lambda: cached_audio_decoder
        ledger.vocoder = lambda: cached_vocoder

        # Disable cleanup to prevent model unloading
        ledger.cleanup_memory = lambda *args, **kwargs: None

        print("   ModelLedger patched - models stay in VRAM!", flush=True)

    def _warmup(self):
        """Run warmup generations to trigger torch.compile and CUDA graph capture."""
        from ltx_core.model.video_vae import TilingConfig

        # Use production resolution for warmup so CUDA graphs match
        warmup_height, warmup_width, warmup_frames = 480, 832, 49

        # Run multiple warmup iterations for torch.compile to optimize
        num_warmup_iters = 2
        print(f"   Warmup: {num_warmup_iters} iterations at {warmup_height}x{warmup_width}, {warmup_frames} frames...", flush=True)
        print("   (First iterations trigger torch.compile - may be slow)", flush=True)

        with torch.inference_mode():
            for i in range(num_warmup_iters):
                start_time = time.time()
                frames = self._generate_segment(
                    prompt="warmup test video generation",
                    seed=42 + i,
                    height=warmup_height,
                    width=warmup_width,
                    num_frames=warmup_frames,
                    frame_rate=24.0,
                    conditioning_latent=None,
                )
                # Consume frames
                frame_count = sum(1 for _ in frames)
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"   Warmup {i+1}/{num_warmup_iters}: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)", flush=True)

        torch.cuda.synchronize()
        print("   Warmup complete - CUDA graphs should be captured!", flush=True)

    def _print_memory_usage(self):
        """Print GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved", flush=True)

    def _generate_segment(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        conditioning_latent=None,
    ):
        """
        Generate a single video segment and yield frames.

        Args:
            conditioning_latent: Optional latent from previous segment for continuity

        Yields:
            numpy arrays of shape [H, W, 3] uint8
        """
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByKeyframeIndex
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
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

        # Encode text
        context_p = encode_text(self._text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        # Sigmas for 8-step distilled denoising
        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=self._transformer,
                ),
            )

        output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # Build conditionings
        conditionings = []
        if conditioning_latent is not None:
            # Use last frames from previous segment as conditioning
            video_conditioning = VideoConditionByKeyframeIndex(
                keyframes=conditioning_latent,
                frame_idx=0,
                strength=1.0,
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

        # Initialize audio state (empty)
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

        # Run 8-step denoising
        video_state, audio_state = denoising_loop(sigmas, video_state, audio_state, stepper)

        # Clear conditioning and unpatchify
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # Store latent for next segment
        self._last_segment_latent = video_state.latent.clone()

        # Decode audio
        audio_waveform = decode_audio(
            latent=audio_state.latent[:1],
            audio_decoder=self._audio_decoder,
            vocoder=self._vocoder,
        )
        # Convert to numpy int16 for WAV format
        audio_np = audio_waveform.cpu().numpy()
        audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        print(f"   Audio shape: {audio_np.shape}, sample_rate: {self._audio_decoder.sample_rate}", flush=True)

        # Decode video - yields frame chunks
        video_iterator = vae_decode_video(
            video_decoder=self._video_decoder,
            latent=video_state.latent[:1],
        )

        # Collect all chunks and process
        all_frames = []
        for chunk in video_iterator:
            if isinstance(chunk, torch.Tensor):
                all_frames.append(chunk)
            else:
                all_frames.append(torch.tensor(chunk))

        # Concatenate chunks - typically [B, T, H, W, C] or [B, C, T, H, W]
        if len(all_frames) > 0:
            video_tensor = torch.cat(all_frames, dim=0) if len(all_frames) > 1 else all_frames[0]

            # Debug shape
            print(f"   Video tensor shape: {video_tensor.shape}", flush=True)

            # Handle various tensor formats
            video = video_tensor.cpu().numpy()

            # Remove batch dimension if present
            while len(video.shape) > 4:
                video = video.squeeze(0)

            # Now should be [T, H, W, C] or [C, T, H, W] or [T, C, H, W]
            if len(video.shape) == 4:
                # Check if channels are first or last
                if video.shape[0] == 3:  # [C, T, H, W]
                    video = np.transpose(video, (1, 2, 3, 0))  # -> [T, H, W, C]
                elif video.shape[1] == 3:  # [T, C, H, W]
                    video = np.transpose(video, (0, 2, 3, 1))  # -> [T, H, W, C]
                # else assume [T, H, W, C] already

            # Normalize to 0-255 if needed
            if video.max() <= 1.0:
                video = video * 255
            video = np.clip(video, 0, 255).astype(np.uint8)

            # Yield individual frames as (frame, None) tuples
            for i in range(video.shape[0]):
                yield (video[i], None)

            # Yield audio at the end of segment as (None, audio_data) tuple
            yield (None, audio_np)

    def generate_rolling(
        self,
        prompt: str,
        seed: int,
        height: int = 480,
        width: int = 832,
        frame_rate: float = 24.0,
        segment_frames: int = 49,  # ~2 seconds at 24fps
        overlap_frames: int = 8,
        reset_history: bool = True,
    ):
        """
        Generator that yields frames and audio continuously using rolling segment generation.

        Each segment uses the last frames from the previous segment as conditioning
        for temporal coherence.

        Args:
            reset_history: If True, clears video history and starts fresh.
                          If False, continues from previous segment for smooth transitions.

        Yields:
            tuple: (frame, None) for video frames, (None, audio_data) for segment audio
        """
        if reset_history:
            self._last_segment_latent = None
        segment_idx = 0

        while True:
            seg_seed = seed + segment_idx

            # Get conditioning from previous segment
            conditioning_latent = None
            if self._last_segment_latent is not None:
                # Take last N latent frames for conditioning
                latent_overlap = max(1, overlap_frames // 8)  # Temporal compression ~8x
                conditioning_latent = self._last_segment_latent[:, :, -latent_overlap:, :, :]

            print(f"Generating segment {segment_idx + 1} ({segment_frames} frames)...", flush=True)
            start_time = time.time()

            frame_count = 0
            segment_audio = None
            for item in self._generate_segment(
                prompt=prompt,
                seed=seg_seed,
                height=height,
                width=width,
                num_frames=segment_frames,
                frame_rate=frame_rate,
                conditioning_latent=conditioning_latent,
            ):
                frame, audio = item

                if frame is not None:
                    # Skip overlap frames for non-first segments
                    if segment_idx > 0 and frame_count < overlap_frames:
                        frame_count += 1
                        continue

                    yield (frame, None)
                    frame_count += 1
                elif audio is not None:
                    # Store audio for the segment
                    segment_audio = audio

            # Yield audio at end of segment
            if segment_audio is not None:
                yield (None, segment_audio)

            gen_time = time.time() - start_time
            fps = frame_count / gen_time if gen_time > 0 else 0
            print(f"Segment {segment_idx + 1}: {frame_count} frames in {gen_time:.2f}s ({fps:.1f} FPS)", flush=True)

            # Skip memory cleanup between segments - we have enough VRAM
            # gc.collect()
            # torch.cuda.empty_cache()

            segment_idx += 1
            yield (None, None)  # Signal segment boundary


def apply_image_adjustments(frame: np.ndarray) -> np.ndarray:
    """Apply brightness, contrast, and gamma adjustments."""
    global brightness, contrast, gamma

    if brightness == 1.0 and contrast == 1.0 and gamma == 1.0:
        return frame

    f = frame.astype(np.float32) / 255.0

    # Gamma correction
    if gamma != 1.0:
        f = np.power(f, gamma)

    # Contrast around midpoint
    if contrast != 1.0:
        f = (f - 0.5) * contrast + 0.5

    # Brightness
    if brightness != 1.0:
        f = f * brightness

    return np.clip(f * 255, 0, 255).astype(np.uint8)


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert numpy frame to base64 JPEG string."""
    frame = apply_image_adjustments(frame)
    img = Image.fromarray(frame)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def audio_to_base64_wav(audio_data: np.ndarray, sample_rate: int) -> str:
    """Convert numpy audio to base64 WAV string."""
    import wave

    # Handle different audio shapes
    if len(audio_data.shape) == 1:
        # Already mono
        num_channels = 1
        audio_flat = audio_data
    elif len(audio_data.shape) == 2:
        if audio_data.shape[0] <= 2:
            # Shape is [channels, samples] - convert to mono by averaging
            num_channels = 1
            audio_flat = audio_data.mean(axis=0).astype(np.int16)
        else:
            # Shape is [samples, channels] - convert to mono by averaging
            num_channels = 1
            audio_flat = audio_data.mean(axis=1).astype(np.int16)
    else:
        # Flatten and hope for the best
        num_channels = 1
        audio_flat = audio_data.flatten()

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_flat.astype(np.int16).tobytes())

    wav_bytes = buffer.getvalue()
    audio_str = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{audio_str}"


@torch.inference_mode()
def generation_loop():
    """Main generation loop with rolling segments."""
    global is_generating, current_prompt, engine, frame_queue, audio_queue, hard_reset_requested, audio_sample_rate

    print("Starting rolling segment generation loop...", flush=True)

    last_prompt = current_prompt
    generator = None
    seed = 42

    # Get audio sample rate from vocoder (outputs 24kHz, not decoder's 16kHz mel rate)
    if engine is not None:
        audio_sample_rate = engine._vocoder.output_sample_rate
        print(f"Audio sample rate (vocoder output): {audio_sample_rate} Hz", flush=True)

    while is_generating:
        try:
            # Check for reset or new start
            if hard_reset_requested or generator is None:
                hard_reset_requested = False
                with prompt_lock:
                    last_prompt = current_prompt

                seed = int(time.time()) % 10000
                generator = engine.generate_rolling(
                    prompt=last_prompt,
                    seed=seed,
                    height=480,
                    width=832,
                    frame_rate=24.0,
                    segment_frames=49,  # ~2 seconds
                    overlap_frames=8,
                    reset_history=True,  # Fresh start
                )
                print(f"Started generation with prompt: {last_prompt[:50]}...", flush=True)

            # Check for prompt update
            with prompt_lock:
                if current_prompt != last_prompt:
                    last_prompt = current_prompt
                    seed = int(time.time()) % 10000
                    generator = engine.generate_rolling(
                        prompt=last_prompt,
                        seed=seed,
                        height=480,
                        width=832,
                        frame_rate=24.0,
                        segment_frames=49,
                        overlap_frames=8,
                        reset_history=False,  # Preserve continuity on prompt change
                    )
                    print(f"Prompt updated (preserving continuity): {last_prompt[:50]}...", flush=True)

            # Get next item (frame, audio) tuple
            item = next(generator)
            frame, audio = item

            if frame is None and audio is None:
                # Segment boundary, continue
                continue

            if frame is not None:
                # Convert and queue video frame
                base64_frame = frame_to_base64(frame)

                try:
                    frame_queue.put(base64_frame, timeout=2.0)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(base64_frame, timeout=1.0)
                    except:
                        pass

            if audio is not None:
                # Convert and queue audio chunk
                base64_audio = audio_to_base64_wav(audio, audio_sample_rate)

                try:
                    audio_queue.put(base64_audio, timeout=2.0)
                except queue.Full:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put(base64_audio, timeout=1.0)
                    except:
                        pass

        except StopIteration:
            # Generator exhausted (shouldn't happen with infinite rolling)
            generator = None
        except Exception as e:
            print(f"Error in generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(1)
            generator = None

    print("Generation loop stopped.", flush=True)


def emission_loop():
    """Emit frames to clients via WebSocket."""
    global is_generating, frame_queue

    frame_interval = 1.0 / 20.0  # 20 FPS display (matches generation speed)
    print(f"Starting emission loop at 20 FPS...", flush=True)

    frame_num = 0
    while is_generating:
        try:
            frame = frame_queue.get(timeout=1.0)
            socketio.emit('frame', {'image': frame})
            frame_num += 1
            if frame_num % 50 == 0:
                print(f"Emitted {frame_num} frames, queue: {frame_queue.qsize()}", flush=True)
            time.sleep(frame_interval)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in emission: {e}", flush=True)
            break

    print("Emission loop stopped.", flush=True)


def audio_emission_loop():
    """Emit audio chunks to clients via WebSocket."""
    global is_generating, audio_queue

    print("Starting audio emission loop...", flush=True)

    audio_num = 0
    while is_generating:
        try:
            audio = audio_queue.get(timeout=1.0)
            socketio.emit('audio', {'audio': audio})
            audio_num += 1
            print(f"Emitted audio chunk {audio_num}", flush=True)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio emission: {e}", flush=True)
            break

    print("Audio emission loop stopped.", flush=True)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LTX-2 Streaming Video Generation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #eee;
            min-height: 100vh;
        }
        h1 { color: #e94560; text-align: center; }
        .badge {
            text-align: center;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .container {
            background: rgba(22, 33, 62, 0.9);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(233, 69, 96, 0.2);
        }
        .video-container {
            background: #0a0a15;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            min-height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #videoFrame { max-width: 100%; border-radius: 8px; }
        .placeholder { color: #666; font-size: 18px; }
        .controls { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
        input[type="text"] {
            flex: 1; min-width: 250px; padding: 14px 18px;
            border: none; border-radius: 8px;
            background: rgba(10, 10, 21, 0.8); color: #fff;
            font-size: 15px; border: 1px solid rgba(233, 69, 96, 0.3);
        }
        input:focus { outline: none; border-color: #e94560; }
        button {
            padding: 14px 28px; border: none; border-radius: 8px;
            font-size: 14px; font-weight: 600; cursor: pointer;
            text-transform: uppercase; transition: transform 0.1s;
        }
        button:hover { transform: scale(1.02); }
        .btn-start { background: linear-gradient(135deg, #e94560, #ff6b6b); color: #fff; }
        .btn-stop { background: linear-gradient(135deg, #ff4757, #ff3344); color: #fff; }
        .btn-reset { background: linear-gradient(135deg, #0f3460, #16213e); color: #fff; border: 1px solid #e94560; }
        .status {
            text-align: center; padding: 12px; border-radius: 8px; margin-top: 12px;
        }
        .status.connected { background: rgba(233, 69, 96, 0.15); color: #e94560; }
        .status.generating { background: rgba(233, 69, 96, 0.25); color: #ff6b6b; animation: pulse 2s infinite; }
        .status.stopped { background: rgba(255, 71, 87, 0.15); color: #ff4757; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .info {
            font-size: 13px; color: #888; margin-top: 12px; text-align: center;
            background: rgba(233, 69, 96, 0.05); padding: 10px; border-radius: 8px;
        }
        .info strong { color: #e94560; }
        .slider-container {
            display: flex; align-items: center; gap: 12px; margin: 12px 0;
            background: rgba(10, 10, 21, 0.5); padding: 12px 16px; border-radius: 8px;
        }
        .slider-container label { color: #e94560; font-weight: 600; min-width: 100px; }
        .slider-container input[type="range"] {
            flex: 1; height: 6px; -webkit-appearance: none; appearance: none;
            background: linear-gradient(90deg, #16213e, #e94560); border-radius: 3px;
        }
        .slider-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 18px; height: 18px;
            background: #e94560; border-radius: 50%; cursor: pointer;
        }
        .slider-value { color: #e94560; font-weight: bold; min-width: 45px; text-align: right; }
    </style>
</head>
<body>
    <h1>LTX-2 Streaming Video</h1>
    <div class="badge">19B Distilled Model | 8-Step Inference | torch.compile | ~20 FPS</div>

    <div class="container">
        <div class="controls">
            <input type="text" id="promptInput" placeholder="Enter your prompt..."
                   value="A beautiful landscape with mountains and flowing water, cinematic, high quality">
            <button class="btn-start" onclick="updatePrompt()">Update Prompt</button>
        </div>
        <div class="controls">
            <button class="btn-start" onclick="startStream()" id="startBtn">Start Stream</button>
            <button class="btn-stop" onclick="stopStream()" id="stopBtn" disabled>Stop</button>
            <button class="btn-reset" onclick="hardReset()">Reset</button>
        </div>
        <div class="slider-container">
            <label>Brightness</label>
            <input type="range" id="brightnessSlider" min="0.5" max="1.5" step="0.05" value="1.0"
                   oninput="updateImageParam('brightness', this.value)">
            <span class="slider-value" id="brightnessValue">1.00</span>
        </div>
        <div class="slider-container">
            <label>Contrast</label>
            <input type="range" id="contrastSlider" min="0.5" max="2.0" step="0.05" value="1.0"
                   oninput="updateImageParam('contrast', this.value)">
            <span class="slider-value" id="contrastValue">1.00</span>
        </div>
        <div class="slider-container">
            <label>Gamma</label>
            <input type="range" id="gammaSlider" min="0.5" max="2.0" step="0.05" value="1.0"
                   oninput="updateImageParam('gamma', this.value)">
            <span class="slider-value" id="gammaValue">1.00</span>
        </div>
        <div class="slider-container">
            <label>Volume</label>
            <input type="range" id="volumeSlider" min="0" max="1" step="0.1" value="0.7">
            <button id="audioToggle" onclick="toggleAudio()" style="padding: 6px 12px; border: none; border-radius: 4px; background: #e94560; color: white; cursor: pointer; min-width: 70px;">Mute</button>
        </div>
        <div class="video-container">
            <img id="videoFrame" style="display:none;">
            <div class="placeholder" id="placeholder">Click "Start Stream" to begin</div>
        </div>
        <div class="status" id="status">Connecting...</div>
        <div class="info">
            <strong>LTX-2 19B Distilled:</strong> 8-step denoising with rolling segment generation and audio.<br>
            Each segment conditions on the previous for temporal coherence.
        </div>
    </div>

    <script>
        const socket = io();
        let isStreaming = false;

        // Audio playback queue
        let audioQueue = [];
        let isAudioPlaying = false;
        let audioEnabled = true;

        function playNextAudio() {
            if (audioQueue.length === 0 || !audioEnabled) {
                isAudioPlaying = false;
                return;
            }

            isAudioPlaying = true;
            const audioData = audioQueue.shift();
            const audio = new Audio(audioData);
            audio.volume = document.getElementById('volumeSlider').value;
            audio.onended = () => {
                playNextAudio();
            };
            audio.onerror = (e) => {
                console.error('Audio playback error:', e);
                playNextAudio();
            };
            audio.play().catch(e => {
                console.error('Audio play failed:', e);
                playNextAudio();
            });
        }

        socket.on('connect', () => {
            document.getElementById('status').textContent = 'Connected - Ready';
            document.getElementById('status').className = 'status connected';
        });

        socket.on('disconnect', () => {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = 'status stopped';
        });

        socket.on('frame', (data) => {
            const img = document.getElementById('videoFrame');
            const placeholder = document.getElementById('placeholder');
            img.src = data.image;
            img.style.display = 'block';
            placeholder.style.display = 'none';
        });

        socket.on('audio', (data) => {
            if (!audioEnabled) return;

            // Add to queue
            audioQueue.push(data.audio);

            // Start playing if not already
            if (!isAudioPlaying) {
                playNextAudio();
            }
        });

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            document.getElementById('audioToggle').textContent = audioEnabled ? 'Mute' : 'Unmute';
            if (!audioEnabled) {
                audioQueue = []; // Clear queue when muting
            }
        }

        function startStream() {
            const prompt = document.getElementById('promptInput').value;
            audioQueue = []; // Clear audio queue on start
            fetch('/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    isStreaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').textContent = 'Generating with rolling segments...';
                    document.getElementById('status').className = 'status generating';
                }
            });
        }

        function stopStream() {
            fetch('/stop', {method: 'POST'}).then(r => r.json()).then(data => {
                isStreaming = false;
                audioQueue = []; // Clear audio queue on stop
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('status').textContent = 'Stopped';
                document.getElementById('status').className = 'status stopped';
            });
        }

        function hardReset() {
            fetch('/hard_reset', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.success) {
                    document.getElementById('status').textContent = 'Reset!';
                }
            });
        }

        function updatePrompt() {
            const prompt = document.getElementById('promptInput').value;
            fetch('/update_prompt', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    document.getElementById('status').textContent = 'Prompt updated!';
                }
            });
        }

        function updateImageParam(param, value) {
            document.getElementById(param + 'Value').textContent = parseFloat(value).toFixed(2);
            fetch('/update_image_params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[param]: parseFloat(value)})
            });
        }

        document.getElementById('promptInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') updatePrompt();
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/start', methods=['POST'])
def start():
    global is_generating, current_prompt, frame_queue, audio_queue

    if is_generating:
        return jsonify({'success': False, 'message': 'Already generating'})

    data = request.json
    with prompt_lock:
        current_prompt = data.get('prompt', current_prompt)

    # Clear queues
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    is_generating = True

    threading.Thread(target=generation_loop, daemon=True).start()
    threading.Thread(target=emission_loop, daemon=True).start()
    threading.Thread(target=audio_emission_loop, daemon=True).start()

    return jsonify({'success': True})


@app.route('/stop', methods=['POST'])
def stop():
    global is_generating
    is_generating = False
    return jsonify({'success': True})


@app.route('/hard_reset', methods=['POST'])
def hard_reset():
    global hard_reset_requested
    hard_reset_requested = True
    return jsonify({'success': True})


@app.route('/update_prompt', methods=['POST'])
def update_prompt():
    global current_prompt
    data = request.json
    with prompt_lock:
        current_prompt = data.get('prompt', current_prompt)
    return jsonify({'success': True})


@app.route('/update_image_params', methods=['POST'])
def update_image_params():
    global brightness, contrast, gamma
    data = request.json
    if 'brightness' in data:
        brightness = float(data['brightness'])
    if 'contrast' in data:
        contrast = float(data['contrast'])
    if 'gamma' in data:
        gamma = float(data['gamma'])
    print(f"Image params: brightness={brightness}, contrast={contrast}, gamma={gamma}", flush=True)
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 60)
    print("LTX-2 Streaming Video Generation")
    print("19B Distilled Model | 8-Step Inference | Rolling Segments")
    print("=" * 60)
    print("\nInitializing model...")

    engine = LTX2StreamingEngine()

    print("\n" + "=" * 60)
    print("Server starting at http://localhost:5003")
    print("=" * 60 + "\n")

    socketio.run(app, host='0.0.0.0', port=5003, debug=False, allow_unsafe_werkzeug=True)

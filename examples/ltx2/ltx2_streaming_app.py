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

# Audio sample rate - vocoder outputs at 24kHz (not the decoder's 16kHz mel rate)
audio_sample_rate = 24000

# Target image conditioning (optional end-frame target)
target_image_latent = None
target_image_lock = threading.Lock()


def get_and_clear_target_image():
    """Get the target image latent and clear it (so it's only used for one segment)."""
    global target_image_latent
    with target_image_lock:
        latent = target_image_latent
        target_image_latent = None
    return latent


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

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode an image tensor to latent space for conditioning."""
        with torch.inference_mode():
            encoded = self._video_encoder(image_tensor)
        return encoded

    def _generate_segment(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        conditioning_latent=None,
        end_frame_latent=None,
    ):
        """
        Generate a single video segment and yield frames.

        Args:
            conditioning_latent: Optional latent from previous segment for continuity
            end_frame_latent: Optional latent for end-frame conditioning (target image)

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
            # Use last frames from previous segment as start-frame conditioning
            start_conditioning = VideoConditionByKeyframeIndex(
                keyframes=conditioning_latent,
                frame_idx=0,
                strength=1.0,
            )
            conditionings.append(start_conditioning)

        if end_frame_latent is not None:
            # Use target image as end-frame conditioning
            end_conditioning = VideoConditionByKeyframeIndex(
                keyframes=end_frame_latent,
                frame_idx=num_frames - 1,  # Last frame
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

            # Get target image latent (if set) and clear it so it's only used once
            end_frame_latent = get_and_clear_target_image()

            print(f"Generating segment {segment_idx + 1} ({segment_frames} frames){' with target image' if end_frame_latent is not None else ''}...", flush=True)
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
                end_frame_latent=end_frame_latent,
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


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert numpy frame to base64 JPEG string."""
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

    # No rate limiting here - client handles playback timing
    # Frames are emitted as fast as they're generated
    print(f"Starting emission loop (client-side buffered playback)...", flush=True)

    frame_num = 0
    while is_generating:
        try:
            frame = frame_queue.get(timeout=1.0)
            socketio.emit('frame', {'image': frame})
            frame_num += 1
            if frame_num % 50 == 0:
                print(f"Emitted {frame_num} frames, queue: {frame_queue.qsize()}", flush=True)
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
        .target-image-row {
            display: flex; gap: 8px; align-items: center; margin-bottom: 12px;
        }
        .target-image-preview {
            width: 80px; height: 45px; border-radius: 4px; background: #0a0a15;
            border: 1px dashed rgba(233, 69, 96, 0.3); display: flex; align-items: center;
            justify-content: center; overflow: hidden; flex-shrink: 0;
        }
        .target-image-preview img { width: 100%; height: 100%; object-fit: cover; }
        .target-image-preview .placeholder-text { color: #666; font-size: 9px; text-align: center; }
        .target-image-preview.has-image { border: 2px solid #4ade80; }
        .file-input-wrapper {
            position: relative; overflow: hidden; display: inline-block;
        }
        .file-input-wrapper input[type=file] {
            position: absolute; left: 0; top: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
        }
        .btn-upload { background: linear-gradient(135deg, #0f3460, #16213e); color: #fff; border: 1px solid #e94560; }
        .btn-clear { background: rgba(255, 71, 87, 0.8); color: #fff; padding: 4px 8px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; }
        .slider-value { color: #e94560; font-weight: bold; min-width: 30px; text-align: right; }
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
        <div class="target-image-row">
            <div class="target-image-preview" id="targetPreview">
                <span class="placeholder-text">No target</span>
            </div>
            <div class="file-input-wrapper">
                <button class="btn-upload" style="padding: 8px 12px; font-size: 12px;">Target Image</button>
                <input type="file" id="targetImageInput" accept="image/*" onchange="stageTargetImage(this)">
            </div>
            <button class="btn-clear" onclick="clearTargetImage()" id="clearTargetBtn" style="display:none;">✕</button>
        </div>
        <div class="slider-container">
            <label>Playback FPS</label>
            <input type="range" id="fpsSlider" min="10" max="24" step="1" value="15" oninput="updateFPS(this.value)">
            <span class="slider-value" id="fpsValue">15</span>
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

        // ===== A/V Sync Buffer System =====
        // Generation: ~41 frames per ~2.5s segment = ~16.4 FPS effective rate
        // Playback must be slower than generation to build/maintain buffer
        const CONTENT_FPS = 24;  // Content is generated at 24 FPS
        let targetFPS = 15;  // Playback FPS (adjustable via slider)
        let frameDurationMS = 1000 / targetFPS;  // ms per frame
        const MIN_BUFFER_FRAMES = 10;  // Start playback after buffering this many frames

        function getAudioPlaybackRate() {
            return targetFPS / CONTENT_FPS;  // Slow audio to match video playback rate
        }

        function updateFPS(value) {
            targetFPS = parseInt(value);
            frameDurationMS = 1000 / targetFPS;
            document.getElementById('fpsValue').textContent = value;
            console.log('Playback FPS set to', targetFPS, '- audio rate:', getAudioPlaybackRate().toFixed(3));
        }

        // Frame buffer
        let frameBuffer = [];
        let isPlaying = false;
        let lastFrameTime = 0;  // Timestamp of last frame display
        let framesPlayed = 0;

        // Audio system using Web Audio API for precise timing
        let audioContext = null;
        let audioEnabled = true;
        let pendingAudioChunks = [];  // Audio waiting to be scheduled
        let nextAudioStartTime = 0;   // When next audio chunk should start (in audioContext time)

        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }

        async function decodeAndScheduleAudio(base64Audio, startTime) {
            if (!audioEnabled || !audioContext) return;

            try {
                // Decode base64 to array buffer
                const base64Data = base64Audio.split(',')[1];
                const binaryString = atob(base64Data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }

                // Decode audio data
                const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);

                // Create source and gain for volume control
                const source = audioContext.createBufferSource();
                const gainNode = audioContext.createGain();
                gainNode.gain.value = parseFloat(document.getElementById('volumeSlider').value);

                source.buffer = audioBuffer;
                // Slow down audio to match video playback rate (this will lower pitch slightly)
                const audioRate = getAudioPlaybackRate();
                source.playbackRate.value = audioRate;
                source.connect(gainNode);
                gainNode.connect(audioContext.destination);

                // Schedule playback at precise time
                const scheduleTime = Math.max(startTime, audioContext.currentTime);
                source.start(scheduleTime);

                // Return adjusted duration for scheduling next chunk (longer due to slower playback)
                return audioBuffer.duration / audioRate;
            } catch (e) {
                console.error('Audio decode/schedule error:', e);
                return 0;
            }
        }

        function renderLoop(timestamp) {
            if (!isPlaying) return;

            // Rate-limited playback: only show one frame per frameDurationMS
            const timeSinceLastFrame = timestamp - lastFrameTime;

            if (timeSinceLastFrame >= frameDurationMS && frameBuffer.length > 0) {
                const frame = frameBuffer.shift();
                const img = document.getElementById('videoFrame');
                const placeholder = document.getElementById('placeholder');
                img.src = frame;
                img.style.display = 'block';
                placeholder.style.display = 'none';
                framesPlayed++;
                lastFrameTime = timestamp;
            }
            // If buffer is empty, we just wait - don't advance lastFrameTime

            // Update buffer status
            const bufferStatus = frameBuffer.length;
            if (bufferStatus < 5) {
                document.getElementById('status').textContent = `Playing (buffer: ${bufferStatus} - low!)`;
            } else {
                document.getElementById('status').textContent = `Playing (buffer: ${bufferStatus} frames)`;
            }

            requestAnimationFrame(renderLoop);
        }

        function startPlayback() {
            if (isPlaying) return;

            initAudioContext();
            isPlaying = true;
            lastFrameTime = performance.now();  // Initialize to now so first frame plays immediately
            framesPlayed = 0;
            nextAudioStartTime = audioContext.currentTime;

            // Schedule any pending audio
            processPendingAudio();

            requestAnimationFrame(renderLoop);
            console.log('Playback started with', frameBuffer.length, 'frames buffered');
        }

        function stopPlayback() {
            isPlaying = false;
            frameBuffer = [];
            pendingAudioChunks = [];
            framesPlayed = 0;
            lastFrameTime = 0;
        }

        async function processPendingAudio() {
            while (pendingAudioChunks.length > 0 && audioEnabled) {
                const audioData = pendingAudioChunks.shift();
                const duration = await decodeAndScheduleAudio(audioData, nextAudioStartTime);
                nextAudioStartTime += duration;
            }
        }

        socket.on('connect', () => {
            document.getElementById('status').textContent = 'Connected - Ready';
            document.getElementById('status').className = 'status connected';
        });

        socket.on('disconnect', () => {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = 'status stopped';
            stopPlayback();
        });

        socket.on('frame', (data) => {
            frameBuffer.push(data.image);

            // Start playback once we have enough buffered
            if (!isPlaying && frameBuffer.length >= MIN_BUFFER_FRAMES) {
                startPlayback();
            }
        });

        socket.on('audio', (data) => {
            if (!audioEnabled) return;

            // Queue audio for scheduling
            pendingAudioChunks.push(data.audio);

            // If already playing, process immediately
            if (isPlaying && audioContext) {
                processPendingAudio();
            }
        });

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            document.getElementById('audioToggle').textContent = audioEnabled ? 'Mute' : 'Unmute';
            if (!audioEnabled) {
                pendingAudioChunks = [];
            }
        }

        function startStream() {
            const prompt = document.getElementById('promptInput').value;
            stopPlayback();  // Reset playback state
            fetch('/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    isStreaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').textContent = 'Buffering...';
                    document.getElementById('status').className = 'status generating';
                }
            });
        }

        function stopStream() {
            fetch('/stop', {method: 'POST'}).then(r => r.json()).then(data => {
                isStreaming = false;
                stopPlayback();
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('status').textContent = 'Stopped';
                document.getElementById('status').className = 'status stopped';
            });
        }

        function hardReset() {
            stopPlayback();  // Clear buffer and reset playback state
            fetch('/hard_reset', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.success) {
                    document.getElementById('status').textContent = 'Reset - buffering...';
                }
            });
        }

        // Staged target image (not yet uploaded)
        let stagedImageFile = null;

        function stageTargetImage(input) {
            if (input.files && input.files[0]) {
                stagedImageFile = input.files[0];

                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('targetPreview');
                    preview.innerHTML = '<img src="' + e.target.result + '" alt="Target">';
                    preview.classList.add('has-image');
                };
                reader.readAsDataURL(stagedImageFile);

                document.getElementById('clearTargetBtn').style.display = 'inline-block';
            }
        }

        function clearTargetImage() {
            stagedImageFile = null;
            document.getElementById('targetPreview').innerHTML = '<span class="placeholder-text">No target</span>';
            document.getElementById('targetPreview').classList.remove('has-image');
            document.getElementById('clearTargetBtn').style.display = 'none';
            document.getElementById('targetImageInput').value = '';

            // Also clear on server if already uploaded
            fetch('/clear_target_image', {method: 'POST'});
        }

        async function updatePrompt() {
            const prompt = document.getElementById('promptInput').value;

            // If there's a staged image, upload it first
            if (stagedImageFile) {
                const formData = new FormData();
                formData.append('image', stagedImageFile);

                try {
                    const response = await fetch('/upload_target_image', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (data.success) {
                        console.log('Target image uploaded');
                    } else {
                        console.error('Failed to upload target image:', data.message);
                    }
                } catch (err) {
                    console.error('Error uploading target image:', err);
                }
                // Clear staged file after upload (it's now on the server)
                stagedImageFile = null;
            } else {
                // No staged image - clear any existing target on server
                await fetch('/clear_target_image', {method: 'POST'});
            }

            // Now update the prompt
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


@app.route('/upload_target_image', methods=['POST'])
def upload_target_image():
    """Upload and encode a target image for end-frame conditioning."""
    global target_image_latent, engine

    if engine is None:
        return jsonify({'success': False, 'message': 'Engine not initialized'})

    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'})

    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')

        # Resize to match generation resolution (480x832)
        target_height, target_width = 480, 832
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Convert to tensor: [B, C, T, H, W] where T=1 for single frame
        # Normalize to [-1, 1] range (same as LTX's normalize_latent: x / 127.5 - 1.0)
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0  # [H, W, C] in [-1, 1] range
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
        img_tensor = img_tensor.to(dtype=torch.bfloat16, device=engine.pipeline.device)

        # Encode to latent space
        with torch.inference_mode():
            encoded_latent = engine.encode_image(img_tensor)

        # Store globally
        with target_image_lock:
            target_image_latent = encoded_latent

        print(f"Target image uploaded and encoded. Latent shape: {encoded_latent.shape}", flush=True)
        return jsonify({'success': True, 'message': 'Target image set'})

    except Exception as e:
        print(f"Error encoding target image: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/clear_target_image', methods=['POST'])
def clear_target_image():
    """Clear the target image conditioning."""
    global target_image_latent

    with target_image_lock:
        target_image_latent = None

    print("Target image cleared", flush=True)
    return jsonify({'success': True, 'message': 'Target image cleared'})


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

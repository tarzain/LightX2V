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
hard_reset_requested = False

# Image adjustment parameters
brightness = 1.0
contrast = 1.0
gamma = 1.0


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

        print("   Loading VAE encoder...", flush=True)
        self._video_encoder = ledger.video_encoder()

        print("   Loading VAE decoder...", flush=True)
        self._video_decoder = ledger.video_decoder()

        torch.cuda.synchronize()

        # Patch ledger to return cached models
        self._patch_model_ledger(ledger)

    def _patch_model_ledger(self, ledger):
        """Patch ModelLedger to return cached models instead of reloading."""
        cached_text_encoder = self._text_encoder
        cached_transformer = self._transformer
        cached_video_encoder = self._video_encoder
        cached_video_decoder = self._video_decoder

        ledger.text_encoder = lambda: cached_text_encoder
        ledger.transformer = lambda: cached_transformer
        ledger.video_encoder = lambda: cached_video_encoder
        ledger.video_decoder = lambda: cached_video_decoder

        # Disable cleanup to prevent model unloading
        ledger.cleanup_memory = lambda *args, **kwargs: None

        print("   ModelLedger patched - models stay in VRAM!", flush=True)

    def _warmup(self):
        """Run a warmup generation."""
        from ltx_core.model.video_vae import TilingConfig

        warmup_height, warmup_width, warmup_frames = 512, 768, 17

        print(f"   Warmup: {warmup_height}x{warmup_width}, {warmup_frames} frames...", flush=True)

        with torch.inference_mode():
            frames = self._generate_segment(
                prompt="warmup test",
                seed=42,
                height=warmup_height,
                width=warmup_width,
                num_frames=warmup_frames,
                frame_rate=24.0,
                conditioning_latent=None,
            )
            # Consume frames
            for _ in frames:
                pass

        torch.cuda.synchronize()
        print("   Warmup complete!", flush=True)

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

            # Yield individual frames
            for i in range(video.shape[0]):
                yield video[i]

    def generate_rolling(
        self,
        prompt: str,
        seed: int,
        height: int = 480,
        width: int = 832,
        frame_rate: float = 24.0,
        segment_frames: int = 49,  # ~2 seconds at 24fps
        overlap_frames: int = 8,
    ):
        """
        Generator that yields frames continuously using rolling segment generation.

        Each segment uses the last frames from the previous segment as conditioning
        for temporal coherence.
        """
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
            for frame in self._generate_segment(
                prompt=prompt,
                seed=seg_seed,
                height=height,
                width=width,
                num_frames=segment_frames,
                frame_rate=frame_rate,
                conditioning_latent=conditioning_latent,
            ):
                # Skip overlap frames for non-first segments
                if segment_idx > 0 and frame_count < overlap_frames:
                    frame_count += 1
                    continue

                yield frame
                frame_count += 1

            gen_time = time.time() - start_time
            fps = frame_count / gen_time if gen_time > 0 else 0
            print(f"Segment {segment_idx + 1}: {frame_count} frames in {gen_time:.2f}s ({fps:.1f} FPS)", flush=True)

            # Memory cleanup between segments
            gc.collect()
            torch.cuda.empty_cache()

            segment_idx += 1
            yield None  # Signal segment boundary


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


@torch.inference_mode()
def generation_loop():
    """Main generation loop with rolling segments."""
    global is_generating, current_prompt, engine, frame_queue, hard_reset_requested

    print("Starting rolling segment generation loop...", flush=True)

    last_prompt = current_prompt
    generator = None
    seed = 42

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
                    )
                    print(f"Prompt updated: {last_prompt[:50]}...", flush=True)

            # Get next frame
            frame = next(generator)

            if frame is None:
                # Segment boundary, continue
                continue

            # Convert and queue
            base64_frame = frame_to_base64(frame)

            try:
                frame_queue.put(base64_frame, timeout=2.0)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put(base64_frame, timeout=1.0)
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

    frame_interval = 1.0 / 19.0  # 19 FPS display (matches generation speed)
    print(f"Starting emission loop at 19 FPS...", flush=True)

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
    <div class="badge">19B Distilled Model | 8-Step Inference | Rolling Segments | ~11 FPS</div>

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
        <div class="video-container">
            <img id="videoFrame" style="display:none;">
            <div class="placeholder" id="placeholder">Click "Start Stream" to begin</div>
        </div>
        <div class="status" id="status">Connecting...</div>
        <div class="info">
            <strong>LTX-2 19B Distilled:</strong> 8-step denoising with rolling segment generation.<br>
            Each segment conditions on the previous for temporal coherence.
        </div>
    </div>

    <script>
        const socket = io();
        let isStreaming = false;

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

        function startStream() {
            const prompt = document.getElementById('promptInput').value;
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
    global is_generating, current_prompt, frame_queue

    if is_generating:
        return jsonify({'success': False, 'message': 'Already generating'})

    data = request.json
    with prompt_lock:
        current_prompt = data.get('prompt', current_prompt)

    # Clear queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

    is_generating = True

    threading.Thread(target=generation_loop, daemon=True).start()
    threading.Thread(target=emission_loop, daemon=True).start()

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

# LTX-2 Video Generation API Documentation

This document describes the REST API and WebSocket API available in the `ltx2_official_modal.py` Modal deployment for LTX-2 video generation.

## Overview

The LTX-2 API provides multiple interfaces for video generation:

1. **REST API** - Synchronous endpoints for batch video generation
2. **WebSocket Streaming API** - Real-time video streaming with frame-by-frame delivery
3. **Gemini Live WebSocket API** - Voice-driven video generation powered by Gemini Live

## Deployment Architecture

The deployment runs on Modal with the following components:

- **GPU Engine (`OfficialLTX2Engine`)**: H200 GPU instance with preloaded models (~60GB VRAM)
- **Streaming UI**: Lightweight CPU-only function serving the web interface
- **Model**: LTX-2 19B parameter distilled model with 8-step inference

---

## REST API Endpoints

### Base URL
```
https://<your-modal-deployment-url>
```

---

### `POST /api/generate`

**Unified video generation endpoint** supporting all conditioning combinations.

#### Request Format

`multipart/form-data`

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of the video to generate |
| `width` | integer | No | 768 | Video width in pixels (must be multiple of 64) |
| `height` | integer | No | 512 | Video height in pixels (must be multiple of 64) |
| `num_frames` | integer | No | 97 | Number of frames to generate |
| `seed` | integer | No | 42 | Random seed for reproducibility |
| `skip_upscaling` | string | No | "true" | `"true"` = 8 steps (faster), `"false"` = 12 steps with 2x upscaling |
| `rolling_mode` | string | No | "false" | Enable autoregressive long video generation |
| `segment_seconds` | float | No | 3.0 | Duration of each segment in rolling mode |
| `first_frame` | File | No | - | First frame image for I2V/FL2V conditioning |
| `last_frame` | File | No | - | Last frame image for FL2V conditioning |
| `audio` | File | No | - | Audio file (WAV, MP3) for A2V conditioning |
| `audio_conditioning_strength` | float | No | 0.3 | Audio conditioning strength (0.0-1.0) |
| `extend_video` | File | No | - | Video file to extend/continue |

#### Generation Modes

The endpoint automatically detects the generation mode based on provided inputs:

| Inputs | Mode | Description |
|--------|------|-------------|
| `prompt` only | Text-to-Video (T2V) | Generate video from text prompt |
| `prompt` + `first_frame` | Image-to-Video (I2V) | Generate video starting from first frame |
| `prompt` + `first_frame` + `last_frame` | First-Last-to-Video (FL2V) | Generate video with start and end frame constraints |
| `prompt` + `audio` | Audio-to-Video (A2V) | Generate video conditioned on audio |
| `prompt` + `first_frame` + `audio` | I2V + A2V | Combined image and audio conditioning |
| `prompt` + `extend_video` | Video Extension | Continue/extend existing video |

#### Response

- **Success (200)**: Returns `video/mp4` binary data
- **Error (500)**: JSON with error details

#### Example Request (cURL)

```bash
# Text-to-Video
curl -X POST "https://your-deployment/api/generate" \
  -F "prompt=A majestic eagle soaring through a golden sunset sky" \
  -F "width=768" \
  -F "height=512" \
  -F "num_frames=97" \
  -F "seed=42" \
  --output output.mp4

# Image-to-Video
curl -X POST "https://your-deployment/api/generate" \
  -F "prompt=A person walking through a forest" \
  -F "first_frame=@input.png" \
  -F "width=768" \
  -F "height=512" \
  --output output.mp4

# Audio-to-Video
curl -X POST "https://your-deployment/api/generate" \
  -F "prompt=A music visualization" \
  -F "audio=@music.mp3" \
  -F "audio_conditioning_strength=0.3" \
  --output output.mp4
```

---

### `POST /api/t2v` (Legacy)

**Text-to-Video generation.** *Deprecated: Use `/api/generate` instead.*

#### Request Format

`application/json`

#### Request Body

```json
{
  "prompt": "A majestic eagle soaring through a golden sunset sky",
  "width": 768,
  "height": 512,
  "num_frames": 97,
  "seed": 42
}
```

#### Response

`video/mp4` binary data

---

### `POST /api/i2v`

**Image-to-Video generation.**

#### Request Format

`multipart/form-data`

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Input image file |
| `prompt` | string | Yes | - | Text description |
| `width` | integer | No | 768 | Video width |
| `height` | integer | No | 512 | Video height |
| `num_frames` | integer | No | 97 | Number of frames |
| `seed` | integer | No | 42 | Random seed |

#### Response

`video/mp4` binary data

---

### `POST /api/fl2v`

**First+Last Frame to Video generation.**

#### Request Format

`multipart/form-data`

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `first_image` | File | Yes | - | First frame image |
| `last_image` | File | Yes | - | Last frame image |
| `prompt` | string | Yes | - | Text description |
| `width` | integer | No | 768 | Video width |
| `height` | integer | No | 512 | Video height |
| `num_frames` | integer | No | 97 | Number of frames |
| `seed` | integer | No | 42 | Random seed |

#### Response

`video/mp4` binary data

---

### `POST /api/a2v`

**Audio-to-Video generation.**

#### Request Format

`multipart/form-data`

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio` | File | Yes | - | Audio file (WAV, MP3, etc.) |
| `prompt` | string | Yes | - | Text description |
| `width` | integer | No | 768 | Video width |
| `height` | integer | No | 512 | Video height |
| `num_frames` | integer | No | 97 | Number of frames |
| `seed` | integer | No | 42 | Random seed |
| `image` | File | No | - | Optional first frame for combined I2V+A2V |
| `audio_conditioning_strength` | float | No | 0.3 | Conditioning strength (0.0 = preserve audio, 1.0 = full diffusion) |

#### Response

`video/mp4` binary data

---

### `GET /health`

**Health check endpoint.**

#### Response

```json
{
  "status": "healthy",
  "model": "LTX-2 19B DistilledPipeline (8-step, FP8)"
}
```

---

## WebSocket Streaming API

### Endpoint

```
wss://<your-modal-deployment-url>/ws/stream
```

The WebSocket API enables real-time video streaming with frame-by-frame delivery and dynamic control during generation.

---

### Client → Server Messages

All messages are JSON objects with an `action` field.

#### `start` - Begin Video Generation

Start continuous video generation with streaming output.

```json
{
  "action": "start",
  "prompt": "A beautiful landscape with mountains",
  "seed": 42,
  "height": 480,
  "width": 832,
  "num_frames": 49,
  "frame_rate": 24.0,
  "use_second_stage": false,
  "max_segments": 10,
  "start_image": "<base64_image_data>",
  "end_image": "<base64_image_data>",
  "start_audio": "<base64_audio_data>",
  "audio_strength": 0.3
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | No | "A beautiful landscape" | Text prompt for generation |
| `seed` | integer | No | 42 | Random seed |
| `height` | integer | No | 480 | Video height |
| `width` | integer | No | 832 | Video width |
| `num_frames` | integer | No | 49 | Frames per segment |
| `frame_rate` | float | No | 24.0 | Frame rate |
| `use_second_stage` | boolean | No | false | Enable 2x upscaling refinement |
| `max_segments` | integer | No | 10 | Maximum segments to generate |
| `start_image` | string | No | - | Base64 image for first frame (I2V) |
| `end_image` | string | No | - | Base64 image for target frame |
| `start_audio` | string | No | - | Base64 audio for conditioning |
| `audio_strength` | float | No | 0.3 | Audio conditioning strength |

---

#### `stop` - Stop Generation

Stop the current generation process.

```json
{
  "action": "stop"
}
```

---

#### `update_prompt` - Update Prompt Mid-Generation

Change the prompt for upcoming segments (takes effect on next segment).

```json
{
  "action": "update_prompt",
  "prompt": "New scene: ocean waves crashing on rocks"
}
```

---

#### `set_target_image` - Queue Target Image

Add a target image to the queue for upcoming segments. The image guides generation toward that visual.

```json
{
  "action": "set_target_image",
  "image": "<base64_image_data>",
  "height": 480,
  "width": 832,
  "position": 1.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | string | Yes | - | Base64-encoded image |
| `height` | integer | No | 480 | Target height for encoding |
| `width` | integer | No | 832 | Target width for encoding |
| `position` | float | No | 1.0 | Frame position (0.0=start, 0.5=middle, 1.0=end) |

---

#### `clear_target_image` - Clear Target Image Queue

Remove all queued target images.

```json
{
  "action": "clear_target_image"
}
```

---

#### `set_audio` - Queue Audio Conditioning

Add audio to the queue for upcoming segments. Long audio is automatically chunked.

```json
{
  "action": "set_audio",
  "audio": "<base64_audio_data>",
  "strength": 0.3,
  "num_frames": 49,
  "frame_rate": 30.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | string | Yes | - | Base64-encoded audio (WAV, MP3, etc.) |
| `strength` | float | No | 0.3 | Conditioning strength (0.0-1.0) |
| `num_frames` | integer | No | 49 | Frames per segment (for chunking) |
| `frame_rate` | float | No | 30.0 | Frame rate (for chunking) |

---

#### `clear_audio` - Clear Audio Queue

Remove all queued audio.

```json
{
  "action": "clear_audio"
}
```

---

#### `reset_history` - Reset Generation State

Clear the model's conditioning state. The next segment will be generated fresh (no continuity from previous segment).

```json
{
  "action": "reset_history"
}
```

---

#### `set_next_segment` - Combined Segment Configuration

Set multiple parameters for the next segment in a single message.

```json
{
  "action": "set_next_segment",
  "prompt": "A sunset over the ocean",
  "target_image": "<base64_image_data>",
  "position": 1.0,
  "audio": "<base64_audio_data>",
  "audio_strength": 0.3,
  "height": 480,
  "width": 832,
  "num_frames": 49,
  "frame_rate": 30.0
}
```

All fields except `action` are optional.

---

### Server → Client Messages

All messages are JSON objects with a `type` field.

#### `started` - Generation Started

Sent when generation begins.

```json
{
  "type": "started",
  "prompt": "A beautiful landscape",
  "start_image_set": true,
  "end_image_set": true,
  "start_audio_set": true,
  "audio_chunks": 3
}
```

---

#### `segment_start` - Segment Beginning

Sent when a new segment begins generating.

```json
{
  "type": "segment_start",
  "segment": 1,
  "prompt": "A beautiful landscape",
  "has_target_image": false
}
```

---

#### `frame` - Video Frame

Individual frame data (JPEG encoded).

```json
{
  "type": "frame",
  "data": "<base64_jpeg_data>",
  "index": 0,
  "segment": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `data` | string | Base64-encoded JPEG image |
| `index` | integer | Frame index within segment |
| `segment` | integer | Segment number |

---

#### `audio` - Audio Data

Audio for the segment (WAV encoded).

```json
{
  "type": "audio",
  "data": "<base64_wav_data>",
  "sample_rate": 24000
}
```

---

#### `segment_complete` - Segment Finished

Sent when a segment finishes generating.

```json
{
  "type": "segment_complete",
  "segment": 1,
  "frames": 49
}
```

---

#### `complete` - Generation Complete

All segments have been generated.

```json
{
  "type": "complete",
  "segments": 10
}
```

---

#### `stopped` - Generation Stopped

Sent in response to a `stop` action.

```json
{
  "type": "stopped"
}
```

---

#### `prompt_updated` - Prompt Changed

Confirmation that the prompt was updated.

```json
{
  "type": "prompt_updated",
  "prompt": "New prompt text"
}
```

---

#### `target_image_set` - Target Image Queued

Confirmation that a target image was added to the queue.

```json
{
  "type": "target_image_set",
  "success": true,
  "position": 1.0,
  "queue_length": 1
}
```

---

#### `target_image_cleared` - Target Image Queue Cleared

```json
{
  "type": "target_image_cleared"
}
```

---

#### `target_image_used` - Target Image Consumed

A target image was used for the current segment.

```json
{
  "type": "target_image_used"
}
```

---

#### `audio_set` - Audio Queued

Confirmation that audio was added to the queue.

```json
{
  "type": "audio_set",
  "success": true,
  "strength": 0.3,
  "chunks_added": 2,
  "queue_length": 2
}
```

---

#### `audio_cleared` - Audio Queue Cleared

```json
{
  "type": "audio_cleared"
}
```

---

#### `audio_used` - Audio Consumed

Audio was used for the current segment.

```json
{
  "type": "audio_used"
}
```

---

#### `history_reset` - State Reset

The generation state was reset.

```json
{
  "type": "history_reset"
}
```

---

#### `next_segment_set` - Combined Configuration Applied

Response to `set_next_segment`.

```json
{
  "type": "next_segment_set",
  "prompt": "New prompt",
  "target_image_set": true,
  "target_image_queue_length": 1,
  "audio_set": true,
  "audio_chunks_added": 1,
  "audio_queue_length": 1
}
```

---

#### `error` - Error Occurred

```json
{
  "type": "error",
  "message": "Error description"
}
```

---

### JavaScript WebSocket Example

```javascript
const ws = new WebSocket('wss://your-deployment/ws/stream');

ws.onopen = () => {
  // Start generation
  ws.send(JSON.stringify({
    action: 'start',
    prompt: 'A beautiful sunset over the ocean',
    seed: 42,
    height: 480,
    width: 832,
    num_frames: 49,
    frame_rate: 24.0,
    max_segments: 5
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  switch (msg.type) {
    case 'frame':
      // Display frame
      const img = new Image();
      img.src = 'data:image/jpeg;base64,' + msg.data;
      document.getElementById('canvas').appendChild(img);
      break;
      
    case 'audio':
      // Play audio
      const audio = new Audio('data:audio/wav;base64,' + msg.data);
      audio.play();
      break;
      
    case 'segment_complete':
      console.log(`Segment ${msg.segment} complete`);
      break;
      
    case 'complete':
      console.log(`All ${msg.segments} segments complete`);
      break;
      
    case 'error':
      console.error('Error:', msg.message);
      break;
  }
};

// Update prompt mid-generation
function updatePrompt(newPrompt) {
  ws.send(JSON.stringify({
    action: 'update_prompt',
    prompt: newPrompt
  }));
}

// Set target image
function setTargetImage(base64Data) {
  ws.send(JSON.stringify({
    action: 'set_target_image',
    image: base64Data,
    position: 1.0  // End of segment
  }));
}

// Stop generation
function stop() {
  ws.send(JSON.stringify({ action: 'stop' }));
}
```

---

## Gemini Live WebSocket API

### Endpoint

```
wss://<your-modal-deployment-url>/ws/gemini-live
```

The Gemini Live API enables voice-driven video generation using Google's Gemini Live model for real-time audio conversations.

---

### Client → Server Messages

#### `start` - Start Gemini Session

Begin a Gemini Live session with video generation.

```json
{
  "action": "start",
  "api_key": "<your_gemini_api_key>",
  "prompt": "A beautiful abstract flowing visualization"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_key` | string | Yes | Your Google Gemini API key |
| `prompt` | string | No | Initial video generation prompt |

---

#### `audio` - Send User Audio

Send user microphone audio to Gemini.

```json
{
  "action": "audio",
  "data": "<base64_pcm_audio>"
}
```

Audio should be 16-bit PCM at 24kHz.

---

#### `stop` - Stop Session

End the Gemini Live session.

```json
{
  "action": "stop"
}
```

---

#### `update_prompt` - Update Video Prompt

Change the video generation prompt.

```json
{
  "action": "update_prompt",
  "prompt": "New visualization style"
}
```

---

### Server → Client Messages

#### `gemini_connected` - Session Established

```json
{
  "type": "gemini_connected"
}
```

---

#### `gemini_text` - Gemini Text Response

Text output from Gemini.

```json
{
  "type": "gemini_text",
  "text": "I'll create a flowing ocean scene for you..."
}
```

---

#### `gemini_audio` - Gemini Audio Response

Audio response from Gemini (used to condition video generation).

```json
{
  "type": "gemini_audio",
  "data": "<base64_pcm_audio>"
}
```

---

#### `gemini_turn_complete` - Gemini Response Complete

```json
{
  "type": "gemini_turn_complete"
}
```

---

#### `segment_start` / `frame` / `segment_complete`

Same format as the standard WebSocket streaming API.

---

## Configuration Parameters

### Video Dimensions

Recommended resolutions (must be multiples of 64):

| Resolution | Aspect Ratio | Use Case |
|------------|--------------|----------|
| 768×512 | 3:2 | Standard |
| 832×480 | 16:9 | Widescreen |
| 512×768 | 2:3 | Portrait |
| 1024×576 | 16:9 | High quality |

### Frame Configuration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `num_frames` | 49-97 | 49 frames ≈ 2s at 24fps, 97 frames ≈ 3s at 30fps |
| `frame_rate` | 24.0 or 30.0 | Higher rates = smoother but more compute |

### Audio Conditioning Strength

| Value | Effect |
|-------|--------|
| 0.0 | Preserve original audio exactly (pass-through) |
| 0.3 | Default - balanced conditioning |
| 0.5 | Moderate diffusion |
| 1.0 | Full diffusion (may introduce artifacts) |

---

## Error Handling

All endpoints return appropriate HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 500 | Internal Server Error |

WebSocket errors are sent as `{"type": "error", "message": "..."}` messages.

---

## Rate Limits and Performance

- **Cold start**: ~30-60 seconds (model loading)
- **Warm container**: ~5-15 seconds per video (depending on length/resolution)
- **Streaming**: First frame in ~3-5 seconds, continuous at ~30fps
- **Container scaling**: Auto-scales based on demand, 5-minute scaledown window

---

## Example Use Cases

### 1. Interactive Video Generation

Use the WebSocket API with `update_prompt` and `set_target_image` to create interactive experiences where users can influence generation in real-time.

### 2. Audio-Reactive Visuals

Feed music or audio through `set_audio` to create visualizations that respond to sound.

### 3. Story-Driven Long Videos

Use rolling mode (`rolling_mode: true`) with the REST API to generate longer coherent videos with autoregressive segment generation.

### 4. Voice-Controlled Generation

Use the Gemini Live WebSocket for hands-free video creation through natural conversation.

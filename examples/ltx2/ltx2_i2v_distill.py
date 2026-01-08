"""
LTX-2 19B 8-Step Distilled Image-to-Video Example

This example demonstrates how to use the LTX-2 19B distilled model
for image-to-video generation with 8 inference steps.

Model: Lightricks LTX-2 19B Distilled
- 19 billion parameters
- 8-step inference (no CFG required)
- Default resolution: 1216x704 @ 30 FPS

Usage:
    python ltx2_i2v_distill.py --image_path /path/to/image.jpg

Requirements:
    - Model weights: Download from Hugging Face (Lightricks/LTX-2)
    - GPU with 24GB+ VRAM (or use offload config for lower VRAM)
"""

import argparse
import os

from lightx2v import LightX2VPipeline


def main():
    parser = argparse.ArgumentParser(description="LTX-2 19B Distilled I2V Example")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to LTX-2 model weights"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file (default: uses ltx2_i2v_distill_8step.json)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Camera slowly zooms in, cinematic motion",
        help="Text prompt describing desired motion"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_ltx2_i2v.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=97,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable CPU offloading for lower VRAM usage"
    )

    args = parser.parse_args()

    # Determine config path
    if args.config_path is None:
        config_dir = os.path.join(os.path.dirname(__file__), "../../configs/ltx2")
        config_path = os.path.join(config_dir, "ltx2_i2v_distill_8step.json")
    else:
        config_path = args.config_path

    # Initialize pipeline
    print(f"Loading LTX-2 model from {args.model_path}")
    print(f"Using config: {config_path}")

    pipeline = LightX2VPipeline(
        model_path=args.model_path,
        config_path=config_path,
    )

    # Override config with command line args
    pipeline.set_config({
        "target_video_length": args.num_frames,
    })

    # Generate video
    print(f"Generating video from image: {args.image_path}")
    print(f"Motion prompt: {args.prompt}")
    print(f"Frames: {args.num_frames}")

    result = pipeline.run(
        prompt=args.prompt,
        image_path=args.image_path,
        seed=args.seed,
        save_result_path=args.output_path,
    )

    print(f"Video saved to: {args.output_path}")

    return result


if __name__ == "__main__":
    main()

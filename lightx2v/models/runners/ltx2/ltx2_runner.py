import gc
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger

from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2, GET_RECORDER_MODE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("ltx2")
class LTX2Runner(DefaultRunner):
    """
    Runner for LTX-2 video generation model.

    Supports:
    - Text-to-video (t2v) generation
    - Image-to-video (i2v) generation
    - 8-step distilled inference (via LTX2DistillRunner)

    Model specifications:
    - 19B parameters
    - DiT architecture
    - Default resolution: 1216x704 @ 30 FPS
    - Supports up to 4K resolution

    Text encoder: T5EncoderModel (google/t5-v1_1-xxl)
    VAE: AutoencoderKLLTXVideo from diffusers
    """

    def __init__(self, config):
        # Set default values for LTX-2
        config.setdefault("hidden_size", 2048)
        config.setdefault("num_heads", 32)
        config.setdefault("head_dim", 64)
        config.setdefault("num_layers", 48)
        config.setdefault("in_channels", 128)
        config.setdefault("out_channels", 128)
        config.setdefault("patch_size", [1, 2, 2])
        config.setdefault("freq_dim", 256)
        config.setdefault("caption_channels", 4096)
        config.setdefault("rope_theta", 10000.0)

        # Default resolution settings
        config.setdefault("target_height", 704)
        config.setdefault("target_width", 1216)
        config.setdefault("target_video_length", 97)  # frames
        config.setdefault("fps", 30)

        # VAE settings
        config.setdefault("vae_stride", [4, 8, 8])

        # Text encoder settings
        config.setdefault("text_len", 256)

        super().__init__(config)

        self.target_size_config = {
            "480p": {"height": 480, "width": 854},
            "720p": {"height": 720, "width": 1280},
            "1080p": {"height": 1080, "width": 1920},
            "4k": {"height": 2160, "width": 3840},
        }

    def init_scheduler(self):
        """Initialize the scheduler for LTX-2."""
        if self.config.get("feature_caching", "NoCaching") == "NoCaching":
            self.scheduler = LTX2Scheduler(self.config)
        else:
            raise NotImplementedError(f"Feature caching not yet supported for LTX-2")

    def load_transformer(self):
        """Load the LTX-2 transformer model."""
        model = LTX2Model(
            self.config["model_path"],
            self.config,
            self.init_device,
            model_type="ltx2"
        )
        return model

    def load_text_encoder(self):
        """
        Load text encoder (T5EncoderModel for LTX-2).

        Uses T5EncoderModel from HuggingFace transformers, which is the
        standard text encoder for LTX-Video models.

        Returns list of text encoders for compatibility with base class.
        """
        from lightx2v.models.input_encoders.hf.ltx2.ltx2_text_encoder import LTX2TextEncoder

        text_encoder_path = self.config.get("text_encoder_path")
        if text_encoder_path is None:
            # Check if model_path contains text_encoder subfolder
            model_path = self.config["model_path"]
            if os.path.isdir(os.path.join(model_path, "text_encoder")):
                text_encoder_path = model_path
            else:
                # Fall back to default T5 model
                text_encoder_path = "google/t5-v1_1-xxl"
                logger.info(f"No text_encoder_path specified, using default: {text_encoder_path}")

        logger.info(f"Loading text encoder from {text_encoder_path}")

        cpu_offload = self.config.get("text_encoder_cpu_offload", self.config.get("cpu_offload", False))
        device = torch.device("cpu") if cpu_offload else self.init_device

        text_encoder = LTX2TextEncoder(
            config=self.config,
            checkpoint_path=text_encoder_path,
            device=device,
            cpu_offload=cpu_offload,
            max_sequence_length=self.config.get("text_len", 256),
        )

        return [text_encoder]

    def load_image_encoder(self):
        """Load image encoder for i2v task."""
        if self.config.get("task") != "i2v":
            return None

        # LTX-2 uses VAE for image encoding in i2v
        return None

    def load_vae_encoder(self):
        """Load VAE encoder using AutoencoderKLLTXVideo from diffusers."""
        from lightx2v.models.video_encoders.hf.ltx2.ltx2_vae import LTX2VAE

        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)

        vae_path = self.config.get("vae_path")
        if vae_path is None:
            vae_path = self.config["model_path"]

        logger.info(f"Loading VAE from {vae_path}")

        vae = LTX2VAE(
            checkpoint_path=vae_path,
            device=vae_device,
            cpu_offload=vae_offload,
        )

        return vae

    def load_vae_decoder(self):
        """Load VAE decoder (same as encoder for LTX-2)."""
        # VAE encoder and decoder are the same model
        if hasattr(self, 'vae_encoder') and self.vae_encoder is not None:
            return self.vae_encoder
        return self.load_vae_encoder()

    def load_vae(self):
        """Load VAE encoder and decoder."""
        vae = self.load_vae_encoder()
        return vae, vae

    def get_latent_shape_with_target_hw(self):
        """Calculate latent shape based on target resolution."""
        vae_stride = self.config.get("vae_stride", [4, 8, 8])
        latent_shape = [
            self.config["in_channels"],
            (self.config["target_video_length"] - 1) // vae_stride[0] + 1,
            self.config["target_height"] // vae_stride[1],
            self.config["target_width"] // vae_stride[2],
        ]
        return latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        """Run input encoders for text-to-video task."""
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)

        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        """Run input encoders for image-to-video task."""
        img_ori = self.read_image_input(self.input_info.image_path)

        # Set latent shape based on image
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()

        # Encode image with VAE
        cond_latents = self.run_vae_encoder(img_ori)

        # Run text encoder
        text_encoder_output = self.run_text_encoder(self.input_info)

        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "cond_latents": cond_latents,
            },
        }

    def run_text_encoder(self, input_info):
        """Run text encoder on the prompt."""
        prompt = input_info.prompt_enhanced if self.config.get("use_prompt_enhancer", False) else input_info.prompt
        neg_prompt = input_info.negative_prompt or ""

        # Get embeddings from text encoder
        if self.config.get("enable_cfg", False):
            result = self.text_encoders[0].encode(
                prompt=prompt,
                negative_prompt=neg_prompt,
            )
        else:
            result = self.text_encoders[0].encode(prompt=prompt)

        return result

    def read_image_input(self, img_path):
        """Read and preprocess input image."""
        if isinstance(img_path, Image.Image):
            img_ori = img_path
        else:
            img_ori = Image.open(img_path).convert("RGB")
        return img_ori

    @ProfilingContext4DebugL1("Run VAE Encoder")
    def run_vae_encoder(self, image):
        """Encode image with VAE for i2v task."""
        target_width = self.config["target_width"]
        target_height = self.config["target_height"]

        # Resize and center crop
        transform = transforms.Compose([
            transforms.Resize(max(target_height, target_width), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # [C, H, W] -> [B, C, T, H, W] with T=1 for single image
        image_tensor = transform(image).unsqueeze(0).unsqueeze(2).to(AI_DEVICE)

        cond_latents = self.vae_encoder.encode(image_tensor)
        return cond_latents

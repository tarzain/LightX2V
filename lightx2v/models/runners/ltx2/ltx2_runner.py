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
        Load text encoder (Gemma 3 for LTX-2).

        Returns list of text encoders for compatibility with base class.
        """
        text_encoder_path = self.config.get("text_encoder_path")
        if text_encoder_path is None:
            text_encoder_path = os.path.join(self.config["model_path"], "text_encoder")

        logger.info(f"Loading text encoder from {text_encoder_path}")

        # For now, we'll use a placeholder that expects pre-computed embeddings
        # In production, this would load the Gemma 3 text encoder
        text_encoder = LTX2TextEncoderWrapper(
            self.config,
            text_encoder_path,
            self.init_device
        )

        return [text_encoder]

    def load_image_encoder(self):
        """Load image encoder for i2v task."""
        if self.config.get("task") != "i2v":
            return None

        # LTX-2 uses VAE for image encoding in i2v
        return None

    def load_vae_encoder(self):
        """Load VAE encoder."""
        from lightx2v.models.video_encoders.hf.ltx2.ltx2_vae import LTX2VAE

        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)

        vae_path = self.config.get("vae_path")
        if vae_path is None:
            vae_path = os.path.join(self.config["model_path"], "vae")

        logger.info(f"Loading VAE from {vae_path}")

        vae = LTX2VAE(
            checkpoint_path=vae_path,
            device=vae_device,
            cpu_offload=vae_offload,
        )

        return vae

    def load_vae_decoder(self):
        """Load VAE decoder (same as encoder for LTX-2)."""
        return self.load_vae_encoder()

    def load_vae(self):
        """Load VAE encoder and decoder."""
        vae = self.load_vae_encoder()
        return vae, vae

    def get_latent_shape_with_target_hw(self):
        """Calculate latent shape based on target resolution."""
        latent_shape = [
            self.config["in_channels"],
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            self.config["target_height"] // self.config["vae_stride"][1],
            self.config["target_width"] // self.config["vae_stride"][2],
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
        context = self.text_encoders[0].encode(prompt)

        text_encoder_output = {
            "context": context,
        }

        if self.config.get("enable_cfg", False):
            context_null = self.text_encoders[0].encode(neg_prompt)
            text_encoder_output["context_null"] = context_null

        return text_encoder_output

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

        image_tensor = transform(image).unsqueeze(0).unsqueeze(2).to(AI_DEVICE)

        cond_latents = self.vae_encoder.encode(image_tensor)
        return cond_latents


class LTX2TextEncoderWrapper:
    """
    Wrapper for LTX-2 text encoder (Gemma 3).

    This wrapper provides a consistent interface for text encoding.
    In production, this would load and run the actual Gemma 3 model.
    """

    def __init__(self, config, checkpoint_path, device):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.cpu_offload = config.get("text_encoder_cpu_offload", config.get("cpu_offload", False))

        # Placeholder for actual model loading
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Gemma 3 text encoder model."""
        # For now, this is a placeholder
        # In production, would load from transformers:
        # from transformers import AutoModel, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        # self.model = AutoModel.from_pretrained(self.checkpoint_path)
        logger.info(f"Text encoder initialized (placeholder) from {self.checkpoint_path}")

    def encode(self, text):
        """
        Encode text to embeddings.

        Args:
            text: Input text string

        Returns:
            torch.Tensor: Text embeddings [1, seq_len, hidden_size]
        """
        # Placeholder implementation
        # Returns dummy embeddings for testing
        # In production, this would run the actual encoder
        max_length = self.config.get("text_len", 256)
        caption_channels = self.config.get("caption_channels", 4096)

        # Create placeholder embeddings
        embeddings = torch.zeros(
            1, max_length, caption_channels,
            dtype=torch.bfloat16,
            device=self.device
        )

        if self.cpu_offload:
            embeddings = embeddings.to(AI_DEVICE)

        return embeddings

    def infer(self, text):
        """Alias for encode method."""
        return self.encode(text)

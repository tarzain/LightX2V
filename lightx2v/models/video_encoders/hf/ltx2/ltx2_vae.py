import os

import torch
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class LTX2VAE:
    """
    VAE encoder/decoder for LTX-2 video generation model.

    Uses AutoencoderKLLTXVideo from diffusers library.

    The LTX-2 VAE has:
    - 128 latent channels
    - Spatial compression: 8x
    - Temporal compression: 4x
    - Scaling factor for latent normalization
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
        cpu_offload=False,
        dtype=None,
        parallel=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device if device is not None else torch.device(AI_DEVICE)
        self.cpu_offload = cpu_offload
        self.dtype = dtype if dtype is not None else GET_DTYPE()
        self.parallel = parallel

        # VAE parameters (will be updated from model config)
        self.latent_channels = 128
        self.spatial_compression = 8
        self.temporal_compression = 4

        self._load_model()

    def _load_model(self):
        """Load VAE model from checkpoint using diffusers."""
        try:
            from diffusers import AutoencoderKLLTXVideo
        except ImportError:
            raise ImportError(
                "diffusers library is required for LTX-2 VAE. "
                "Install it with: pip install diffusers>=0.32.0"
            )

        logger.info(f"Loading LTX-2 VAE from {self.checkpoint_path}")

        # Determine if path is a local directory or HuggingFace model ID
        if os.path.isdir(self.checkpoint_path):
            # Local path - check for vae subfolder
            vae_path = self.checkpoint_path
            if os.path.isdir(os.path.join(self.checkpoint_path, "vae")):
                vae_path = os.path.join(self.checkpoint_path, "vae")

            self.model = AutoencoderKLLTXVideo.from_pretrained(
                vae_path,
                torch_dtype=torch.float32,  # VAE typically needs float32 for precision
                local_files_only=True,
            )
        else:
            # HuggingFace model ID
            self.model = AutoencoderKLLTXVideo.from_pretrained(
                self.checkpoint_path,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

        # Get scaling factor from model config
        self.scaling_factor = self.model.config.scaling_factor

        # Update compression ratios from model config if available
        if hasattr(self.model.config, "latent_channels"):
            self.latent_channels = self.model.config.latent_channels

        # Move to device if not using CPU offload
        if not self.cpu_offload:
            self.model = self.model.to(self.device)

        # Enable memory efficient attention if available
        if hasattr(self.model, "enable_slicing"):
            self.model.enable_slicing()

        logger.info(f"LTX-2 VAE loaded successfully (scaling_factor={self.scaling_factor})")

    @torch.no_grad()
    def encode(self, x):
        """
        Encode video frames to latent space.

        Args:
            x: Video tensor [B, C, T, H, W] or image tensor [B, C, 1, H, W]
               Values should be in range [-1, 1]

        Returns:
            Latent tensor [B, latent_channels, T', H', W']
            where T' = ceil(T / temporal_compression)
                  H' = H / spatial_compression
                  W' = W / spatial_compression
        """
        if self.cpu_offload:
            self.model = self.model.to(AI_DEVICE)

        # Ensure input is on correct device
        x = x.to(self.model.device, dtype=torch.float32)

        # Encode using the VAE
        latent_dist = self.model.encode(x).latent_dist
        latents = latent_dist.sample()

        # Scale latents
        latents = latents * self.scaling_factor

        # Convert to inference dtype
        latents = latents.to(self.dtype)

        if self.cpu_offload:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        return latents

    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latents to video frames.

        Args:
            latents: Latent tensor [B, latent_channels, T', H', W']

        Returns:
            Video tensor [B, C, T, H, W] with values in range [-1, 1]
        """
        if self.cpu_offload:
            self.model = self.model.to(AI_DEVICE)

        # Ensure latents are on correct device and dtype
        latents = latents.to(self.model.device, dtype=torch.float32)

        # Unscale latents
        latents = latents / self.scaling_factor

        # Decode using the VAE
        video = self.model.decode(latents).sample

        if self.cpu_offload:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        return video

    @torch.no_grad()
    def decode_tiled(self, latents, tile_size=256, tile_overlap=32):
        """
        Decode latents using tiled decoding for memory efficiency.

        Args:
            latents: Latent tensor [B, latent_channels, T', H', W']
            tile_size: Size of each tile in pixels
            tile_overlap: Overlap between tiles in pixels

        Returns:
            Video tensor [B, C, T, H, W] with values in range [-1, 1]
        """
        if self.cpu_offload:
            self.model = self.model.to(AI_DEVICE)

        # Enable tiling if available
        if hasattr(self.model, "enable_tiling"):
            self.model.enable_tiling(
                tile_sample_min_height=tile_size,
                tile_sample_min_width=tile_size,
                tile_overlap_factor=tile_overlap / tile_size,
            )

        # Ensure latents are on correct device and dtype
        latents = latents.to(self.model.device, dtype=torch.float32)

        # Unscale latents
        latents = latents / self.scaling_factor

        # Decode using the VAE with tiling
        video = self.model.decode(latents).sample

        # Disable tiling after use
        if hasattr(self.model, "disable_tiling"):
            self.model.disable_tiling()

        if self.cpu_offload:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        return video

    def to(self, device):
        """Move VAE to specified device."""
        self.device = device
        if self.model is not None and not self.cpu_offload:
            self.model = self.model.to(device)
        return self

    def to_cuda(self):
        """Move VAE to CUDA."""
        return self.to(AI_DEVICE)

    def to_cpu(self):
        """Move VAE to CPU."""
        return self.to("cpu")

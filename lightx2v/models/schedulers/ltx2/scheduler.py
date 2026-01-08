import math

import torch
import torch.distributed as dist
from torch.nn import functional as F

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class LTX2Scheduler(BaseScheduler):
    """
    Scheduler for LTX-2 video generation model.
    Implements flow matching scheduler for diffusion process.
    """

    def __init__(self, config):
        super().__init__(config)
        self.reverse = True
        self.num_train_timesteps = 1000
        self.sample_shift = self.config.get("sample_shift", 3.0)
        self.keep_latents_dtype_in_scheduler = True
        self.sample_guide_scale = self.config.get("sample_guide_scale", 3.0)

        # LTX-2 specific parameters
        self.patch_size = self.config.get("patch_size", [1, 2, 2])
        self.patch_size_t = self.config.get("patch_size_t", 1)

        if self.config.get("seq_parallel", False):
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        """Prepare latents and timesteps for inference."""
        self.prepare_latents(seed, latent_shape, dtype=torch.bfloat16)
        self.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.sample_shift)
        self.cos_sin = self.prepare_rotary_pos_embed(latent_shape)

        # Handle image conditioning for i2v task
        if image_encoder_output is not None and self.config.get("task") == "i2v":
            self.cond_latents = image_encoder_output.get("cond_latents", None)
        else:
            self.cond_latents = None

    def prepare_latents(self, seed, latent_shape, dtype=torch.bfloat16):
        """Initialize random latents for denoising."""
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(
            1,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )

    def set_timesteps(self, num_inference_steps, device, shift):
        """Set up timesteps with optional time shifting."""
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift for flow matching
        if shift != 1.0:
            sigmas = self.time_shift(sigmas, shift)

        if not self.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(dtype=torch.float32, device=device)

    def time_shift(self, t: torch.Tensor, shift):
        """Apply time shift transformation for flow matching."""
        return (shift * t) / (1 + (shift - 1) * t)

    def step_post(self):
        """Post-step processing: update latents using flow prediction."""
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self.latents = sample + model_output * dt

    def prepare_rotary_pos_embed(self, latent_shape):
        """
        Prepare rotary positional embeddings for LTX-2 transformer.
        LTX-2 uses 3D rotary embeddings for video (time, height, width).
        """
        # latent_shape: [channels, frames, height, width]
        t = latent_shape[1]
        h = latent_shape[2]
        w = latent_shape[3]

        head_dim = self.config.get("head_dim", 64)
        rope_theta = self.config.get("rope_theta", 10000.0)

        # Dimensions for each axis
        dim_t = head_dim // 4
        dim_h = head_dim // 4
        dim_w = head_dim // 2

        # Generate frequencies for each dimension
        freqs_t = self._get_1d_rotary_pos_embed(dim_t, t, rope_theta)
        freqs_h = self._get_1d_rotary_pos_embed(dim_h, h, rope_theta)
        freqs_w = self._get_1d_rotary_pos_embed(dim_w, w, rope_theta)

        # Combine into 3D rope embeddings
        freqs_cos = torch.cat([freqs_t[0], freqs_h[0], freqs_w[0]], dim=-1)
        freqs_sin = torch.cat([freqs_t[1], freqs_h[1], freqs_w[1]], dim=-1)

        cos_sin = torch.stack([freqs_cos, freqs_sin], dim=-1).to(AI_DEVICE)

        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)
            seqlen = cos_sin.shape[0]
            padding_size = (world_size - (seqlen % world_size)) % world_size
            if padding_size > 0:
                cos_sin = F.pad(cos_sin, (0, 0, 0, 0, 0, padding_size))
            cos_sin = torch.chunk(cos_sin, world_size, dim=0)[cur_rank]

        return cos_sin

    def _get_1d_rotary_pos_embed(self, dim, seq_len, theta=10000.0):
        """Generate 1D rotary position embeddings."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin

    def clear(self):
        """Clear scheduler state."""
        if hasattr(self, "latents"):
            del self.latents
        if hasattr(self, "cos_sin"):
            del self.cos_sin

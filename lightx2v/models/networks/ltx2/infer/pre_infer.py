import torch

from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

from .module_io import GridOutput, LTX2PreInferModuleOutput


def sinusoidal_embedding_1d(dim, position):
    """Generate sinusoidal embeddings for position encoding."""
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=position.device) * -emb)
    emb = position.float().unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class LTX2PreInfer:
    """
    Pre-inference module for LTX-2 model.
    Handles input processing, patch embedding, and time embedding.
    """

    def __init__(self, config):
        self.config = config
        self.task = config.get("task", "t2v")
        self.hidden_size = config.get("hidden_size", 2048)
        self.freq_dim = config.get("freq_dim", 256)
        self.patch_size = config.get("patch_size", [1, 2, 2])
        self.in_channels = config.get("in_channels", 128)
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def set_scheduler(self, scheduler):
        """Set reference to scheduler for accessing latents and timesteps."""
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, inputs):
        """
        Run pre-inference processing.

        Args:
            weights: Model weights container
            inputs: Dictionary containing text_encoder_output and optionally image_encoder_output

        Returns:
            LTX2PreInferModuleOutput with processed inputs
        """
        x = self.scheduler.latents
        t = self.scheduler.timesteps[self.scheduler.step_index]

        # Get text conditioning
        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"].get("context_null", inputs["text_encoder_output"]["context"])

        # Handle image conditioning for i2v task
        if self.task == "i2v" and inputs.get("image_encoder_output") is not None:
            cond_latents = inputs["image_encoder_output"].get("cond_latents", None)
            if cond_latents is not None:
                # Concatenate image latents with noise latents along channel dim
                x = torch.cat([x, cond_latents], dim=1)

        # Patch embedding: [B, C, T, H, W] -> [B, hidden_size, T', H', W']
        x = weights.patch_embedding.apply(x)

        # Get grid sizes after patching
        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]

        # Flatten spatial dimensions: [B, hidden_size, T', H', W'] -> [B, T'*H'*W', hidden_size]
        x = x.flatten(2).transpose(1, 2).contiguous()

        # Time embedding
        t_emb = torch.tensor([t], dtype=torch.float32, device=x.device)
        embed = sinusoidal_embedding_1d(self.freq_dim, t_emb)

        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)

        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_1.apply(embed)

        # Text embedding projection
        if self.sensitive_layer_dtype != self.infer_dtype:
            context = weights.text_embedding.apply(context.to(self.sensitive_layer_dtype))
        else:
            context = weights.text_embedding.apply(context)

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        grid_sizes = GridOutput(
            tensor=torch.tensor([[grid_sizes_t, grid_sizes_h, grid_sizes_w]], dtype=torch.int32, device=x.device),
            tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w)
        )

        return LTX2PreInferModuleOutput(
            x=x.squeeze(0),
            embed=embed,
            context=context.squeeze(0) if context.dim() == 3 else context,
            grid_sizes=grid_sizes,
            cos_sin=self.scheduler.cos_sin,
            adapter_args={}
        )

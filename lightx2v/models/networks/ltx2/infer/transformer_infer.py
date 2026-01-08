import torch
import torch.nn.functional as F
import torch.distributed as dist

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.registry_factory import ROPE_REGISTER


def apply_rotary_pos_emb(q, k, cos_sin):
    """Apply rotary positional embeddings to query and key tensors."""
    # cos_sin: [seq_len, head_dim, 2] where last dim is [cos, sin]
    cos = cos_sin[..., 0]
    sin = cos_sin[..., 1]

    # Reshape for broadcast: [seq_len, 1, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def rotate_half(x):
    """Rotate half of the hidden dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class LTX2TransformerInfer(BaseTransformerInfer):
    """
    Transformer inference module for LTX-2 model.
    Implements DiT-style transformer blocks with RMS normalization.
    """

    def __init__(self, config):
        self.config = config
        self.task = config.get("task", "t2v")
        self.blocks_num = config.get("num_layers", 48)  # LTX-2 19B has more layers
        self.num_heads = config.get("num_heads", 32)
        self.head_dim = config.get("head_dim", 64)
        self.hidden_size = config.get("hidden_size", 2048)

        # Attention types
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.self_attn_type = config.get("self_attn_type", "flash_attn2")
        self.cross_attn_type = config.get("cross_attn_type", "flash_attn2")

        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

        # Sequence parallel settings
        if config.get("seq_parallel", False):
            self.seq_p_group = config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

        self.cos_sin = None

    def set_scheduler(self, scheduler):
        """Set reference to scheduler."""
        self.scheduler = scheduler

    def get_scheduler_values(self):
        """Get rotary embeddings from scheduler."""
        self.cos_sin = self.scheduler.cos_sin

    def reset_infer_states(self):
        """Reset inference state variables."""
        self.self_attn_cu_seqlens = None
        self.cross_attn_cu_seqlens_q = None
        self.cross_attn_cu_seqlens_kv = None

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        """
        Run transformer inference.

        Args:
            weights: Transformer weights container
            pre_infer_out: Output from pre-inference

        Returns:
            Transformed hidden states
        """
        self.get_scheduler_values()
        self.reset_infer_states()

        x = pre_infer_out.x
        embed = pre_infer_out.embed
        context = pre_infer_out.context

        # Run through transformer blocks
        for block_idx in range(len(weights.blocks)):
            x = self.infer_block(weights.blocks[block_idx], x, embed, context)

        # Final normalization and projection
        x = self.infer_non_blocks(weights, x, embed)

        return x

    def infer_block(self, block, x, embed, context):
        """
        Run inference for a single transformer block.

        LTX-2 uses DiT-style modulation with adaptive layer norm.
        """
        # Get modulation parameters from time embedding
        scale_shift = block.ada_norm.apply(embed)

        if scale_shift.dim() == 2:
            scale_shift = scale_shift.unsqueeze(0)

        # Split into scale and shift for each sub-layer
        # LTX-2 has 6 modulation parameters per block
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = scale_shift.chunk(6, dim=-1)

        # Self-attention with adaptive norm
        residual = x
        x = block.norm1.apply(x)
        x = x * (1 + scale_msa.squeeze()) + shift_msa.squeeze()

        # Self-attention
        x = self.infer_self_attn(block, x)
        x = gate_msa.squeeze() * x
        x = residual + x

        # Cross-attention (if context available)
        if context is not None and hasattr(block, 'cross_attn_q'):
            residual = x
            x = block.norm2.apply(x)
            x = self.infer_cross_attn(block, x, context)
            x = residual + x

        # Feed-forward with adaptive norm
        residual = x
        x = block.norm3.apply(x)
        x = x * (1 + scale_mlp.squeeze()) + shift_mlp.squeeze()
        x = self.infer_ffn(block, x)
        x = gate_mlp.squeeze() * x
        x = residual + x

        return x

    def infer_self_attn(self, block, x):
        """Run self-attention computation."""
        s, n, d = x.shape[0], self.num_heads, self.head_dim

        # Project to Q, K, V
        q = block.self_attn_q.apply(x).view(s, n, d)
        k = block.self_attn_k.apply(x).view(s, n, d)
        v = block.self_attn_v.apply(x).view(s, n, d)

        # Apply QK normalization (LTX-2 uses RMS norm across heads)
        if hasattr(block, 'self_attn_norm_q'):
            q = block.self_attn_norm_q.apply(q)
            k = block.self_attn_norm_k.apply(k)

        # Apply rotary position embeddings
        if self.cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, self.cos_sin)

        # Compute attention
        attn_out = self.compute_attention(q, k, v)

        # Output projection
        attn_out = block.self_attn_out.apply(attn_out.reshape(s, -1))

        return attn_out

    def infer_cross_attn(self, block, x, context):
        """Run cross-attention computation."""
        s_q = x.shape[0]
        s_kv = context.shape[0]
        n, d = self.num_heads, self.head_dim

        # Project to Q from hidden states, K, V from context
        q = block.cross_attn_q.apply(x).view(s_q, n, d)
        k = block.cross_attn_k.apply(context).view(s_kv, n, d)
        v = block.cross_attn_v.apply(context).view(s_kv, n, d)

        # Apply QK normalization
        if hasattr(block, 'cross_attn_norm_q'):
            q = block.cross_attn_norm_q.apply(q)
            k = block.cross_attn_norm_k.apply(k)

        # Compute attention
        attn_out = self.compute_cross_attention(q, k, v)

        # Output projection
        attn_out = block.cross_attn_out.apply(attn_out.reshape(s_q, -1))

        return attn_out

    def compute_attention(self, q, k, v):
        """Compute self-attention using configured backend."""
        # Simple scaled dot-product attention as fallback
        # In practice, this would use flash attention
        scale = 1.0 / (self.head_dim ** 0.5)

        # [seq, heads, dim] -> [1, heads, seq, dim]
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

        # [1, heads, seq, dim] -> [seq, heads, dim]
        return attn.transpose(1, 2).squeeze(0)

    def compute_cross_attention(self, q, k, v):
        """Compute cross-attention using configured backend."""
        scale = 1.0 / (self.head_dim ** 0.5)

        # [seq, heads, dim] -> [1, heads, seq, dim]
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

        # [1, heads, seq, dim] -> [seq, heads, dim]
        return attn.transpose(1, 2).squeeze(0)

    def infer_ffn(self, block, x):
        """Run feed-forward network computation."""
        # GELU-gated FFN
        hidden = block.ffn_fc1.apply(x)
        gate = block.ffn_gate.apply(x)
        hidden = hidden * torch.nn.functional.gelu(gate, approximate="tanh")
        out = block.ffn_fc2.apply(hidden)

        if self.clean_cuda_cache:
            del hidden, gate
            torch.cuda.empty_cache()

        return out

    def infer_non_blocks(self, weights, x, embed):
        """Run final normalization and output projection."""
        # Final adaptive norm
        scale_shift = weights.final_ada_norm.apply(embed)
        shift, scale = scale_shift.chunk(2, dim=-1)

        x = weights.final_norm.apply(x)
        x = x * (1 + scale.squeeze()) + shift.squeeze()

        # Output projection
        x = weights.out_proj.apply(x)

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return x

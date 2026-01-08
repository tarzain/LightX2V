from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class LTX2TransformerWeights(WeightModule):
    """
    Transformer weights for LTX-2 model.
    Contains weights for all transformer blocks.
    """

    def __init__(self, config, lazy_load_path=None):
        super().__init__()
        self.blocks_num = config.get("num_layers", 48)
        self.task = config.get("task", "t2v")
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")

        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True

        if config.get("do_mm_calib", False):
            self.mm_type = "Calib"
            assert not config.get("cpu_offload", False)

        self.lazy_load = config.get("lazy_load", False)

        # Create transformer blocks
        self.blocks = WeightModuleList([
            LTX2TransformerBlock(
                block_index=i,
                task=self.task,
                mm_type=self.mm_type,
                config=self.config,
                create_cuda_buffer=False,
                block_prefix="transformer_blocks",
            )
            for i in range(self.blocks_num)
        ])
        self.add_module("blocks", self.blocks)

        # Register offload buffers if needed
        self.register_offload_buffers(config, lazy_load_path)

        # Final layer weights
        self.register_parameter(
            "final_norm",
            LN_WEIGHT_REGISTER["Default"]("norm_out.norm.weight", "norm_out.norm.bias")
        )
        self.add_module(
            "final_ada_norm",
            MM_WEIGHT_REGISTER["Default"]("norm_out.linear.weight", "norm_out.linear.bias")
        )
        self.add_module(
            "out_proj",
            MM_WEIGHT_REGISTER["Default"]("proj_out.weight", "proj_out.bias")
        )

    def register_offload_buffers(self, config, lazy_load_path):
        """Register buffers for CPU offloading if enabled."""
        if config.get("cpu_offload", False):
            if config.get("offload_granularity", "block") == "block":
                self.offload_blocks_num = 2
                self.offload_block_cuda_buffers = WeightModuleList([
                    LTX2TransformerBlock(
                        block_index=i,
                        task=self.task,
                        mm_type=self.mm_type,
                        config=self.config,
                        create_cuda_buffer=True,
                        block_prefix="transformer_blocks",
                    )
                    for i in range(self.offload_blocks_num)
                ])
                self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
                self.offload_phase_cuda_buffers = None

    def non_block_weights_to_cuda(self):
        """Move non-block weights to CUDA."""
        self.final_norm.to_cuda()
        self.final_ada_norm.to_cuda()
        self.out_proj.to_cuda()

    def non_block_weights_to_cpu(self):
        """Move non-block weights to CPU."""
        self.final_norm.to_cpu()
        self.final_ada_norm.to_cpu()
        self.out_proj.to_cpu()


class LTX2TransformerBlock(WeightModule):
    """
    Single transformer block weights for LTX-2.
    Implements DiT-style block with adaptive layer norm.
    """

    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        block_prefix="transformer_blocks",
    ):
        super().__init__()
        self.block_index = block_index
        self.task = task
        self.mm_type = mm_type
        self.config = config
        self.hidden_size = config.get("hidden_size", 2048)
        self.num_heads = config.get("num_heads", 32)
        self.head_dim = config.get("head_dim", 64)

        prefix = f"{block_prefix}.{block_index}"

        # Register block weights
        self._register_attention_weights(prefix)
        self._register_ffn_weights(prefix)
        self._register_norm_weights(prefix)

    def _register_attention_weights(self, prefix):
        """Register self-attention and cross-attention weights."""
        mm_class = MM_WEIGHT_REGISTER.get(self.mm_type, MM_WEIGHT_REGISTER["Default"])
        rms_class = RMS_WEIGHT_REGISTER.get("Default")

        # Adaptive layer norm for modulation
        self.add_module(
            "ada_norm",
            MM_WEIGHT_REGISTER["Default"](f"{prefix}.scale_shift_table", None)
        )

        # Self-attention
        self.add_module(
            "self_attn_q",
            mm_class(f"{prefix}.attn1.to_q.weight", f"{prefix}.attn1.to_q.bias")
        )
        self.add_module(
            "self_attn_k",
            mm_class(f"{prefix}.attn1.to_k.weight", f"{prefix}.attn1.to_k.bias")
        )
        self.add_module(
            "self_attn_v",
            mm_class(f"{prefix}.attn1.to_v.weight", f"{prefix}.attn1.to_v.bias")
        )
        self.add_module(
            "self_attn_out",
            mm_class(f"{prefix}.attn1.to_out.0.weight", f"{prefix}.attn1.to_out.0.bias")
        )

        # QK normalization (RMS norm across heads)
        self.register_parameter(
            "self_attn_norm_q",
            rms_class(f"{prefix}.attn1.norm_q.weight")
        )
        self.register_parameter(
            "self_attn_norm_k",
            rms_class(f"{prefix}.attn1.norm_k.weight")
        )

        # Cross-attention
        self.add_module(
            "cross_attn_q",
            mm_class(f"{prefix}.attn2.to_q.weight", f"{prefix}.attn2.to_q.bias")
        )
        self.add_module(
            "cross_attn_k",
            mm_class(f"{prefix}.attn2.to_k.weight", f"{prefix}.attn2.to_k.bias")
        )
        self.add_module(
            "cross_attn_v",
            mm_class(f"{prefix}.attn2.to_v.weight", f"{prefix}.attn2.to_v.bias")
        )
        self.add_module(
            "cross_attn_out",
            mm_class(f"{prefix}.attn2.to_out.0.weight", f"{prefix}.attn2.to_out.0.bias")
        )

        # Cross-attention QK normalization
        self.register_parameter(
            "cross_attn_norm_q",
            rms_class(f"{prefix}.attn2.norm_q.weight")
        )
        self.register_parameter(
            "cross_attn_norm_k",
            rms_class(f"{prefix}.attn2.norm_k.weight")
        )

    def _register_ffn_weights(self, prefix):
        """Register feed-forward network weights."""
        mm_class = MM_WEIGHT_REGISTER.get(self.mm_type, MM_WEIGHT_REGISTER["Default"])

        # Gated FFN (SwiGLU/GELU-gated)
        self.add_module(
            "ffn_fc1",
            mm_class(f"{prefix}.ff.net.0.proj.weight", f"{prefix}.ff.net.0.proj.bias")
        )
        self.add_module(
            "ffn_gate",
            mm_class(f"{prefix}.ff.net.0.proj.weight", f"{prefix}.ff.net.0.proj.bias")  # Shared in some variants
        )
        self.add_module(
            "ffn_fc2",
            mm_class(f"{prefix}.ff.net.2.weight", f"{prefix}.ff.net.2.bias")
        )

    def _register_norm_weights(self, prefix):
        """Register layer normalization weights."""
        ln_class = LN_WEIGHT_REGISTER["Default"]

        # Pre-attention norms
        self.register_parameter(
            "norm1",
            ln_class(f"{prefix}.norm1.weight", f"{prefix}.norm1.bias")
        )
        self.register_parameter(
            "norm2",
            ln_class(f"{prefix}.norm2.weight", f"{prefix}.norm2.bias")
        )
        self.register_parameter(
            "norm3",
            ln_class(f"{prefix}.norm3.weight", f"{prefix}.norm3.bias")
        )

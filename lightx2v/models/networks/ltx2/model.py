import gc
import glob
import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.ltx2.infer.pre_infer import LTX2PreInfer
from lightx2v.models.networks.ltx2.infer.post_infer import LTX2PostInfer
from lightx2v.models.networks.ltx2.infer.transformer_infer import LTX2TransformerInfer
from lightx2v.models.networks.ltx2.weights.pre_weights import LTX2PreWeights
from lightx2v.models.networks.ltx2.weights.transformer_weights import LTX2TransformerWeights
from lightx2v.models.networks.ltx2.weights.post_weights import LTX2PostWeights
from lightx2v.utils.custom_compiler import CompiledMethodsMixin
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class LTX2Model(CompiledMethodsMixin):
    """
    LTX-2 Video Generation Model.

    A 19B parameter DiT-based model for high-quality video generation.
    Supports 8-step distilled inference for fast generation.

    Key features:
    - DiT (Diffusion Transformer) architecture
    - RMS normalization across attention heads
    - Adaptive layer norm for time conditioning
    - Supports both text-to-video (t2v) and image-to-video (i2v)
    """

    pre_weight_class = LTX2PreWeights
    transformer_weight_class = LTX2TransformerWeights
    post_weight_class = LTX2PostWeights

    def __init__(self, model_path, config, device, model_type="ltx2"):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model_type = model_type

        self.cpu_offload = config.get("cpu_offload", False)
        self.offload_granularity = config.get("offload_granularity", "block")
        self.dit_quantized = config.get("dit_quantized", False)

        if config.get("seq_parallel", False):
            self.seq_p_group = config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

        self.remove_keys = []

        # Initialize model components
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        """Initialize inference class types."""
        self.pre_infer_class = LTX2PreInfer
        self.post_infer_class = LTX2PostInfer
        self.transformer_infer_class = LTX2TransformerInfer

    def _init_weights(self, weight_dict=None):
        """Initialize and load model weights."""
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        sensitive_layer = {"norm", "embedding", "modulation", "time"}

        if weight_dict is None:
            if not self.dit_quantized:
                weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
            else:
                weight_dict = self._load_quant_ckpt(unified_dtype, sensitive_layer)

            self.original_weight_dict = weight_dict
        else:
            self.original_weight_dict = weight_dict

        # Initialize weight containers
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)

        self._apply_weights()

    def _apply_weights(self, weight_dict=None):
        """Apply loaded weights to weight containers."""
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            gc.collect()

        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        gc.collect()

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        """Load checkpoint from safetensors files."""
        if self.config.get("dit_original_ckpt", None):
            safetensors_path = self.config["dit_original_ckpt"]
        else:
            safetensors_path = self.model_path

        if os.path.isdir(safetensors_path):
            safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            safetensors_files = [safetensors_path]

        weight_dict = {}
        for file_path in safetensors_files:
            logger.info(f"Loading weights from {file_path}")
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)

        return weight_dict

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Load a single safetensor file to dictionary."""
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.device.type != "cpu" and dist.is_initialized():
            device = dist.get_rank()
        else:
            device = str(self.device)

        with safe_open(file_path, framework="pt", device=device) as f:
            return {
                key: (
                    f.get_tensor(key).to(GET_DTYPE())
                    if unified_dtype or all(s not in key for s in sensitive_layer)
                    else f.get_tensor(key).to(GET_SENSITIVE_DTYPE())
                )
                for key in f.keys()
                if not any(remove_key in key for remove_key in remove_keys)
            }

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        """Load quantized checkpoint."""
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.config.get("dit_quantized_ckpt", None):
            safetensors_path = self.config["dit_quantized_ckpt"]
        else:
            safetensors_path = self.model_path

        if os.path.isdir(safetensors_path):
            safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            safetensors_files = [safetensors_path]

        weight_dict = {}
        for safetensor_path in safetensors_files:
            with safe_open(safetensor_path, framework="pt") as f:
                logger.info(f"Loading quantized weights from {safetensor_path}")
                for k in f.keys():
                    if any(remove_key in k for remove_key in remove_keys):
                        continue
                    tensor = f.get_tensor(k)
                    if tensor.dtype in [torch.float16, torch.bfloat16, torch.float]:
                        if unified_dtype or all(s not in k for s in sensitive_layer):
                            weight_dict[k] = tensor.to(GET_DTYPE()).to(self.device)
                        else:
                            weight_dict[k] = tensor.to(GET_SENSITIVE_DTYPE()).to(self.device)
                    else:
                        weight_dict[k] = tensor.to(self.device)

        return weight_dict

    def _init_infer(self):
        """Initialize inference modules."""
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    def set_scheduler(self, scheduler):
        """Set scheduler reference for all inference modules."""
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)

    def to_cpu(self):
        """Move weights to CPU."""
        self.pre_weight.to_cpu()
        self.transformer_weights.to_cpu()

    def to_cuda(self):
        """Move weights to CUDA."""
        self.pre_weight.to_cuda()
        self.transformer_weights.to_cuda()

    @torch.no_grad()
    def infer(self, inputs):
        """
        Run model inference.

        Args:
            inputs: Dictionary containing:
                - text_encoder_output: Dict with 'context' tensor
                - image_encoder_output: Optional dict with 'cond_latents' for i2v

        The noise prediction is stored in scheduler.noise_pred
        """
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if self.config.get("enable_cfg", False):
            # Classifier-free guidance (not typically used for distilled model)
            noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
            noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)
            self.scheduler.noise_pred = (
                noise_pred_uncond +
                self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            )
        else:
            # Direct inference (standard for distilled model)
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        """Run inference with conditional or unconditional context."""
        self.scheduler.infer_condition = infer_condition

        # Pre-processing
        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        # Transformer
        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        # Post-processing
        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        return noise_pred

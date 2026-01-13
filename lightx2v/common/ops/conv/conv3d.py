from abc import ABCMeta, abstractmethod

import torch

from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class Conv3dWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.config = {}

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config


@CONV3D_WEIGHT_REGISTER("Default")
class Conv3dWeight(Conv3dWeightTemplate):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(weight_name, bias_name, stride, padding, dilation, groups)

    @staticmethod
    def _resolve_key(weight_dict, expected_key: str):
        """Resolve a possibly-prefixed key in a checkpoint dict.

        Some checkpoints prefix parameter names (e.g. 'model.', 'transformer.', 'module.').
        We fall back to a unique suffix match to keep weight templates stable.
        """
        if expected_key is None:
            return None
        # Some official LTX-2 checkpoints name the patch embed conv as `patchify_proj.*` instead of
        # `patch_embed.proj.*`. Add a small compatibility alias list here so weight loading works
        # even if upstream key naming differs.
        alias_keys = [expected_key]
        if expected_key == "patch_embed.proj.weight":
            alias_keys.append("patchify_proj.weight")
        elif expected_key == "patch_embed.proj.bias":
            alias_keys.append("patchify_proj.bias")

        for candidate in alias_keys:
            if candidate in weight_dict:
                return candidate
            matches = [k for k in weight_dict.keys() if k.endswith(candidate)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                matches.sort(key=len)
                return matches[0]

        return expected_key

    def load(self, weight_dict):
        weight_key = self._resolve_key(weight_dict, self.weight_name)
        bias_key = self._resolve_key(weight_dict, self.bias_name) if self.bias_name is not None else None

        device = weight_dict[weight_key].device
        if device.type == "cpu":
            weight_shape = weight_dict[weight_key].shape
            weight_dtype = weight_dict[weight_key].dtype
            self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
            self.pin_weight.copy_(weight_dict[weight_key])

            if self.bias_name is not None:
                bias_shape = weight_dict[bias_key].shape
                bias_dtype = weight_dict[bias_key].dtype
                self.pin_bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
                self.pin_bias.copy_(weight_dict[bias_key])
            else:
                self.bias = None
                self.pin_bias = None
            del weight_dict[weight_key]
        else:
            self.weight = weight_dict[weight_key]
            if self.bias_name is not None:
                self.bias = weight_dict[bias_key]
            else:
                self.bias = None

    def apply(self, input_tensor):
        input_tensor = torch.nn.functional.conv3d(
            input_tensor,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return input_tensor

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.to(AI_DEVICE, non_blocking=non_blocking)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.to(AI_DEVICE, non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight"):
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight  # .cpu().detach().clone().contiguous()
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias  # .cpu().detach().clone()
        return destination

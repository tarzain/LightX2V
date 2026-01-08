import math

import torch

from lightx2v.utils.envs import GET_DTYPE


class LTX2PostInfer:
    """
    Post-inference module for LTX-2 model.
    Handles unpatchifying and final projection.
    """

    def __init__(self, config):
        self.out_channels = config.get("out_channels", 128)
        self.patch_size = tuple(config.get("patch_size", [1, 2, 2]))
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)

    def set_scheduler(self, scheduler):
        """Set reference to scheduler."""
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        """
        Run post-inference processing.

        Args:
            x: Transformer output tensor
            pre_infer_out: Output from pre-inference containing grid sizes

        Returns:
            List of unpatchified tensors as float
        """
        x = self.unpatchify(x, pre_infer_out.grid_sizes.tuple)

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        """
        Convert patch tokens back to spatial representation.

        Args:
            x: Patch tokens [N, hidden_size] where N = T' * H' * W'
            grid_sizes: Tuple of (T', H', W') grid dimensions

        Returns:
            List containing unpatchified tensor [C, T, H, W]
        """
        c = self.out_channels
        # x: [N, hidden_size] -> [T', H', W', p_t, p_h, p_w, C]
        x = x[: math.prod(grid_sizes)].view(*grid_sizes, *self.patch_size, c)
        # Rearrange to [C, T'*p_t, H'*p_h, W'*p_w]
        x = torch.einsum("fhwpqrc->cfphqwr", x)
        x = x.reshape(c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return [x]

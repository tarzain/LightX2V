from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    CONV3D_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class LTX2PreWeights(WeightModule):
    """
    Pre-processing weights for LTX-2 model.
    Includes patch embedding, time embedding, and text projection.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 2048)
        patch_size = config.get("patch_size", [1, 2, 2])
        # Some configs store patch_size as a scalar (e.g. `2`) meaning spatial patching of (1, 2, 2).
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        elif isinstance(patch_size, (list, tuple)):
            if len(patch_size) == 2:
                patch_size = (1, patch_size[0], patch_size[1])
            elif len(patch_size) == 1:
                patch_size = (1, patch_size[0], patch_size[0])
        self.patch_size = tuple(patch_size)

        # Patch embedding (3D convolution)
        self.add_module(
            "patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"](
                # Official LTX-2 checkpoints name this `patchify_proj.*` (under `diffusion_model.`).
                "patchify_proj.weight",
                "patchify_proj.bias",
                stride=self.patch_size
            ),
        )

        # Time embedding MLP
        self.add_module(
            "time_embedding_0",
            # Official LTX-2 uses `adaln_single.emb.timestep_embedder.linear_1.*`
            MM_WEIGHT_REGISTER["Default"]("adaln_single.emb.timestep_embedder.linear_1.weight", "adaln_single.emb.timestep_embedder.linear_1.bias"),
        )
        self.add_module(
            "time_embedding_1",
            MM_WEIGHT_REGISTER["Default"]("adaln_single.emb.timestep_embedder.linear_2.weight", "adaln_single.emb.timestep_embedder.linear_2.bias"),
        )

        # Text projection
        self.add_module(
            "text_embedding",
            # Official LTX-2 uses `text_embedding_projection.aggregate_embed.weight` (no bias tensor).
            MM_WEIGHT_REGISTER["Default"]("text_embedding_projection.aggregate_embed.weight", None),
        )

        # NOTE: The official checkpoint’s `caption_projection` is a multi-layer module (`caption_projection.linear_1/2.*`)
        # and LightX2V’s current LTX2 inference path doesn’t consume `caption_proj` here, so we intentionally do not
        # load a mismatched projection to avoid hard KeyErrors during weight load.

        # Image conditioning for i2v task
        if config.get("task") == "i2v":
            self.add_module(
                "img_proj",
                MM_WEIGHT_REGISTER["Default"]("img_embed.proj.weight", "img_embed.proj.bias"),
            )

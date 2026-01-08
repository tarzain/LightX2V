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
        self.patch_size = tuple(config.get("patch_size", [1, 2, 2]))

        # Patch embedding (3D convolution)
        self.add_module(
            "patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"](
                "patch_embed.proj.weight",
                "patch_embed.proj.bias",
                stride=self.patch_size
            ),
        )

        # Time embedding MLP
        self.add_module(
            "time_embedding_0",
            MM_WEIGHT_REGISTER["Default"]("time_embed.mlp.0.weight", "time_embed.mlp.0.bias"),
        )
        self.add_module(
            "time_embedding_1",
            MM_WEIGHT_REGISTER["Default"]("time_embed.mlp.2.weight", "time_embed.mlp.2.bias"),
        )

        # Text projection
        self.add_module(
            "text_embedding",
            MM_WEIGHT_REGISTER["Default"]("context_embedder.weight", "context_embedder.bias"),
        )

        # Caption channels projection (from text encoder dim to hidden size)
        if config.get("caption_channels", 4096) != config.get("hidden_size", 2048):
            self.add_module(
                "caption_proj",
                MM_WEIGHT_REGISTER["Default"]("caption_projection.weight", "caption_projection.bias"),
            )

        # Image conditioning for i2v task
        if config.get("task") == "i2v":
            self.add_module(
                "img_proj",
                MM_WEIGHT_REGISTER["Default"]("img_embed.proj.weight", "img_embed.proj.bias"),
            )

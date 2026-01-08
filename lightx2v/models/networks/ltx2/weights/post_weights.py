from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class LTX2PostWeights(WeightModule):
    """
    Post-processing weights for LTX-2 model.
    Includes final normalization and output projection.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Note: Most post weights are in transformer_weights
        # This is kept for consistency with other models

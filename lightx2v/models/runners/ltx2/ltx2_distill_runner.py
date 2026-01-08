from loguru import logger

from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner
from lightx2v.models.schedulers.ltx2.step_distill.scheduler import LTX2StepDistillScheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("ltx2_distill")
class LTX2DistillRunner(LTX2Runner):
    """
    Runner for LTX-2 19B 8-step distilled model.

    The distilled model achieves fast inference with 8 predefined denoising steps
    without requiring classifier-free guidance.

    Key features:
    - 8-step inference (vs ~50 steps for base model)
    - No CFG required
    - Maintains high visual quality
    - Supports both t2v and i2v tasks

    Default denoising schedule: [1000, 875, 750, 625, 500, 375, 250, 125]
    """

    def __init__(self, config):
        # Set distill-specific defaults
        config.setdefault("infer_steps", 8)
        config.setdefault("enable_cfg", False)  # Distilled model doesn't need CFG
        config.setdefault("sample_shift", 3.0)

        # Default 8-step denoising schedule
        config.setdefault("denoising_step_list", [1000, 875, 750, 625, 500, 375, 250, 125])

        super().__init__(config)

        logger.info(f"LTX-2 Distill Runner initialized with {config['infer_steps']} steps")
        logger.info(f"Denoising schedule: {config['denoising_step_list']}")

    def init_scheduler(self):
        """Initialize the step-distill scheduler for LTX-2."""
        if self.config.get("feature_caching", "NoCaching") == "NoCaching":
            self.scheduler = LTX2StepDistillScheduler(self.config)
        else:
            raise NotImplementedError(
                f"Feature caching type '{self.config.get('feature_caching')}' "
                f"not yet supported for LTX-2 distilled model"
            )

    def load_transformer(self):
        """Load the LTX-2 distilled transformer model."""
        # The distilled model uses the same architecture, just different weights
        model = LTX2Model(
            self.config["model_path"],
            self.config,
            self.init_device,
            model_type="ltx2_distill"
        )
        return model

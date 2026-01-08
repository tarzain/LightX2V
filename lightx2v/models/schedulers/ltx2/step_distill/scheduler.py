import torch

from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class LTX2StepDistillScheduler(LTX2Scheduler):
    """
    Step distillation scheduler for LTX-2 19B 8-step distilled model.

    The distilled model uses predefined sigma values for fast inference
    without requiring classifier-free guidance.
    """

    def __init__(self, config):
        super().__init__(config)

        # 8-step distillation uses predefined denoising steps
        self.denoising_step_list = config.get("denoising_step_list", [1000, 875, 750, 625, 500, 375, 250, 125])
        self.infer_steps = len(self.denoising_step_list)

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

        # Distilled models typically don't use CFG
        self.enable_cfg = config.get("enable_cfg", False)

    def set_timesteps(self, num_inference_steps, device, shift):
        """
        Set timesteps using predefined denoising steps for distilled model.
        The distilled model has predetermined optimal sigma values.
        """
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]

        # Apply time shift for flow matching
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        # Use predefined denoising step indices
        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def step_post(self):
        """
        Post-step processing for distilled model.
        Uses direct flow prediction update.
        """
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()

        # Update latents using flow matching formula
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred

        # Add noise for next step if not final step
        if self.step_index < self.infer_steps - 1:
            sigma_n = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = noisy_image_or_video + flow_pred * sigma_n

        self.latents = noisy_image_or_video.to(self.latents.dtype)

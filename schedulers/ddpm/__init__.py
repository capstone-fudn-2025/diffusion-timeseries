import torch
from schedulers.base import BaseScheduler
from utils.config import GlobalConfig


class DDPMScheduler(BaseScheduler):
    '''
    Denoising Diffusion Probabilistic Model (DDPM) for denoising and diffusion process
    '''

    def __init__(self, config):
        # Get timesteps and device and initialize the scheduler
        super(DDPMScheduler, self).__init__(config)
        self.beta_1 = self.config.scheduler_config.get('beta_1', 1e-4)
        self.beta_2 = self.config.scheduler_config.get('beta_2', 0.02)

        # Initialize beta values
        self.b_t = (self.beta_2 - self.beta_1) * torch.linspace(0, 1,
                                                                self.config.timesteps + 1, device=self.config.device) + self.beta_1

        # Initialize alpha values
        self.a_t = 1 - self.b_t

        # Initialize cumulative alpha values
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()

        # Set the first value of ab_t from 0.9999 to 1
        self.ab_t[0] = 1

    def denoise_diffusion(self, sample, timestep, pred_noise, add_noise):
        if add_noise is None:
            add_noise = torch.randn_like(sample).to(self.config.device)

        # Define noise to add, from high to low
        _noise_to_add = self.b_t[timestep].sqrt() * add_noise

        # Define noise to remove, from low to high
        _noise_to_remove = (
            (1 - self.a_t[timestep]) / (1 - self.ab_t[timestep]).sqrt()) * pred_noise

        # Denoise and diffuse the sample
        return (sample - _noise_to_remove) / self.a_t[timestep].sqrt() + _noise_to_add

    def training_diffusion(self, sample, timestep, noise):
        return self.ab_t[timestep].sqrt() * sample + (1 - self.ab_t[timestep]) * noise

    @torch.no_grad()
    def forward(self, backbone, mask):
        # Check device of backbone model
        backbone.to(self.config.device)

        # Initialize sample
        sample = torch.rand_like(mask).to(self.config.device)

        for i in self.iterate_timesteps():
            # Mask the sample
            mask_sample = torch.where(
                mask == 0, self.config.data_simulate_replacement_value or 0, sample)

            # Define timestep tensor
            # Shape: (n, 1, 1)
            timestep = torch.tensor(
                [i / self.config.timesteps])[:, None, None].to(self.config.device)

            # Generate noise. Only add if not the last timestep
            add_noise = torch.randn_like(sample) if i > 1 else 0

            # Predict noise
            pred_noise = backbone(sample, timestep)

            # Denoise and diffuse
            sample = self.denoise_diffusion(
                mask_sample, i, pred_noise, add_noise)

            # Save trace samples
            if self.is_saving_trace_sample(i):
                self.trace_samples.append(sample.detach().cpu().numpy())

        return sample

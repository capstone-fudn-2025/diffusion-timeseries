import torch
from schedulers.base import BaseScheduler


class DDPMScheduler(BaseScheduler):
    '''
    Denoising Diffusion Probabilistic Model (DDPM) for denoising and diffusion process
    '''

    def __init__(
        self,
        mask_value: float = 0.0,
        beta_1: float = 1e-4,
        beta_2: float = 0.02,
        **kwargs
    ):
        # Get timesteps and device and initialize the scheduler
        timesteps = kwargs.get('timesteps', 100)
        device = kwargs.get('device', None)
        trace_interval = kwargs.get('trace_interval', 20)
        super(DDPMScheduler, self).__init__(
            timesteps=timesteps, device=device, trace_interval=trace_interval)

        # Define mask value for masking protected regions
        self.mask_value = mask_value

        # Initialize beta values
        self.b_t = (beta_2 - beta_1) * torch.linspace(0, 1,
                                                      timesteps + 1, device=self.device) + beta_1

        # Initialize alpha values
        self.a_t = 1 - self.b_t

        # Initialize cumulative alpha values
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()

        # Set the first value of ab_t from 0.9999 to 1
        self.ab_t[0] = 1

    def denoise_diffusion(self, sample, timestep, pred_noise, add_noise):
        if add_noise is None:
            add_noise = torch.randn_like(sample).to(self.device)

        # Define noise to add, from high to low
        _noise_to_add = self.b_t[timestep].sqrt() * add_noise

        # Define noise to remove, from low to high
        _noise_to_remove = (
            (1 - self.a_t[timestep]) / (1 - self.ab_t[timestep]).sqrt()) * pred_noise

        # Denoise and diffuse the sample
        return (sample - _noise_to_remove) / self.a_t[timestep].sqrt() + _noise_to_add

    @torch.no_grad()
    def forward(self, backbone, mask):
        # Check device of backbone model
        backbone.to(self.device)

        # Initialize sample and mask sample
        sample = torch.rand_like(mask).to(self.device)
        mask_sample = torch.where(
            mask == 0, self.mask_value, sample).to(self.device)

        for i in self.iterate_timesteps():
            # Define timestep tensor
            # Shape: (n, 1, 1)
            timestep = torch.tensor(
                [i / self.timesteps])[:, None, None].to(self.device)

            # Generate noise. Only add if not the last timestep
            add_noise = torch.randn_like(sample) if i > 1 else 0

            # Predict noise
            pred_noise = backbone(sample, timestep)

            # Denoise and diffuse
            mask_sample = self.denoise_diffusion(
                mask_sample, i, pred_noise, add_noise)

            # Save trace samples
            if self.is_saving_trace_sample(i):
                self.trace_samples.append(mask_sample.detach().cpu().numpy())

        return mask_sample

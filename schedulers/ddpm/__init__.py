import torch
from schedulers.base import BaseScheduler


class DDPMScheduler(BaseScheduler):
    '''
    Denoising Diffusion Probabilistic Model (DDPM) for denoising and diffusion process
    '''

    def __init__(
        self,
        timesteps,
        beta_1=1e-4,
        beta_2=0.02,
        device=None,
    ):
        super().__init__(timesteps, device)

        self.b_t = (beta_2 - beta_1) * torch.linspace(0, 1,
                                                      timesteps + 1, device=device) + beta_1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    def denoise_diffusion(self, sample, timestep, pred_noise, add_noise):
        _noise = self.b_t.sqrt()[timestep] * add_noise
        _mean = (sample - pred_noise * ((1 - self.a_t[timestep]) / (
            1 - self.ab_t[timestep]).sqrt())) / self.a_t[timestep].sqrt()
        return _mean + _noise

    @torch.inference_mode()
    def forward(self, backbone):
        sample = torch.randn(1, 1, 100).to(self.device)

        for i in range(self.timesteps, 0, -1):
            # Reshape time tensor
            timestep = torch.tensor(
                [i / self.timesteps])[:, None, None, None].to(self.device)

            # Generate noise
            add_noise = torch.randn_like(sample) if i > 1 else 0

            # Predict noise
            pred_noise = backbone(sample, timestep)

            # Denoise and diffuse
            sample = self.denoise_diffusion(sample, i, pred_noise, add_noise)

        return sample

import torch
from torch import nn
from abc import abstractmethod
from backbones.base import TripleB


class BaseScheduler(nn.Module):
    '''
    Base Scheduler for duffusion and denoising process
    '''

    @abstractmethod
    def __init__(
        self,
        timesteps: int,
        device: str = None
    ):
        super(BaseScheduler, self).__init__()
        self.timesteps = timesteps
        self.device = device

        # Initialize device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def denoise_diffusion(self, sample: torch.Tensor, timestep: int,
                          pred_noise: torch.Tensor, add_noise: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, backbone: TripleB): ...

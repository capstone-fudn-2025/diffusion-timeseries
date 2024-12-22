from abc import abstractmethod
import numpy.typing as npt
import torch
from torch import nn
from tqdm import tqdm
from backbones.base import TripleB
from utils.config import GlobalConfig


class BaseScheduler(nn.Module):
    '''
    Base Scheduler for diffusion and denoising process

    Args:
        timesteps (int): Number of timesteps
        device (str): Device to use (default: None)

    Inputs:
        backbone (TripleB): Backbone model for noise prediction
        mask (torch.Tensor): Mask for protected regions. Include 0 for protected regions and 1 for unprotected regions
    '''

    def __init__(self, config: GlobalConfig):
        self.config = config
        super(BaseScheduler, self).__init__()

        # Tracing variables
        self.trace_samples: list[npt.ArrayLike] = []

    def iterate_timesteps(self):
        return tqdm(range(self.config.timesteps, 0, -1), desc=f'{self.__class__.__name__}')

    def is_saving_trace_sample(self, timestep: int) -> bool:
        '''
        Check if the current timestep is saving trace sample
        - timestep (int): Current timestep
        '''
        return timestep % self.config.trace_interval == 0 or timestep == 1

    def visualize_trace_samples(self, clear: bool = True):
        '''
        Visualize trace samples
        - clear (bool): Clear the trace samples after visualization (default: True)
        '''
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(self.trace_samples),
                               figsize=(5*len(self.trace_samples), 3))
        for i, sample in enumerate(self.trace_samples):
            ax[i].plot(sample.flatten())
            ax[i].axis('off')
            ax[i].set_title(f'Timestep {i * self.config.trace_interval}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        if clear:
            self.trace_samples = []

    @abstractmethod
    def denoise_diffusion(
        self,
        sample: torch.Tensor,
        timestep: int,
        pred_noise: torch.Tensor,
        add_noise: torch.Tensor = None
    ) -> torch.Tensor:
        '''
        Denoise and diffuse the sample
        - sample (torch.Tensor): Input sample
        - timestep (int): Current timestep
        - pred_noise (torch.Tensor): Predicted noise
        - add_noise (torch.Tensor): Additional noise
        '''
        raise NotImplementedError(
            "Method 'denoise_diffusion' must be implemented")

    @abstractmethod
    def training_diffusion(
        self,
        sample: torch.Tensor,
        timestep: int,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Add noise to the sample and train the diffusion process
        - sample (torch.Tensor): Input sample
        - timestep (int): Current timestep
        - noise (torch.Tensor): Noise tensor
        '''
        raise NotImplementedError(
            "Method 'training_diffusion' must be implemented")

    @abstractmethod
    def forward(self, backbone: TripleB, mask: torch.Tensor):
        '''
        Forward pass
        - backbone (TripleB): Backbone model for noise prediction
        - mask (torch.Tensor): Mask for protected regions. Include 0 for protected regions and 1 for unprotected regions
        '''
        raise NotImplementedError("Method 'forward' must be implemented")

import torch
from torch import nn
from abc import abstractmethod
from utils.config import GlobalConfig


class TripleB(nn.Module):
    '''
    Base Back Bone class for noise prediction models
    '''

    def __init__(self, config: GlobalConfig):
        self.config = config
        super(TripleB, self).__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor = None
    ) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Noise input tensor. Shape (N, C, L)
            t (torch.Tensor): Timestep context feature. Shape (N, Ct)
            c (torch.Tensor): Context feature. Shape (N, Cn)

        Returns:
            torch.Tensor: Output tensor. Shape (N, C, L)
        '''
        raise NotImplementedError('Forward method must be implemented')

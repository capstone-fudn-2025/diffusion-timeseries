import torch
from torch import nn
from abc import abstractmethod


class TripleB(nn.Module):
    '''
    Base Back Bone class for noise prediction models
    '''

    def __init__(self, *args, **kwargs):
        super(TripleB, self).__init__(*args, **kwargs)

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

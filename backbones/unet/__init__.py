import torch
from backbones.base import TripleB
from .components import *


class Unet(TripleB):
    '''
    Unet Backbone for noise prediction.

    Args:
        in_channels (int): number of input channels
        n_features (int): number of intermediate feature maps
        n_context_features (int): number of context features

    Shape:
        Input:
            - x (n, in_channels, l): Input tensor
            - t (n, 1, 1): Timestep tensor
            - c (n, n_context_features): Context tensor
        Output: (n, in_channels, l)

    - n: batch size
    - in_channels: number of input channels
    - l: length of the input sequence
    - n_features: number of intermediate feature maps
    - n_context_features: number of context features
    '''

    def __init__(
        self,
        in_channels: int,
        n_features: int = 256,
        n_context_features: int = 10
    ):
        super(Unet, self).__init__()

        # Set the number of input channels, intermediate feature maps, and context features
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_context_features = n_context_features

        # Initialize the initial convolutional layer
        # Shape: (n, in_channels, l) -> (n, n_features, l)
        self.init_conv = ResidualConvBlock(
            in_channels, n_features, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        # Shape: (n, n_features, l) -> (n, n_features, l / 2)
        self.down1 = UnetDown(n_features, n_features)
        # Shape: (n, n_features, l / 2) -> (n, 2 * n_features, l / 4)
        self.down2 = UnetDown(n_features, 2 * n_features)

        # Convert the feature maps to a vector and apply an activation
        to_vec_layers = [
            # Shape: (n, 2 * n_features, l / 4) -> (n, 2 * n_features, l / 16)
            nn.AvgPool1d(4),

            # Shape: (n, 2 * n_features, l / 16) -> (n, 2 * n_features, l / 16)
            nn.GELU()
        ]
        self.to_vec = nn.Sequential(*to_vec_layers)

        # Embed the timestep and context labels with a one-layer fully connected neural network
        # Shape: (n, 1, 1) -> (n, 2 * n_features)
        self.timeembed1 = EmbedFC(1, 2 * n_features)

        # Shape: (n, 1, 1) -> (n, n_features)
        self.timeembed2 = EmbedFC(1, n_features)

        # Shape: (n, n_context_features) -> (n, 2 * n_features)
        self.contextembed1 = EmbedFC(n_context_features, 2 * n_features)

        # Shape: (n, n_context_features) -> (n, n_features)
        self.contextembed2 = EmbedFC(n_context_features, n_features)

        # Initialize the up-sampling path of the U-Net with three levels
        up_layers = [
            # Shape: (n, 2 * n_features, l / 16) -> (n, 2 * n_features, l / 4)
            nn.ConvTranspose1d(2 * n_features, 2 * n_features,
                               kernel_size=4, stride=4),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU()
        ]
        self.up0 = nn.Sequential(*up_layers)

        # Shape: (n, 4 * n_features, l / 4) -> (n, n_features, l / 2).
        # The input is the concatenation of the upsampled feature maps and skip connections from the down-sampling path
        self.up1 = UnetUp(4 * n_features, n_features)

        # Shape: (n, 2 * n_features, l / 2) -> (n, n_features, l).
        # The input is the concatenation of the upsampled feature maps and skip connections from the down-sampling path
        self.up2 = UnetUp(2 * n_features, n_features)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        out_layers = [
            # Shape: (n, 2 * n_features, l) -> (n, n_features, l)
            nn.Conv1d(2 * n_features, n_features,
                      kernel_size=3, stride=1, padding=1),

            nn.GroupNorm(8, n_features),

            nn.ReLU(),

            # Shape: (n, n_features, l) -> (n, in_channels, l)
            nn.Conv1d(n_features, in_channels,
                      kernel_size=3, stride=1, padding=1),
        ]
        self.out = nn.Sequential(*out_layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor = None
    ):
        '''
        x: (n, in_channels, l) - Input tensor
        t: (n, 1, 1) - Timestep tensor
        c: (n, n_context_features) - Context tensor
        '''

        # Shape: (n, in_channels, l) -> (n, n_features, l)
        x = self.init_conv(x)

        # Shape: (n, n_features, l) -> (n, n_features, l / 2)
        down1 = self.down1(x)

        # Shape: (n, n_features, l / 2) -> (n, 2 * n_features, l / 4)
        down2 = self.down2(down1)

        # Shape: (n, 2 * n_features, l / 4) -> (n, 2 * n_features, l / 16)
        hiddenvec = self.to_vec(down2)

        # Create zero tensors for the context if not provided
        if c is None:
            c = torch.zeros(x.shape[0], self.n_context_features).to(x.device)

        # Embed the timestep and context labels for the first up-sampling path
        # Shape: (n, n_context_features) -> (n, 2 * n_features) -> (n, 2 * n_features, 1)
        cemb1 = self.contextembed1(c).view(-1, self.n_features * 2, 1)

        # Shape: (n, 1, 1) -> (n, 2 * n_features) -> (n, 2 * n_features, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_features * 2, 1)

        # Embed the timestep and context labels for the second up-sampling path
        # Shape: (n, n_context_features) -> (n, n_features) -> (n, n_features, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_features, 1)

        # Shape: (n, 1, 1) -> (n, n_features) -> (n, n_features, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_features, 1)

        # Shape: (n, 2 * n_features, l / 16) -> (n, 2 * n_features, l / 4)
        up1 = self.up0(hiddenvec)

        # Shape: (n, 2 * n_features, l / 4) -> (n, n_features, l / 2)
        up2 = self.up1(cemb1 * up1 + temb1, down2)

        # Shape: (n, n_features, l / 2) -> (n, n_features, l)
        up3 = self.up2(cemb2 * up2 + temb2, down1)

        # Shape: (n, 2 * n_features, l) -> (n, in_channels, l)
        out = self.out(
            torch.cat((up3, x), 1)
        )
        return out

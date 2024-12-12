import torch
from backbones.base import TripleB
from .components import *


class Unet(TripleB):
    '''
    Unet Backbone for noise prediction.

    Args:
        in_channels (int): number of input channels
        n_feat (int): number of intermediate feature maps
        n_cfeat (int): number of context features

    Assume that we have an time series input (n, c, l) = (1, 1, 100)
    - n: batch size
    - c: number of channels (1 - univariate)
    - l: length of the time series (100)
    '''

    def __init__(self, in_channels: int, n_feat=256, n_cfeat=10):
        super(Unet, self).__init__()
        # NOTE: Assume we have input tensor of shape (1, 1, 100); in_channels=1, n_feat=256, n_cfeat=10

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat

        # Initialize the initial convolutional layer
        # NOTE: (1, 1, 100) -> (1, 256, 100)
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        # NOTE: (1, 256, 100) -> (1, 256, 50)
        self.down1 = UnetDown(n_feat, n_feat)
        # NOTE: (1, 256, 50) -> (1, 512, 25)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # NOTE: (1, 512, 25) -> (1, 512, 6)
        self.to_vec = nn.Sequential(**[
            # NOTE: (1, 512, 25) -> (1, 512, 6)
            nn.AvgPool1d(4),

            # NOTE: (1, 512, 6) -> (1, 512, 6)
            nn.GELU()
        ])

        # Embed the timestep and context labels with a one-layer fully connected neural network
        # NOTE: (1, 512) -> (512, 512)
        self.timeembed1 = EmbedFC(1, 2*n_feat)

        # NOTE: (1, 256) -> (256, 256)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        # NOTE: (1, 10) -> (10, 512)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)

        # NOTE: (1, 10) -> (10, 256)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            # NOTE: (1, 512, 6) -> (1, 512, 25)
            nn.ConvTranspose1d(in_channels=2 * n_feat, out_channels=2 * n_feat,
                               kernel_size=5, stride=4),

            # NOTE: (1, 512, 25) -> (1, 512, 25)
            nn.GroupNorm(8, 2 * n_feat),

            # NOTE: (1, 512, 25) -> (1, 512, 25)
            nn.ReLU(),
        )

        # NOTE: (1, 512, 25) -> (1, 256, 50)
        self.up1 = UnetUp(4 * n_feat, n_feat)

        # NOTE: (1, 256, 50) -> (1, 256, 100)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            # NOTE: (1, 512, 100) -> (1, 256, 100)
            nn.Conv1d(2 * n_feat, n_feat, kernel_size=3, stride=1, padding=1),

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.GroupNorm(8, n_feat),

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.ReLU(),

            # NOTE: (1, 256, 100) -> (1, 1, 100)
            nn.Conv1d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        # NOTE: Assume we have input tensor x: (1, 1, 100), t: (1, 10), c: None

        # pass the input image through the initial convolutional layer
        # NOTE: (1, 1, 100) -> (1, 256, 100)
        x = self.init_conv(x)

        # pass the result through the down-sampling path
        # NOTE: (1, 256, 100) -> (1, 256, 50)
        down1 = self.down1(x)

        # NOTE: (1, 256, 50) -> (1, 512, 25)
        down2 = self.down2(down1)

        # convert the feature maps to a vector and apply an activation
        # NOTE: (1, 512, 25) -> (1, 512, 6)
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            # c = (1, 10)
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        # NOTE: (1, 10) -> (1, 512) -> (1, 512, 1)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1)
        # NOTE: (1, 1) -> (1, 512) -> (1, 512, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1)
        # NOTE: (1, 10) -> (1, 256) -> (1, 256, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1)
        # NOTE: (1, 1) -> (1, 256) -> (1, 256, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1)

        # NOTE: (1, 512, 6) -> (1, 512, 25)
        up1 = self.up0(hiddenvec)

        # NOTE: (1, 512, 1) * (1, 512, 25) + (1, 512, 1), (1, 512, 25) -> (1, 256, 50)
        up2 = self.up1(cemb1*up1 + temb1, down2)

        # NOTE: (1, 256, 1) * (1, 256, 50) + (1, 256, 1), (1, 256, 1) -> (1, 256, 100)
        up3 = self.up2(cemb2*up2 + temb2, down1)

        # NOTE: (1, 256, 100), (1, 256, 100) -> (1, 1, 100)
        out = self.out(torch.cat((up3, x), 1))
        return out

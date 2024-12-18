import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    '''
    This class defines a residual convolutional block, which consists of two convolutional layers with GELU activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        is_res (bool): Flag for whether or not to use residual connection

    Shape:
        Input:
            - x (n, in_channels, l): Input tensor
        Output: (n, out_channels, l)

    - n: batch size
    - in_channels: number of input channels
    - out_channels: number of output channels
    - l: length of the time series
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False
    ) -> None:
        super(ResidualConvBlock, self).__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # Normalize value for output
        self.norm = 1.414

        # First convolutional layer.
        # Shape: (n, in_channels, l) -> (n, out_channels, l)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        # Second convolutional layer.
        # Shape: (n, out_channels, l) -> (n, out_channels, l)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (n, in_channels, l) - Input tensor
        '''

        # If using residual connection
        if self.is_res:
            # Shape: (n, in_channels, l) -> (n, out_channels, l)
            x1 = self.conv1(x)

            # Shape: (n, out_channels, l) -> (n, out_channels, l)
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                # Shape: (n, out_channels, l) + (n, out_channels, l) -> (n, out_channels, l)
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection.
                shortcut = nn.Conv1d(
                    x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0
                ).to(x.device)

                # Shape: (n, in_channels, l) -> (n, out_channels, l) + (n, out_channels, l) -> (n, out_channels, l)
                out = shortcut(x) + x2

            # Shape: (n, out_channels, l) -> (n, out_channels, l)
            return out / self.norm

        # If not using residual connection, return output of second convolutional layer
        else:
            # Shape: (n, in_channels, l) -> (n, out_channels, l)
            x1 = self.conv1(x)

            # Shape: (n, out_channels, l) -> (n, out_channels, l)
            x2 = self.conv2(x1)
            return x2


class UnetUp(nn.Module):
    '''
    This class defines an upsampling block for the U-Net architecture, which consists of a ConvTranspose2d layer for
    upsampling, followed by two ResidualConvBlock layers.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Shape:
        Input:
            - x (n, in_channels, l): Input tensor
            - skip (n, in_channels, l): Skip connection tensor
        Output: (n, out_channels, l * 2)

    - n: batch size
    - in_channels: number of input channels
    - out_channels: number of output channels
    - l: length of the time series
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(UnetUp, self).__init__()

        # Create a list of layers for the upsampling block
        layers = [
            # Shape: (n, in_channels, l) -> (n, out_channels, l * 2)
            nn.ConvTranspose1d(in_channels, out_channels,
                               kernel_size=2, stride=2),

            # Shape: (n, out_channels, l * 2) -> (n, out_channels, l * 2)
            ResidualConvBlock(out_channels, out_channels),

            # Shape: (n, out_channels, l * 2) -> (n, out_channels, l * 2)
            ResidualConvBlock(out_channels, out_channels),
        ]

        # Use the layers to create a sequential model.
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        '''
        x: (n, in_channels, l) - Input tensor
        skip: (n, in_channels, l) - Skip connection tensor
        '''

        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        # Shape: (n, in_channels, l) + (n, in_channels, l) -> (n, in_channels * 2, l)
        x = torch.cat((x, skip), 1)

        # Pass the concatenated tensor through the sequential model and return the output
        # Shape: (n, in_channels * 2, l) -> (n, out_channels, l * 2)
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    '''
    This class defines a downsampling block for the U-Net architecture, which consists of two ResidualConvBlock layers
    followed by a MaxPool2d layer for downsampling.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Shape:
        Input:
            - x (n, in_channels, l): Input tensor
        Output: (n, out_channels, l / 2)

    - n: batch size
    - in_channels: number of input channels
    - out_channels: number of output channels
    - l: length of the time series
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(UnetDown, self).__init__()

        # Create a list of layers for the downsampling block
        layers = [
            # Shape: (n, in_channels, l) -> (n, out_channels, l)
            ResidualConvBlock(in_channels, out_channels),

            # Shape: (n, out_channels, l) -> (n, out_channels, l)
            ResidualConvBlock(out_channels, out_channels),

            # Shape: (n, out_channels, l) -> (n, out_channels, l/2)
            nn.MaxPool1d(2)
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (n, in_channels, l) - Input tensor
        '''
        # Pass the input through the sequential model and return the output
        # Shape: (n, in_channels, l) -> (n, out_channels, l / 2)
        return self.model(x)


class EmbedFC(nn.Module):
    '''
    This class defines a generic one layer feed-forward neural network for embedding input data of
    dimensionality input_dim to an embedding space of dimensionality emb_dim.

    Args:
        input_dim (int): Number of input dimensions
        emb_dim (int): Number of embedding dimensions

    Shape:
        Input:
            - x (n, input_dim): Input tensor
        Output: (n, emb_dim)

    - n: batch size
    - input_dim: number of input dimensions
    - emb_dim: number of embedding dimensions
    '''

    def __init__(
        self,
        input_dim: int,
        emb_dim: int
    ):
        super(EmbedFC, self).__init__()

        # Input dimensionality
        self.input_dim = input_dim

        # Define the layers for the network.
        layers = [
            # Shape: (n, input_dim) -> (n, emb_dim)
            nn.Linear(input_dim, emb_dim),

            nn.GELU(),

            # Shape: (n, emb_dim) -> (n, emb_dim)
            nn.Linear(emb_dim, emb_dim),
        ]

        # Create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (n, input_dim) - Input tensor
        '''

        # Flatten the input tensor
        # Shape: (n, input_dim) -> (n, input_dim)
        x = x.view(-1, self.input_dim)

        # Apply the model layers to the flattened tensor
        # Shape: (n, input_dim) -> (n, emb_dim)
        return self.model(x)

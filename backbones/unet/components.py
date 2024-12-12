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
        Input: (n, c_in, l)
        Output: (n, c_out, l)

    - n: batch size
    - c_in: number of input channels
    - c_out: number of output channels
    - l: length of the time series
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False
    ) -> None:
        super().__init__()
        # NOTE: Assume we have input tensor of shape (1, 1, 100); in_channels=1, out_channels=256

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels  # NOTE: -> False

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # Normalize threshold
        self.norm = 1.414

        # First convolutional layer.
        self.conv1 = nn.Sequential(
            # NOTE: (1, 1, 100) -> (1, 256, 100)
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.BatchNorm1d(out_channels),

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.GELU(),
        )

        # Second convolutional layer. Example, out_channels=1
        self.conv2 = nn.Sequential(
            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.BatchNorm1d(out_channels),

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: Assume we have input tensor x: (1, 1, 100)

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)  # NOTE: (1, 1, 100) -> (1, 256, 100)

            # Apply second convolutional layer
            x2 = self.conv2(x1)  # NOTE: (1, 256, 100) -> (1, 256, 100)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                # NOTE: (1, 1, 100) + (1, 1, 100) -> (1, 1, 100)
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection. Example, in_channels=1, out_channels=2
                shortcut = nn.Conv1d(
                    x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0
                ).to(x.device)  # NOTE: (1, 1, 100) -> (1, 256, 100)

                # NOTE: (1, 256, 100) + (1, 256, 100) -> (1, 256, 100)
                out = shortcut(x) + x2

            # Normalize output tensor
            return out / self.norm  # NOTE: (1, 256, 100)

        # If not using residual connection, return output of second convolutional layer
        else:
            # NOTE: (1, 1, 100) -> (1, 256, 100)
            x1 = self.conv1(x)

            # NOTE: (1, 256, 100) -> (1, 256, 100)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


class UnetUp(nn.Module):
    '''
    This class defines an upsampling block for the U-Net architecture, which consists of a ConvTranspose2d layer for
    upsampling, followed by two ResidualConvBlock layers.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Shape:
        Input: (n, c_in, l)
        Output: (n, c_out, l * 2)
    '''

    def __init__(self, in_channels: int, out_channels: int):
        super(UnetUp, self).__init__()
        # NOTE: Assume we have in_channels=1024, out_channels=256

        # Create a list of layers for the upsampling block
        layers = [
            # NOTE: (1, 1024, 25) -> (1, 256, 50)
            nn.ConvTranspose1d(in_channels, out_channels,
                               kernel_size=2, stride=2),

            # NOTE: (1, 256, 50) -> (1, 256, 50)
            ResidualConvBlock(out_channels, out_channels),

            # NOTE: (1, 256, 50) -> (1, 256, 50)
            ResidualConvBlock(out_channels, out_channels),
        ]

        # Use the layers to create a sequential model.
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # NOTE: Assume we have input tensor x: (1, 512, 25) and skip tensor: (1, 512, 25)

        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        # NOTE: (1, 512, 25) + (1, 512, 25) -> (1, 1024, 25)
        x = torch.cat((x, skip), 1)

        # Pass the concatenated tensor through the sequential model and return the output
        # NOTE: (1, 1024, 25) -> (1, 256, 50)
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
        Input: (n, c_in, l)
        Output: (n, c_out, l/2)

    - n: batch size
    - c_in: number of input channels
    - c_out: number of output channels
    - l: length of the time series
    '''

    def __init__(self, in_channels: int, out_channels: int):
        super(UnetDown, self).__init__()
        # NOTE: Assume we have in_channels=256, out_channels=512

        # Create a list of layers for the downsampling block
        layers = [
            # NOTE: (1, 256, 100) -> (1, 512, 100)
            ResidualConvBlock(in_channels, out_channels),

            # NOTE: (1, 512, 100) -> (1, 512, 100)
            ResidualConvBlock(out_channels, out_channels),

            # NOTE: (1, 512, 100) -> (1, 512, 50)
            nn.MaxPool1d(2)
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)


class EmbedFC(nn.Module):
    '''
    This class defines a generic one layer feed-forward neural network for embedding input data of
    dimensionality input_dim to an embedding space of dimensionality emb_dim.

    Args:
        input_dim (int): Number of input dimensions
        emb_dim (int): Number of embedding dimensions

    Shape:
        Input: (n, emb_dim)
        Output: (input_dim, emb_dim)

    - n: batch size
    - input_dim: number of input dimensions
    - emb_dim: number of embedding dimensions
    '''

    def __init__(self, input_dim: int, emb_dim: int):
        super(EmbedFC, self).__init__()
        # NOTE: Assume we have input_dim=1, emb_dim=10
        self.input_dim = input_dim

        # define the layers for the network. Example, input_dim=1, emb_dim=2
        layers = [
            # NOTE: (1, 1) -> (1, 10)
            nn.Linear(input_dim, emb_dim),

            # NOTE: (1, 10) -> (1, 10)
            nn.GELU(),

            # NOTE: (1, 10) -> (1, 10)
            nn.Linear(emb_dim, emb_dim),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Example, x = (1, 10)

        # flatten the input tensor
        # NOTE: (1, 10) -> (10, 1)
        x = x.view(-1, self.input_dim)

        # apply the model layers to the flattened tensor
        # NOTE: (10, 1) -> (10, 10)
        return self.model(x)

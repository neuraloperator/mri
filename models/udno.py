"""
U-shaped DISCO Neural Operator
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_harmonics.convolution import (
    EquidistantDiscreteContinuousConv2d as DISCO2d,
)


class UDNO(nn.Module):
    """
    U-shaped DISCO Neural Operator in PyTorch
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        in_shape: Tuple[int, int] = (320, 320),
        kernel_shape: Tuple[int, int] = (3, 4),
    ):
        """
        Parameters
        ----------
        in_chans : int
            Number of channels in the input to the U-Net model.
        out_chans : int
            Number of channels in the output to the U-Net model.
        radius_cutoff : float
            Control the effective radius of the DISCO kernel. Values are
            between 0.0 and 1.0. The radius_cutoff is represented as a proportion
            of the normalized input space, to ensure that kernels are resolution
            invaraint.
        chans : int, optional
            Number of output channels of the first DISCO layer. Default is 32.
        num_pool_layers : int, optional
            Number of down-sampling and up-sampling layers. Default is 4.
        drop_prob : float, optional
            Dropout probability. Default is 0.0.
        in_shape : Tuple[int, int]
            Shape of the input to the UDNO. This is required to dynamically
            compile DISCO kernels for resolution invariance.
        kernel_shape : Tuple[int, int], optional
            Shape of the DISCO kernel. Default is (3, 4). This corresponds to 3
            rings and 4 anisotropic basis functions. Under the hood, each DISCO
            kernel has (3 - 1) * 4 + 1 = 9 parameters, equivalent to a standard
            3x3 convolution kernel.

            Note: This is NOT kernel_size, as under the DISCO framework,
            kernels are dynamically compiled to support resolution invariance.
        """
        super().__init__()
        assert len(in_shape) == 2, "Input shape must be 2D"

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.in_shape = in_shape
        self.kernel_shape = kernel_shape

        self.down_sample_layers = nn.ModuleList(
            [
                DISCOBlock(
                    in_chans,
                    chans,
                    radius_cutoff,
                    drop_prob,
                    in_shape,
                    kernel_shape,
                )
            ]
        )
        ch = chans
        shape = (in_shape[0] // 2, in_shape[1] // 2)
        radius_cutoff = radius_cutoff * 2
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                DISCOBlock(
                    ch,
                    ch * 2,
                    radius_cutoff,
                    drop_prob,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                )
            )
            ch *= 2
            shape = (shape[0] // 2, shape[1] // 2)
            radius_cutoff *= 2

            # test commit

        self.bottleneck = DISCOBlock(
            ch,
            ch * 2,
            radius_cutoff,
            drop_prob,
            in_shape=shape,
            kernel_shape=kernel_shape,
        )

        self.up = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose.append(
                TransposeDISCOBlock(
                    ch * 2,
                    ch,
                    radius_cutoff,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                )
            )
            shape = (shape[0] * 2, shape[1] * 2)
            radius_cutoff /= 2
            self.up.append(
                DISCOBlock(
                    ch * 2,
                    ch,
                    radius_cutoff,
                    drop_prob,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                )
            )
            ch //= 2

        self.up_transpose.append(
            TransposeDISCOBlock(
                ch * 2,
                ch,
                radius_cutoff,
                in_shape=shape,
                kernel_shape=kernel_shape,
            )
        )
        shape = (shape[0] * 2, shape[1] * 2)
        radius_cutoff /= 2
        self.up.append(
            nn.Sequential(
                DISCOBlock(
                    ch * 2,
                    ch,
                    radius_cutoff,
                    drop_prob,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                ),
                nn.Conv2d(
                    ch, self.out_chans, kernel_size=1, stride=1
                ),  # 1x1 conv is always res-invariant (pixel wise channel transformation)
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.bottleneck(output)

        # apply up-sampling layers
        for transpose, disco in zip(self.up_transpose, self.up):
            downsample_layer = stack.pop()
            output = transpose(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = disco(output)

        return output


class DISCOBlock(nn.Module):
    """
    A DISCO Block that consists of two DISCO layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        drop_prob: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int] = (3, 4),
    ):
        """
        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        radius_cutoff : float
            Control the effective radius of the DISCO kernel. Values are
            between 0.0 and 1.0. The radius_cutoff is represented as a proportion
            of the normalized input space, to ensure that kernels are resolution
            invaraint.
        in_shape : Tuple[int]
            Unbatched spatial 2D shape of the input to this block.
            Rrequired to dynamically compile DISCO kernels for resolution invariance.
        kernel_shape : Tuple[int, int], optional
            Shape of the DISCO kernel. Default is (3, 4). This corresponds to 3
            rings and 4 anisotropic basis functions. Under the hood, each DISCO
            kernel has (3 - 1) * 4 + 1 = 9 parameters, equivalent to a standard
            3x3 convolution kernel.

            Note: This is NOT kernel_size, as under the DISCO framework,
            kernels are dynamically compiled to support resolution invariance.
        drop_prob : float
            Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            DISCO2d(
                in_chans,
                out_chans,
                kernel_shape=kernel_shape,
                in_shape=in_shape,
                bias=False,
                radius_cutoff=radius_cutoff,
                padding_mode="constant",
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            DISCO2d(
                out_chans,
                out_chans,
                kernel_shape=kernel_shape,
                in_shape=in_shape,
                bias=False,
                radius_cutoff=radius_cutoff,
                padding_mode="constant",
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : ndarray
            Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        ndarray
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeDISCOBlock(nn.Module):
    """
    A transpose DISCO Block that consists of an up-sampling layer followed by a
    DISCO layer, instance normalization, and LeakyReLU activation.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int] = (3, 4),
    ):
        """
        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        radius_cutoff : float
            Control the effective radius of the DISCO kernel. Values are
            between 0.0 and 1.0. The radius_cutoff is represented as a proportion
            of the normalized input space, to ensure that kernels are resolution
            invaraint.
        in_shape : Tuple[int]
            Unbatched spatial 2D shape of the input to this block.
            Rrequired to dynamically compile DISCO kernels for resolution invariance.
        kernel_shape : Tuple[int, int], optional
            Shape of the DISCO kernel. Default is (3, 4). This corresponds to 3
            rings and 4 anisotropic basis functions. Under the hood, each DISCO
            kernel has (3 - 1) * 4 + 1 = 9 parameters, equivalent to a standard
            3x3 convolution kernel.

            Note: This is NOT kernel_size, as under the DISCO framework,
            kernels are dynamically compiled to support resolution invariance
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DISCO2d(
                in_chans,
                out_chans,
                kernel_shape=kernel_shape,
                in_shape=(2 * in_shape[0], 2 * in_shape[1]),
                bias=False,
                radius_cutoff=(radius_cutoff / 2),
                padding_mode="constant",
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

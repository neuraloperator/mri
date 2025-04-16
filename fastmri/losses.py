"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Initialize the Losses class.

        Parameters
        ----------
        win_size : int, optional
            Window size for SSIM calculation.
        k1 : float, optional
            k1 parameter for SSIM calculation.
        k2 : float, optional
            k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None].to(X.device)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        # Compute means
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)

        # Compute variances
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)

        # Compute covariances
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        # Compute SSIM components
        A1, A2 = 2 * ux * uy + C1, 2 * vxy + C2
        B1, B2 = ux**2 + uy**2 + C1, vx + vy + C2
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the SSIMLoss module and move it to the GPU
    ssim_loss = SSIMLoss().to(device)

    # Create example tensors and move them to the GPU
    X = torch.randn(4, 1, 256, 256).to(device)
    Y = torch.randn(4, 1, 256, 256).to(device)
    data_range = torch.rand(4).to(device)

    # Compute the loss
    loss = ssim_loss(X, Y, data_range)
    print(loss)

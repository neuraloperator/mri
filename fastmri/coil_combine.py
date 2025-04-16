"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

import fastmri


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    The RSS is computed assuming that `dim` is the coil dimension.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor.
    dim : int, optional
        The dimension along which to apply the RSS transform (default is 0).

    Returns
    -------
    torch.Tensor
        The computed RSS value.
    """
    return torch.sqrt((data**2).sum(dim))


def mvue(spatial_pred, sens_maps, dim: int = 0) -> torch.Tensor:
    spatial_pred = torch.view_as_complex(spatial_pred)
    sens_maps = torch.view_as_complex(sens_maps)

    numerator = torch.sum(spatial_pred * torch.conj(sens_maps), dim=dim)
    denominator = torch.sqrt(torch.sum(torch.square(torch.abs(sens_maps)), dim=dim))
    res = numerator / denominator
    res = torch.abs(res)
    return res


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    The RSS is computed assuming that `dim` is the coil dimension.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor containing complex values.
    dim : int, optional
        The dimension along which to apply the RSS transform (default is 0).

    Returns
    -------
    torch.Tensor
        The computed RSS value for complex inputs.
    """
    return torch.sqrt(fastmri.complex_abs_sq(data).sum(dim))

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch
import torch.fft


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply a centered 2-dimensional Fast Fourier Transform (FFT).

    Parameters
    ----------
    data : torch.Tensor
        Complex-valued input data containing at least 3 dimensions.
        Dimensions -3 and -2 are spatial dimensions, and dimension -1 has size 2.
        All other dimensions are assumed to be batch dimensions.
    norm : str
        Normalization mode. Refer to `torch.fft.fft` for details on normalization options.

    Returns
    -------
    torch.Tensor
        The FFT of the input data.
    """

    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply a centered 2-dimensional Inverse Fast Fourier Transform (IFFT).

    Parameters
    ----------
    data : torch.Tensor
        Complex-valued input data containing at least 3 dimensions.
        Dimensions -3 and -2 are spatial dimensions, and dimension -1 has size 2.
        All other dimensions are assumed to be batch dimensions.
    norm : str
        Normalization mode. Refer to `torch.fft.ifft` for details on normalization options.

    Returns
    -------
    torch.Tensor
        The IFFT of the input data.
    """

    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Roll a PyTorch tensor along a specified dimension.

    This function is similar to `torch.roll` but operates on a single dimension.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be rolled.
    shift : int
        Amount to roll.
    dim : int
        The dimension along which to roll the tensor.

    Returns
    -------
    torch.Tensor
        A tensor with the same shape as `x`, but rolled along the specified dimension.
    """

    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Parameters
    ----------
    x : torch.Tensor
        A PyTorch tensor.
    shift : int
        Amount to roll.
    dim : int
        Which dimension to roll.

    Returns
    -------
    torch.Tensor
        Rolled version of x.
    """

    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors.

    Parameters
    ----------
    x : torch.Tensor
        A PyTorch tensor.
    dim : list of int, optional
        Which dimension to apply fftshift. If None, the shift is applied to all dimensions (default is None).

    Returns
    -------
    torch.Tensor
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors.

    Parameters
    ----------
    x : torch.Tensor
        A PyTorch tensor.
    dim : list of int, optional
        Which dimension to apply ifftshift. If None, the shift is applied to all dimensions (default is None).

    Returns
    -------
    torch.Tensor
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

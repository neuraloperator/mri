"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    Multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Parameters
    ----------
    x : torch.Tensor
        A PyTorch tensor with the last dimension of size 2.
    y : torch.Tensor
        A PyTorch tensor with the last dimension of size 2.

    Returns
    -------
    torch.Tensor
        A PyTorch tensor with the last dimension of size 2, representing
        the result of the complex multiplication.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    Applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Parameters
    ----------
    x : torch.Tensor
        A PyTorch tensor with the last dimension of size 2.

    Returns
    -------
    torch.Tensor
        A PyTorch tensor with the last dimension of size 2, representing
        the complex conjugate of the input tensor.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex-valued input tensor.

    Parameters
    ----------
    data : torch.Tensor
        A complex-valued tensor, where the size of the final dimension
        should be 2.

    Returns
    -------
    torch.Tensor
        Absolute value of the input tensor.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Parameters
    ----------
    data : torch.Tensor
        A complex-valued tensor, where the size of the final dimension
        should be 2.

    Returns
    -------
    torch.Tensor
        Squared absolute value of the input tensor.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Convert a complex PyTorch tensor to a NumPy array.

    Parameters
    ----------
    data : torch.Tensor
        Input data to be converted to a NumPy array.

    Returns
    -------
    np.ndarray
        A complex NumPy array version of the input tensor.
    """
    return torch.view_as_complex(data).numpy()

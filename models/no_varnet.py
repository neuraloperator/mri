import math
from typing import List, Literal, Optional, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri import transforms
from models.udno import UDNO


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates F (x sens_maps)

    Parameters
    ----------
    x : ndarray
        Single-channel image of shape (..., H, W, 2)
    sens_maps : ndarray
        Sensitivity maps (image space)

    Returns
    -------
    ndarray
        Result of the operation F (x sens_maps)
    """
    return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))


def sens_reduce(k: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates F^{-1}(k) * conj(sens_maps)
    where conj(sens_maps) is the element-wise applied complex conjugate

    Parameters
    ----------
    k : ndarray
        Multi-channel k-space of shape (B, C, H, W, 2)
    sens_maps : ndarray
        Sensitivity maps (image space)

    Returns
    -------
    ndarray
        Result of the operation F^{-1}(k) * conj(sens_maps)
    """
    return fastmri.complex_mul(fastmri.ifft2c(k), fastmri.complex_conj(sens_maps)).sum(
        dim=1, keepdim=True
    )


def chans_to_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Reshapes batched multi-channel samples into multiple single channel samples.

    Parameters
    ----------
    x : torch.Tensor
        x has shape (b, c, h, w, 2)

    Returns
    -------
    Tuple[torch.Tensor, int]
        tensor of shape (b * c, 1, h, w, 2), b
    """
    b, c, h, w, comp = x.shape
    return x.view(b * c, 1, h, w, comp), b


def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshapes batched independent samples into original multi-channel samples.

    Parameters
    ----------
    x : torch.Tensor
        tensor of shape (b * c, 1, h, w, 2)
    batch_size : int
        batch size

    Returns
    -------
    torch.Tensor
        original multi-channel tensor of shape (b, c, h, w, 2)
    """
    bc, _, h, w, comp = x.shape
    c = bc // batch_size
    return x.view(batch_size, c, h, w, comp)


class NormUDNO(nn.Module):
    """
    Normalized UDNO model.

    Inputs are normalized before the UDNO for numerically stable training.
    """

    def __init__(
        self,
        chans: int,
        num_pool_layers: int,
        radius_cutoff: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int],
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Initialize the VarNet model.

        Parameters
        ----------
        chans : int
            Number of output channels of the first convolution layer.
        num_pools : int
            Number of down-sampling and up-sampling layers.
        in_chans : int, optional
            Number of channels in the input to the U-Net model. Default is 2.
        out_chans : int, optional
            Number of channels in the output to the U-Net model. Default is 2.
        drop_prob : float, optional
            Dropout probability. Default is 0.0.
        """
        super().__init__()

        self.udno = UDNO(
            in_chans=in_chans,
            out_chans=out_chans,
            radius_cutoff=radius_cutoff,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            in_shape=in_shape,
            kernel_shape=kernel_shape,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def norm_new(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIXME: not working, wip
        # group norm
        b, c, h, w = x.shape
        num_groups = 2
        assert c % num_groups == 0, (
            f"Number of channels ({c}) must be divisible by number of groups ({num_groups})."
        )

        x = x.view(b, num_groups, c // num_groups * h * w)

        mean = x.mean(dim=2).view(b, num_groups, 1, 1)
        std = x.std(dim=2).view(b, num_groups, 1, 1)
        print(x.shape, mean.shape, std.shape)

        x = x.view(b, c, h, w)
        mean = (
            mean.view(b, num_groups, 1, 1)
            .repeat(1, c // num_groups, h, w)
            .view(b, c, h, w)
        )
        std = (
            std.view(b, num_groups, 1, 1)
            .repeat(1, c // num_groups, h, w)
            .view(b, c, h, w)
        )

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        chans = x.shape[1]
        if chans == 2:
            # FIXME: hard coded skip norm/pad temporarily to avoid group norm bug
            x = self.complex_to_chan_dim(x)
            x = self.udno(x)
            return self.chan_complex_to_last_dim(x)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.udno(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Learn sensitivity maps
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        radius_cutoff: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int],
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Parameters
        ----------
        chans : int
            Number of output channels of the first convolution layer.
        num_pools : int
            Number of down-sampling and up-sampling layers.
        in_chans : int, optional
            Number of channels in the input to the U-Net model. Default is 2.
        out_chans : int, optional
            Number of channels in the output to the U-Net model. Default is 2.
        drop_prob : float, optional
            Dropout probability. Default is 0.0.
        mask_center : bool, optional
            Whether to mask center of k-space for sensitivity map calculation.
            Default is True.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_udno = NormUDNO(
            chans,
            num_pools,
            radius_cutoff,
            in_shape,
            kernel_shape,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or any(
            torch.any(t == 0) for t in num_low_frequencies
        ):
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        images, batches = chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            batch_chans_to_chan_dim(self.norm_udno(images), batches)
        )


class VarNetBlock(nn.Module):
    """
    Model block for iterative refinement of k-space data.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.

    aka Refinement Module in Fig 1
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        use_dc_term: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            current_kspace: The current k-space data (frequency domain data)
                            being processed by the network. (torch.Tensor)
            ref_kspace: Original subsampled k-space data (from which we are
                reconstrucintg the image (reference k-space). (torch.Tensor)
            mask: A binary mask indicating the locations in k-space where
                data consistency should be enforced. (torch.Tensor)
            sens_maps: Sensitivity maps for the different coils in parallel
                    imaging. (torch.Tensor)
        """

        # model-term see orange box of Fig 1 in E2E-VarNet paper!
        # multi channel k-space -> single channel image-space
        b, c, h, w, _ = current_kspace.shape

        if c == 30:
            # get kspace and inpainted kspace
            kspace = current_kspace[:, :15, :, :, :]
            in_kspace = current_kspace[:, 15:, :, :, :]
            # convert to image space
            image = sens_reduce(kspace, sens_maps)
            in_image = sens_reduce(in_kspace, sens_maps)
            # concatenate both onto each other
            reduced_image = torch.cat([image, in_image], dim=1)
        else:
            reduced_image = sens_reduce(current_kspace, sens_maps)

        # single channel image-space
        refined_image = self.model(reduced_image)

        # single channel image-space -> multi channel k-space
        model_term = sens_expand(refined_image, sens_maps)

        # only use first 15 channels (masked_kspace) in the update
        # current_kspace = current_kspace[:, :15, :, :, :]

        if not use_dc_term:
            return current_kspace - model_term

        """
        Soft data consistency term:
            - Calculates the difference between current k-space and reference k-space where the mask is true.
            - Multiplies this difference by the data consistency weight.
        """
        # dc_term: see green box of Fig 1 in E2E-VarNet paper!
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        return current_kspace - soft_dc - model_term


class NOVarnet(nn.Module):
    """
    Neural Operator model for MRI reconstruction.

    Uses a variational architecture (iterative updates) with a learned sensitivity
    model. All operations are resolution invariant employing neural operator
    modules (UDNO).
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        kno_chans: int = 16,
        kno_pools: int = 4,
        kno_radius_cutoff: float = 0.02,
        kno_kernel_shape: Tuple[int, int] = (6, 7),
        radius_cutoff: float = 0.01,
        kernel_shape: Tuple[int, int] = (3, 4),
        in_shape: Tuple[int, int] = (640, 320),
        mask_center: bool = True,
        use_dc_term: bool = True,
        reduction_method: Literal["batch", "rss"] = "rss",
        skip_method: Literal["replace", "add", "add_inv", "concat"] = "add",
    ):
        """
        Parameters
        ----------
        num_cascades : int
            Number of cascades (i.e., layers) for variational network.
        sens_chans : int
            Number of channels for sensitivity map U-Net.
        sens_pools : int
            Number of downsampling and upsampling layers for sensitivity map U-Net.
        chans : int
            Number of channels for cascade U-Net.
        pools : int
            Number of downsampling and upsampling layers for cascade U-Net.
        mask_center : bool
            Whether to mask center of k-space for sensitivity map calculation.
        use_dc_term : bool
            Whether to use the data consistency term.
        reduction_method : "batch" or "rss"
            Method for reducing sensitivity maps to single channel.
            "batch" reduces to single channel by stacking channels.
            "rss" reduces to single channel by root sum of squares.
        skip_method : "replace" or "add" or "add_inv" or "concat"
            "replace" replaces the input with the output of the KNO
            "add" adds the output of the KNO to the input
            "add_inv" adds the output of the KNO to the input (only where samples are missing)
            "concat" concatenates the output of the KNO to the input
        """

        super().__init__()

        self.sens_net = SensitivityModel(
            sens_chans,
            sens_pools,
            radius_cutoff,
            in_shape,
            kernel_shape,
            mask_center=mask_center,
        )
        self.kno = NormUDNO(
            kno_chans,
            kno_pools,
            in_shape=in_shape,
            radius_cutoff=radius_cutoff,
            kernel_shape=kernel_shape,
            # radius_cutoff=kno_radius_cutoff,
            # kernel_shape=kno_kernel_shape,
            in_chans=2,
            out_chans=2,
        )
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    NormUDNO(
                        chans,
                        pools,
                        radius_cutoff,
                        in_shape,
                        kernel_shape,
                        in_chans=(
                            4 if skip_method == "concat" and cascade_idx == 0 else 2
                        ),
                        out_chans=2,
                    )
                )
                for cascade_idx in range(num_cascades)
            ]
        )
        self.use_dc_term = use_dc_term
        self.reduction_method = reduction_method
        self.skip_method = skip_method

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        # (B, C, X, Y, 2)
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)

        # reduce before inpainting
        if self.reduction_method == "rss":
            # (B, 1, H, W, 2) single channel image space
            x_reduced = sens_reduce(masked_kspace, sens_maps)
            # (B, 1, H, W, 2)
            k_reduced = fastmri.fft2c(x_reduced)
        elif self.reduction_method == "batch":
            k_reduced, b = chans_to_batch_dim(masked_kspace)

        # inpainting
        if self.skip_method == "replace":
            kspace_pred = self.kno(k_reduced)
        elif self.skip_method == "add_inv":
            # FIXME: this is not correct (mask has shape B, 1, H, W, 2 and self.gno(k_reduced) has shape B*C, 1, H, W, 2)
            kspace_pred = k_reduced.clone() + (~mask * self.kno(k_reduced))
        elif self.skip_method == "add":
            kspace_pred = k_reduced.clone() + self.kno(k_reduced)
        elif self.skip_method == "concat":
            kspace_pred = torch.cat([k_reduced.clone(), self.kno(k_reduced)], dim=1)
        else:
            raise NotImplementedError("skip_method not implemented")

        # expand after inpainting
        if self.reduction_method == "rss":
            if self.skip_method == "concat":
                # kspace_pred is (B, 2, H, W, 2)
                kspace = kspace_pred[:, :1, :, :, :]
                in_kspace = kspace_pred[:, 1:, :, :, :]
                # B, 2C, H, W, 2
                kspace_pred = torch.cat(
                    [sens_expand(kspace, sens_maps), sens_expand(in_kspace, sens_maps)],
                    dim=1,
                )
            else:
                # (B, 1, H, W, 2) -> (B, C, H, W, 2) multi-channel k space
                kspace_pred = sens_expand(kspace_pred, sens_maps)
        elif self.reduction_method == "batch":
            # (B, C, H, W, 2) multi-channel k space
            if self.skip_method == "concat":
                kspace = kspace_pred[:, :1, :, :, :]
                in_kspace = kspace_pred[:, 1:, :, :, :]
                # B, 2C, H, W, 2
                kspace_pred = torch.cat(
                    [
                        batch_chans_to_chan_dim(kspace, b),
                        batch_chans_to_chan_dim(in_kspace, b),
                    ],
                    dim=1,
                )
            else:
                kspace_pred = batch_chans_to_chan_dim(kspace_pred, b)

        # iterative update
        for cascade in self.cascades:
            kspace_pred = cascade(
                kspace_pred, masked_kspace, mask, sens_maps, self.use_dc_term
            )

        spatial_pred = fastmri.ifft2c(kspace_pred)
        spatial_pred_abs = fastmri.complex_abs(spatial_pred)
        combined_spatial = fastmri.rss(spatial_pred_abs, dim=1)

        return combined_spatial

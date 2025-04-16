"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from sigpy.mri import poisson, radial, spiral


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of"
                " accelerations if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        center_mask, accel_mask, num_low_frequencies = self.sample_mask(shape, offset)
        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        if len(mask.shape) == 1:
            mask = torch.tensor(mask)
            mask_num_freqs = len(mask)
            mask = mask.reshape(1, 1, mask_num_freqs, 1)
            mask = mask.expand(shape)
        return mask.expand(shape)

    def reshape_mask_old(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for s in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols, dtype=np.float32)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


class MagicMaskFunc(MaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFunc``, this mask will usually undereshoot the
    target acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=acceleration)

        if offset % 2 == 0:
            offset_pos = offset + 1
            offset_neg = offset + 2
        else:
            offset_pos = offset - 1 + 3
            offset_neg = offset - 1 + 0

        poslen = (num_cols + 1) // 2
        neglen = num_cols - (num_cols + 1) // 2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        return np.fft.fftshift(mask)  # shift mask and return


class MagicMaskFractionFunc(MagicMaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFractionFunc``, this method exactly matches
    the target acceleration by adjusting the offsets.
    """

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        fraction_low_freqs, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_frequencies = round(num_cols * fraction_low_freqs)

        # bound the number of low frequencies between 1 and target columns
        target_columns_to_sample = round(num_cols / acceleration)
        num_low_frequencies = max(min(num_low_frequencies, target_columns_to_sample), 1)

        # adjust acceleration rate based on target acceleration.
        adjusted_target_columns_to_sample = (
            target_columns_to_sample - num_low_frequencies
        )
        adjusted_acceleration = 0
        if adjusted_target_columns_to_sample > 0:
            adjusted_acceleration = round(num_cols / adjusted_target_columns_to_sample)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        accel_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, adjusted_acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, accel_mask, num_low_frequencies


class Gaussian2DMaskFunc(MaskFunc):
    """Gaussian 2D Masking

    Args:
        MaskFunc (_type_): _description_
    """

    def __init__(
        self,
        accelerations: Sequence[int],
        stds: Sequence[float],
        seed: Optional[int] = None,
    ):
        """initialize Gaussian 2D Mask

        Args:
            accelerations (Sequence[int]): list of acceleration factors, when
                generating a mask, an acceleration factor from this list will be chosen
            stds (Sequence[float]): list of torch.Normal scale (~std) to choose from
            seed (Optional[int], optional): Seed for selecting mask parameters. Defaults to None.
        """
        self.rng = np.random.RandomState(seed)
        self.accelerations = accelerations
        self.stds = stds

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        acceleration = self.rng.choice(self.accelerations)
        std = self.rng.choice(self.stds)

        x, y = shape[-3], shape[-2]
        mean_x = x // 2
        mean_y = y // 2
        num_samples_collected = 0

        dist = D.Normal(
            loc=torch.tensor([mean_x, mean_y], dtype=torch.float32),
            scale=std,
        )

        N = (
            int(1 / acceleration * x * y) + 10000
        )  # add constant or won't reach desired subsampling rate
        sample_x, sample_y = (
            torch.zeros(N, dtype=torch.int),
            torch.zeros(N, dtype=torch.int),
        )

        while num_samples_collected < N:
            samples = dist.sample((N,))  # type: ignore
            valid_samples = (
                (samples[:, 0] >= 0)
                & (samples[:, 0] < x)
                & (samples[:, 1] >= 0)
                & (samples[:, 1] < y)
            )

            valid_x = samples[valid_samples, 0].int()
            valid_y = samples[valid_samples, 1].int()

            num_to_take = min(N - num_samples_collected, valid_x.size(0))
            sample_x[num_samples_collected : num_samples_collected + num_to_take] = (
                valid_x[:num_to_take]
            )
            sample_y[num_samples_collected : num_samples_collected + num_to_take] = (
                valid_y[:num_to_take]
            )
            num_samples_collected += num_to_take

        mask = torch.zeros((x, y))
        mask[sample_x, sample_y] = 1.0

        # broadcasting mask (x, y) --> (N, x, y, C) C=2, N=batch_size
        mask = mask.unsqueeze(-1)  # (x, y, 1)
        mask = mask.unsqueeze(0)  # (1, x, y, 1)
        mask = mask.expand((1, mask.shape[1], mask.shape[2], 2)).clone()

        # num_low_freqs doesn't make sense so just return std (a number)
        # returning None doesn't work since we can't stack for multiple batches
        return mask, std


class Poisson2DMaskFunc(MaskFunc):
    """
    Variable Density Poisson Disk Sampling
    https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.poisson.html#sigpy.mri.poisson
    """

    def __init__(
        self,
        accelerations: Sequence[int],
        stds: None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ):
        """initialize VDPD (Poisson) mask

        Args:
            accelerations (Sequence[int]): list of acceleration factors to
                choose from
            stds: Dummy param. Do not pass value. Defaults to None.
            seed (Optional[int], optional): Seed for selecting mask params.
                Defaults to None.
        """
        self.rng = np.random.RandomState(seed)
        self.accelerations = accelerations
        self.use_cache = use_cache
        if use_cache:
            self.cache: Dict[int, np.ndarray] = dict()
            for acc in accelerations:
                assert os.path.exists(f"fastmri/poisson_cache/poisson_{acc}x.npy")
                self.cache[acc] = np.load(f"fastmri/poisson_cache/poisson_{acc}x.npy")

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.use_cache:
            acceleration = self.rng.choice(self.accelerations)
            return torch.from_numpy(self.cache[acceleration]), 1.0  # type: ignore
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        acceleration = self.rng.choice(self.accelerations)
        x, y = shape[-3], shape[-2]

        mask = poisson(img_shape=(x, y), accel=acceleration, dtype=np.float32)
        mask = torch.from_numpy(mask)

        # broadcasting mask (x, y) --> (N, x, y, C) C=2, N=batch_size
        mask = mask.unsqueeze(-1)  # (x, y, 1e
        mask = mask.unsqueeze(0)  # (1, x, y, 1)
        mask = mask.expand((1, mask.shape[1], mask.shape[2], 2)).clone()

        # num low freqs doesn't make sense here, so we return arbitrary value 1.0
        return mask, 100.0


class Radial2DMaskFunc(MaskFunc):
    """
    Radial trajectory MRI masking method.
    https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.radial.html#sigpy.mri.radial
    """

    def __init__(
        self,
        accelerations: Sequence[int],
        arms: Optional[Sequence[int]],
        seed: Optional[int] = None,
    ):
        """
        initialize Radial mask

        Args:
            accelerations (Sequence[int]): list of acceleration factors to
                choose from
            arms: Number of radial arms.
            seed (Optional[int], optional): Seed for selecting mask params.
                Defaults to None.
        """
        self.rng = np.random.RandomState(seed)
        self.accelerations = accelerations
        self.arms = arms

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        acceleration = self.rng.choice(self.accelerations)
        x, y = shape[-3], shape[-2]
        npoints = int(x * y * (1 / acceleration))
        if self.arms:
            arms = self.rng.choice(self.arms)
        else:
            points_per_arm = x // 3
            arms = npoints // points_per_arm

        # calculate radial parameters to satisfy acceleration factor
        ntr = arms  # num radial lines
        nro = npoints // arms  # num points on each radial line
        ndim = 2  # 2D

        # gen trajectory w/ shape (ntr, nro, ndim)
        traj = radial(
            coord_shape=[ntr, nro, ndim],
            img_shape=(x, y),
            golden=True,
            dtype=int,
        )

        mask = torch.zeros(x, y, dtype=torch.float32)
        x_coords = traj[..., 0].flatten() + (x // 2)
        y_coords = traj[..., 1].flatten() + (y // 2)
        mask[x_coords, y_coords] = 1.0

        # broadcasting mask (x, y) --> (N, x, y, C) C=2, N=batch_size
        mask = mask.unsqueeze(-1)  # (x, y, 1)
        mask = mask.unsqueeze(0)  # (1, x, y, 1)
        mask = mask.expand((1, mask.shape[1], mask.shape[2], 2)).clone()

        # num low freqs doesn't make sense here, so we return arbitrary value 1.0
        return mask, 100.0


class Spiral2DMaskFunc(MaskFunc):
    """
    Spiral trajectory MRI masking method.
    https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.spiral.html#sigpy.mri.spiral
    """

    def __init__(
        self,
        accelerations: Sequence[int],
        arms: Sequence[int],
        seed: Optional[int] = None,
    ):
        """
        initialize Radial mask

        Args:
            accelerations (Sequence[int]): list of acceleration factors to
                choose from
            arms: Number of radial arms.
            seed (Optional[int], optional): Seed for selecting mask params.
                Defaults to None.
        """
        self.rng = np.random.RandomState(seed)
        self.accelerations = accelerations
        self.arms = arms

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        raise (NotImplementedError("Spiral2D not implemented"))


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Optional[Sequence],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "gaussian_2d":
        return Gaussian2DMaskFunc(
            stds=center_fractions,
            accelerations=accelerations,
        )
    elif mask_type_str == "poisson_2d":
        return Poisson2DMaskFunc(
            accelerations=accelerations,
            stds=None,
        )
    elif mask_type_str == "radial_2d":
        return Radial2DMaskFunc(
            accelerations=accelerations,
            arms=([int(arm) for arm in center_fractions] if center_fractions else None),
        )
    elif mask_type_str == "spiral_2d":
        raise NotImplementedError("spiral_2d not implemented")
    else:
        raise ValueError(f"{mask_type_str} not supported")

import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import lmdb
import numpy as np
import pandas as pd
import torch
import yaml

import fastmri
import fastmri.transforms as T


class RawSample(NamedTuple):
    fname: Path
    slice_num: int
    metadata: Dict[str, Any]


class SliceSample(NamedTuple):
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: int
    target: torch.Tensor
    max_value: float
    # attrs: Dict[str, Any]
    fname: str
    slice_num: int


class SliceSampleMVUE(NamedTuple):
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: int
    target: torch.Tensor
    rss: torch.Tensor
    max_value: float
    # attrs: Dict[str, Any]
    fname: str
    slice_num: int


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    Query an XML document using ElementTree.

    This function allows querying an XML document by specifying a root and a list of nested queries.
    It supports optional XML namespaces.

    Parameters
    ----------
    root : ElementTree.Element
        The root element of the XML to search through.
    qlist : list of str
        A list of strings for nested searches, e.g., ["Encoding", "matrixSize"].
    namespace : str, optional
        An optional XML namespace to prepend to the query (default is None).

    Returns
    -------
    str
        The retrieved data as a string.
    """

    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(torch.utils.data.Dataset):
    """
    A simplified PyTorch Dataset that provides access to multicoil MR image
    slices from the fastMRI dataset.
    """

    def __init__(
        self,
        # root: Optional[Path | str],
        body_part: Literal["knee", "brain"],
        partition: Literal["train", "val", "test"],
        mask_fns: Optional[List[Callable]] = None,
        sample_rate: float = 1.0,
        complex: bool = False,
        crop_shape: Tuple[int, int] = (320, 320),
        slug: str = "",
        contrast: Optional[Literal["T1", "T2"]] = None,
        coils: Optional[int] = None,
    ):
        """
        Initializes the fastMRI multi-coil challenge dataset.

        Samples are individual 2D slices taken from k-space volume data.

        Parameters
        ----------
        body_part : {'knee', 'brain'}
            The body part to analyze.
        partition : {'train', 'val', 'test'}
            The data partition type.
        mask_fns : list of callable, optional
            A list of masking functions to apply to samples.
            If multiple are given, a mask is randomly chosen for each sample.
        sample_rate : float, optional
            Fraction of data to sample, by default 1.0.
        complex : bool, optional
            Whether the $k$-space data should return complex-valued, by default False.
            If True, kspace values will be complex.
            If False, kspace values will be real (shape, 2).
        crop_shape : tuple of two ints, optional
            The shape to center crop the k-space data, by default (320, 320).
        slug : string
            dataset slug name
        contrast :  {'T1', 'T2'}
            If partition is brain, the contrast of images to use.
        """

        with open("fastmri.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.contrast = contrast
        self.slug = slug
        self.partition = partition
        self.body_part = body_part
        self.root = Path(config.get(f"{body_part}_path")) / f"multicoil_{partition}"
        self.mask_fns = mask_fns
        self.sample_rate = sample_rate
        self.raw_samples: List[RawSample] = self._load_samples()
        self.complex = complex
        self.crop_shape = crop_shape
        self.coils = coils

    def _load_samples(self):
        # Gather all files in the root directory
        if self.body_part == "brain" and self.contrast:
            files = list(self.root.glob(f"*{self.contrast}*.h5"))
        else:
            files = list(self.root.glob("*.h5"))
        raw_samples = []

        # Load and process metadata from each file
        for fname in sorted(files):
            with h5py.File(fname, "r") as hf:
                metadata, num_slices = self._retrieve_metadata(fname)

                # Collect samples for each slice, discard first c slices, and last c slices
                c = 6
                for slice_num in range(num_slices):
                    if c <= slice_num <= num_slices - c - 1:
                        raw_samples.append(RawSample(fname, slice_num, metadata))

        # Subsample if desired
        if self.sample_rate < 1.0:
            raw_samples = random.sample(
                raw_samples, int(len(raw_samples) * self.sample_rate)
            )

        return raw_samples

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx) -> SliceSample:
        try:
            raw_sample: RawSample = self.raw_samples[idx]
            fname, slice_num, metadata = raw_sample

            # load kspace and target
            with h5py.File(fname, "r") as hf:
                kspace = torch.tensor(hf["kspace"][()][slice_num])
                if not self.complex:
                    kspace = torch.view_as_real(kspace)
                if self.coils:
                    if kspace.shape[0] < self.coils:
                        return None
                    kspace = kspace[: self.coils, :, :, :]
                target_key = (
                    "reconstruction_rss"
                    if self.partition in ["train", "val"]
                    else "reconstruction_esc"
                )
                target = hf.get(target_key, None)
                if target is not None:
                    target = torch.tensor(target[()][slice_num])
                if self.body_part == "brain":
                    target = T.center_crop(target, self.crop_shape)

            # center crop to enable collating for batching
            if self.complex:
                # if complex, crop across dims: -2 and -1 (last 2)
                raise NotImplementedError("Not implemented for complex native")
            else:
                # crop in image space, to not lose high-frequency information
                image = fastmri.ifft2c(kspace)
                image_cropped = T.complex_center_crop(image, self.crop_shape)
                kspace = fastmri.fft2c(image_cropped)

            # apply transform mask if there is one
            if self.mask_fns:
                # choose a random mask
                mask_fn = random.choice(self.mask_fns)
                kspace, mask, num_low_frequencies = T.apply_mask(
                    kspace,
                    mask_fn,
                    # seed=seed,
                )
                mask = mask.bool()
            else:
                mask = torch.ones_like(kspace, dtype=torch.bool)
                num_low_frequencies = 0
            sample = SliceSample(
                kspace,
                mask,
                num_low_frequencies,
                target,
                metadata["max"],
                fname.name,
                slice_num,
            )
            return sample
        except:
            return None


class SliceDatasetLMDB(torch.utils.data.Dataset):
    """
    A simplified PyTorch Dataset that provides access to multicoil MR image
    slices from the fastMRI dataset. Loads from LMDB saved samples.
    """

    def __init__(
        self,
        body_part: Literal["knee", "brain"],
        partition: Literal["train", "val", "test"],
        root: Optional[Path | str] = None,
        mask_fns: Optional[List[Callable]] = None,
        sample_rate: float = 1.0,
        complex: bool = False,
        crop_shape: Tuple[int, int] = (320, 320),
        slug: str = "",
        coils: int = 15,
    ):
        """
        Initializes the fastMRI multi-coil challenge dataset.

        Samples are individual 2D slices taken from k-space volume data.

        Parameters
        ----------
        body_part : {'knee', 'brain'}
            The body part to analyze.
        root : Path or str, optional
            Root to lmdb dataset. If not provided, the root is automatically
            loaded directly from fastmri.yaml config
        partition : {'train', 'val', 'test'}
            The data partition type.
        mask_fns : list of callable, optional
            A list of masking functions to apply to samples.
            If multiple are given, a mask is randomly chosen for each sample.
        sample_rate : float, optional
            Fraction of data to sample, by default 1.0.
        complex : bool, optional
            Whether the $k$-space data should return complex-valued, by default False.
            If True, kspace values will be complex.
            If False, kspace values will be real (shape, 2).
        crop_shape : tuple of two ints, optional
            The shape to center crop the k-space data, by default (320, 320).
        slug : string
            dataset slug name
        """

        # set attrs
        self.coils = coils
        self.slug = slug
        self.partition = partition
        self.mask_fns = mask_fns
        self.sample_rate = sample_rate
        self.complex = complex
        self.crop_shape = crop_shape

        # load lmdb info
        if root:
            if isinstance(root, str):
                root = Path(root)
            assert root.exists(), "Provided root doesn't exist."
            self.root = root
        else:
            with open("fastmri.yaml", "r") as file:
                config = yaml.safe_load(file)
            self.root = Path(config["lmdb"][f"{body_part}_{partition}_path"])
        self.meta = np.load(self.root / "meta.npy")
        self.kspace_env = lmdb.open(
            str(self.root / "kspace"),
            readonly=True,
            lock=False,
            create=False,
        )
        self.kspace_txn = self.kspace_env.begin(write=False)
        self.rss_env = lmdb.open(
            str(self.root / "rss"),
            readonly=True,
            lock=False,
            create=False,
        )
        self.rss_txn = self.rss_env.begin(write=False)
        self.length = self.kspace_txn.stat()["entries"]

    def __len__(self):
        return int(self.sample_rate * self.length)

    def __getitem__(self, idx) -> SliceSample:
        idx_key = str(idx).encode("utf-8")

        # load sample data
        kspace = torch.from_numpy(
            np.frombuffer(self.kspace_txn.get(idx_key), dtype=np.float32)
            .reshape(self.coils, 320, 320, 2)
            .copy()
        )
        rss = torch.from_numpy(
            np.frombuffer(self.rss_txn.get(idx_key), dtype=np.float32)
            .reshape(320, 320)
            .copy()
        )

        # crop in image space, to not lose high-frequency information
        if self.crop_shape and self.crop_shape != (320, 320):
            image = fastmri.ifft2c(kspace)
            image_cropped = T.complex_center_crop(image, self.crop_shape)
            kspace = fastmri.fft2c(image_cropped)
            rss = T.center_crop(rss, self.crop_shape)

        # load and apply mask
        if self.mask_fns:
            # choose a random mask
            mask_fn = random.choice(self.mask_fns)
            kspace, mask, num_low_frequencies = T.apply_mask(
                kspace,
                mask_fn,  # type: ignore
            )
            mask = mask.bool()
        else:
            mask = torch.ones_like(kspace, dtype=torch.bool)
            num_low_frequencies = 0

        # load metadata
        fname, slice_num, max_value = self.meta[idx]
        fname = str(fname)
        slice_num = int(slice_num)
        max_value = float(max_value)

        return SliceSample(
            kspace,
            mask,
            num_low_frequencies,
            rss,
            max_value,
            fname,
            slice_num,
        )

"""
Transform SliceDataset into LMDB dataset (SliceDatasetLMDB)
"""

import argparse
import os
from pathlib import Path

import lmdb
import numpy as np
import tqdm

from fastmri.datasets import SliceDataset, SliceSample

KNEE_COILS = 15
BRAIN_COILS = 16


def main(args):
    num_coils = KNEE_COILS
    contrast = None  # don't filter by contrast for knee (there is only 1)
    if args.body_part == "brain":
        num_coils = BRAIN_COILS
        contrast = "T2"

    dataset = SliceDataset(
        args.body_part,
        partition=args.partition,
        complex=False,
        sample_rate=args.sample_rate,
        crop_shape=(320, 320),
        contrast=contrast,
        coils=num_coils,
    )
    process(
        dataset,
        num_coils,
        args.out_path,
    )


def process(
    dataset: SliceDataset,
    coils: int,
    out_dir: Path | str,
    n_jobs=-1,
    chunk_size=10,
):
    N = len(dataset)

    kspace_arr = np.zeros((N, coils, 320, 320, 2), dtype=np.float32)  # type: ignore
    rss_arr = np.zeros((N, 320, 320), dtype=np.float32)  # type: ignore
    meta = []

    N_actual = 0
    for i in tqdm.trange(N):
        sample: SliceSample = dataset[i]
        if sample == None:
            continue
        kspace = sample.masked_kspace
        target = sample.target  # rss targets
        max_value = sample.max_value
        fname = sample.fname
        slice_num = sample.slice_num

        kspace_arr[N_actual] = kspace
        rss_arr[N_actual] = target
        meta.append((fname, slice_num, max_value))

        N_actual += 1

    os.makedirs(out_dir, exist_ok=True)

    # Save kspace
    kspace_arr = kspace_arr[:N_actual]
    env = lmdb.open(f"{out_dir}/kspace", map_size=int(1e12), readahead=False)
    save2db(kspace_arr, env, 0)
    env.close()

    # Save rss target
    rss_arr = rss_arr[:N_actual]
    env = lmdb.open(f"{out_dir}/rss", map_size=int(1e12), readahead=False)
    save2db(rss_arr, env, 0)
    env.close()

    # Save meta
    np.save(f"{out_dir}/meta.npy", meta)


def save2db(batch_data, env, cur):
    """
    Args:
        - batch_data (np.ndarray): (batchsize, H, W)
        - env (lmdb.Environment): lmdb environment
        - cur (int): current index
    """
    num_samples = batch_data.shape[0]
    with env.begin(write=True) as txn:
        for i in range(num_samples):
            key = f"{cur + i}".encode()
            txn.put(key, batch_data[i])
    return cur + num_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script to convert SliceDataset samples into lmdb format."
    )
    parser.add_argument(
        "--body_part",
        "-bp",
        type=str,
        choices=["brain", "knee"],
        required=True,
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        choices=["train", "val"],
        required=True,
    )
    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

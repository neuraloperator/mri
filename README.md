[![arXiv](https://img.shields.io/badge/arXiv-2410.16290-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.16290)
[![](https://img.shields.io/badge/Blog-armeet.ca%2Fnomri-yellow?style=flat-square)](https://armeet.ca/nomri)

# A Unified Model for Compressed Sensing MRI Across Undersampling Patterns

> [**A Unified Model for Compressed Sensing MRI Across Undersampling Patterns**](https://arxiv.org/abs/2410.16290)  
> Armeet Singh Jatyani, Jiayun Wang, Aditi Chandrashekar, Zihui Wu, Miguel Liu-Schiaffini, Bahareh Tolooshams, Anima Anandkumar  
> *Paper at [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)*

![intro](https://github.com/user-attachments/assets/79aee2fd-0956-4a05-b6c8-2037618e47b1)

> _**(a) Unified Model:** NO works across various undersampling patterns, unlike CNNs (e.g., [E2E-VarNet](#)) that need separate models for each._ \
> _**(b) Consistent Performance:** NO consistently outperforms CNNs, especially for 2× acceleration with a single unrolled cascade._ \
> _**(c) Resolution-Agnostic:** Maintains fixed kernel size regardless of image resolution, reducing aliasing risks._ \
> _**(d) Zero-Shot Super-Resolution:** Outperforms CNNs in reconstructing high-res MRIs without retraining._

![super](https://github.com/user-attachments/assets/3675a80e-c05f-4d41-9fdf-531de0576751)

> _**(a) Zero-Shot Extended FOV:** On 4x Gaussian undersampling, NO achieves higher PSNR and fewer artifacts than E2E-VN, despite both models being trained only on 160 x 160 FOV._ \
> _**(b) Zero-Shot Super-Resolution in Image Space:** For 2x radial with 640 x 640 input via bilinear upsampling, NO preserves quality while E2E-VN introduces artifacts._

## Requirements
We have tested training/inference on the following hardware/software versions, however there is no reason it shouldn't work on slightly older driver/cuda versions.
- tested on RTX 4090 and A100 with CUDA 12.4 and NVML/Driver version 550
- Ubuntu 22.04.3 LTS & SUSE Linux Enterprise Server 15
- All python packages are in `pyproject.toml` (see Setup)

## Setup

We use `uv` for environment setup. It is 10-100x faster than vanilla pip and conda. If you don't have `uv`, please install it from [here](https://docs.astral.sh/uv/getting-started/installation/) (no sudo required). If you're on a Linux environment you can install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Of course, if you would like to use a virtual environment handled by vanilla python or conda, all package and their versions are provided in `pyproject.toml` under "dependencies."

In the root directory, run
```bash
uv sync
```

Then you can activate the environment with:
```bash
source .venv/bin/activate
```
Note this is optional. You can run scripts with this venv without activating the environment by using `uv run python script.py` or abbreviated `uv run script.py`.

 `uv` will create a virtual environment for you and install all packages.

Then to download the pretrained weights, run:
```bash
uv run scripts/download_weights.py
```
This downloads pretrained weights into the `weights/` directory.

Finally to run scripts, make them executable:
```bash
chmod u+x scripts/*
```

Then you can run any script. For example:
```bash
./scripts/knee_multipatt.sh
```

By default weights & biases (WANDB) is disabled, so scripts will print results to stdout. If you want to visualize results in 
weights and biases, add your WANDB api key at the top of the script. We log image predictions
as well as PSNR, NMSE, SSIM metrics for each epoch.
```bash
export WANDB_API_KEY=***************
```

Before you can begin training/inference, you will need to download and process the dataset. See the "Datasets" section below.

## Datasets

We use the fastMRI dataset, which can be downloaded [here](https://fastmri.med.nyu.edu/). \
Dataset classes are provided in `fastmri/datasets.py`:
- `SliceDatasetLMDB`: dataset in significantly faster LMDB format
- `SliceDataset`: dataset class for original fastMRI dataset

We convert the raw fastMRI HDF5 formatted samples into a significantly faster LMDB format.
This accelerates training/validation by a significant factor. Once you have downloaded the fastMRI dataset,
you will need to run `scripts/gen_lmdb_dataset.py` to convert the original fastMRI dataset into LMDB format.

```bash
uv run scripts/gen_lmdb_dataset.py --body_part brain --partition val -o /path/to/lmdb/dataset
```

Do this for every dataset you need: (brain, knee) x (train, val). To choose a smaller subset for faster training/inference add `--sample_rate 0.Xx`.

By default we use the LMDB format. If you want to use the original SliceDataset class, you can swap out the dataset class in `main.py`.

Finally, modify your `fastmri.yaml` with the correct dataset paths

```yaml
log_path: /tmp/logs
checkpoint_path: /tmp/checkpoints

lmdb:
  knee_train_path: **/**/**/knee_train_lmdb
  knee_val_path: **/**/**/knee_val_lmdb
  brain_train_path: **/**/**/brain_train_lmdb
  brain_val_path: **/**/**/brain_val_lmdb
```

## Training and Validation

`main.py` is used for both training and validation. We follow the original fastMRI repo
and use Lightning. We provide both a simple PyTorch model `models/no_varnet.py` (if you want
a thinner abstraction), as well as a Lightning wrapped `models/lightning/no_varnet_module.py` that
makes distributed training across multiple GPUs easier.

## Citation

If you found our work helpful or used any of our models (UDNO), please cite the following:
```bibtex
@inproceedings{jatyani2025nomri,
  author    = {Armeet Singh Jatyani* and Jiayun Wang* and Aditi Chandrashekar and Zihui Wu and Miguel Liu-Schiaffini and Bahareh Tolooshams and Anima Anandkumar},
  title     = {A Unified Model for Compressed Sensing MRI Across Undersampling Patterns},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR) Proceedings},
  abbr      = {CVPR},
  year      = {2025}
}
```

![paper_preview](https://github.com/user-attachments/assets/7e6adaa5-a5fa-4b68-bd8c-5279f6f643d7)

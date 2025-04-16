import datetime
import itertools
import logging
import os
import random
import sys
import traceback
import uuid
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

import wandb
from fastmri.datasets import SliceDatasetLMDB
from fastmri.subsample import create_mask_for_mask_type
from models.lightning.no_varnet_module import NOVarnetModule
from models.lightning.varnet_module import VarNetModule
from type_utils import tuple_type

SEED = 999


def main(run_id, args):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pl.seed_everything(SEED)

    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Lightning version: {pl.__version__}")  # type: ignore
    logger.info(f"CUDA version: {torch.version.cuda}")  # type: ignore
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"Wandb version: {wandb.__version__}")

    # torch config
    torch.set_float32_matmul_precision("highest")

    # load paths from fastmri.yaml
    with open("fastmri.yaml", "r") as f:
        fmri_config = yaml.safe_load(f)

    acceleration_to_fractions = {
        1: 1,
        2: 0.16,
        4: 0.08,
        6: 0.06,
        8: 0.04,
        16: 0.02,
        32: 0.01,
    }

    # training setting
    if args.mode == "train":
        exp_train = {
            "wandb_name": args.name,
            "wandb_tags": [
                "training",
            ],
            "mask_type": args.train_patterns,
            "center_fractions": [
                acceleration_to_fractions[acc] for acc in args.train_accelerations
            ],
            "accelerations": args.train_accelerations,
            "stds": [70],
        }

        # create masks
        train_mask_fns = []
        for mask in exp_train["mask_type"]:
            if mask == "gaussian_2d":
                mask = create_mask_for_mask_type(
                    mask,
                    exp_train["stds"],
                    exp_train["accelerations"],
                )
            elif mask == "radial_2d":
                mask = create_mask_for_mask_type(
                    mask,
                    None,
                    exp_train["accelerations"],
                )
            else:
                mask = create_mask_for_mask_type(
                    mask,
                    exp_train["center_fractions"],
                    exp_train["accelerations"],
                )
            train_mask_fns.append(mask)

    # validation setting
    exp_val = {
        "val_mask_type": args.val_patterns,
        "val_center_fractions": [
            acceleration_to_fractions[acc] for acc in args.val_accelerations
        ],
        "val_accelerations": args.val_accelerations,
        "stds": [70],
    }

    val_mask_fns = []
    for pattern, acc in itertools.product(
        exp_val["val_mask_type"], exp_val["val_accelerations"]
    ):
        if pattern == "gaussian_2d":
            mask = create_mask_for_mask_type(
                pattern,
                exp_val["stds"],
                [acc],
            )
        elif pattern == "radial_2d":
            mask = create_mask_for_mask_type(
                pattern,
                None,  # calculate num arms dynamically
                [acc],
            )
        else:
            mask = create_mask_for_mask_type(
                pattern,
                [acceleration_to_fractions[acc]],
                [acc],
            )
        val_mask_fns.append((f"{pattern}-{acc}x", mask))
    if args.val_subset:
        val_datasets = [
            Subset(
                SliceDatasetLMDB(
                    args.body_part,
                    partition="val",
                    mask_fns=[fn],
                    complex=False,
                    sample_rate=1.0,
                    crop_shape=(320, 320),
                    slug=slug,
                    coils=16 if args.body_part == "brain" else 15,
                ),
                args.val_subset,
            )
            for slug, fn in val_mask_fns
        ]
    else:
        # for training, use smaller val
        val_samplerate = 0.1 if args.mode == "train" else args.sample_rate
        val_datasets = [
            SliceDatasetLMDB(
                args.body_part,
                partition="val",
                mask_fns=[fn],
                complex=False,
                sample_rate=val_samplerate,
                crop_shape=(320, 320),
                slug=slug,
                coils=16 if args.body_part == "brain" else 15,
            )
            for slug, fn in val_mask_fns
        ]
    val_dataloaders = {
        ds.slug: DataLoader(  # type: ignore
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for ds in val_datasets
    }
    combined_val_dataloader = CombinedLoader(val_dataloaders, mode="sequential")

    # datasets & dataloaders
    if args.mode == "train":
        train_dataset = SliceDatasetLMDB(
            args.body_part,
            partition="train",
            mask_fns=train_mask_fns,  # type: ignore
            complex=False,
            sample_rate=args.sample_rate,
            crop_shape=args.crop_shape,
            coils=16 if args.body_part == "brain" else 15,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # load model
    if args.model == "vn":
        module = VarNetModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
        )
    elif args.model == "no_vn":
        module = NOVarnetModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            kno_chans=args.kno_chans,
            kno_pools=args.kno_pools,
            kno_radius_cutoff=args.kno_radius_cutoff,
            kno_kernel_shape=args.kno_kernel_shape,
            radius_cutoff=args.radius_cutoff,
            kernel_shape=args.kernel_shape,
            in_shape=args.in_shape,
            use_dc_term=args.use_dc_term,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            reduction_method=args.reduction_method,
            skip_method=args.skip_method,
        )
    else:
        raise NotImplementedError("model not implemented!")

    # Init cloud logger (wandb)
    if args.no_logs:
        wandb_logger = WandbLogger(mode="disabled")
    else:
        wandb_logger = WandbLogger(
            project="no-medical",
            log_model=False,
            dir=(Path(fmri_config["log_path"])),
            entity="armeet-team",  # replace this with your wandb team name
            name=args.name,
            id=os.getenv("SLURM_JOB_ID", str(uuid.uuid4())),
            tags=args.wandb_tags,
            config={
                **vars(args),
                "slurm_job_id": os.getenv("SLURM_JOB_ID", None),
                "num_params": f"{module.num_params:,}",
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            },
        )

    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # type: ignore
        dirpath=(Path(fmri_config["checkpoint_path"]) / run_id),
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
    )

    # Trainer
    module.strict_loading = False
    trainer = L.Trainer(
        deterministic=True,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=args.devices,
        strategy="ddp" if args.num_nodes > 1 and args.devices > 1 else "auto",
        max_epochs=args.max_epochs,
        logger=[
            wandb_logger,
        ],
        callbacks=[
            checkpoint_callback,
        ],
    )

    print("RUN_ID", run_id)
    if args.mode == "train":
        trainer.fit(
            model=module,
            train_dataloaders=train_dataloader,  # type: ignore
            val_dataloaders=combined_val_dataloader,
            ckpt_path=args.ckpt_path,
        )
    elif args.mode == "val":
        trainer.validate(
            model=module,
            dataloaders=combined_val_dataloader,
            ckpt_path=args.ckpt_path,
        )
    else:
        raise ValueError("Invalid mode")


def build_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--body_part",
        type=str,
        required=True,
        choices=["knee", "brain"],
        help="Whether to use knee or brain dataset",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Wandb experiment group",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        required=False,
        help="Resume from checkpoint at path",
    )
    # trainer args
    parser.add_argument(
        "--max_epochs",
        required=False,
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_nodes",
        required=False,
        type=int,
        default=1,
        help="Number of training nodes (machines)",
    )
    parser.add_argument(
        "--devices",
        required=False,
        type=int,
        default=1,
        help="Number of training devices (gpus)",
    )

    # script args
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "val"],
        type=str,
        help="Mode of operation: train or validation",
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Wandb exp name",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size for run",
    )

    # model, pattern config
    parser.add_argument(
        "--model",
        required=True,
        # choices=("vn", "simple_no", "no_vn"),
        type=str,
        help="Model architecture to train",
    )

    # data subsampling args
    parser.add_argument(
        "--sample_rate",
        default=1.0,
        type=float,
        help="Sampling rate for the dataset (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--crop_shape",
        default=(320, 320),
        type=tuple_type,
        help="The shape to center crop the k-space data, by default (320, 320).",
    )
    parser.add_argument(
        "--train_accelerations",
        required=False,
        choices=(1, 2, 4, 6, 8, 16, 32),
        type=int,
        nargs="+",
        help="List of training accelerations, separated by spaces",
    )
    parser.add_argument(
        "--train_patterns",
        required=False,
        nargs="+",
        default=[
            "equispaced_fraction",
            "magic",
            "random",
            "gaussian_2d",
            "poisson_2d",
            "radial_2d",
        ],
        choices=(
            "equispaced_fraction",
            "magic",
            "random",
            "equispaced",
            "gaussian_2d",
            "poisson_2d",
            "radial_2d",
        ),
        type=str,
        help="List of training mask patterns, separated by spaces",
    )
    parser.add_argument(
        "--val_accelerations",
        required=True,
        choices=(1, 2, 4, 6, 8, 16, 32),
        type=int,
        nargs="+",
        help="List of validation accelerations, separated by spaces",
    )
    parser.add_argument(
        "--val_patterns",
        default=[
            "equispaced_fraction",
            "magic",
            "random",
            "gaussian_2d",
            "poisson_2d",
            "radial_2d",
        ],
        nargs="+",
        choices=(
            "equispaced_fraction",
            "magic",
            "random",
            "gaussian_2d",
            "poisson_2d",
            "radial_2d",
        ),
        type=str,
        help="List of validation mask patterns, separated by spaces",
    )

    parser.add_argument(
        "--val_subset",
        nargs="+",
        type=int,
        help="List of validation sample indices",
    )

    # misc: logging, debugging
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        help="List of wandb tags to add, separated by spaces",
    )

    parser.add_argument(
        "--no_logs",
        action="store_true",
        help="Disable logging if this flag is set",
    )

    args, _ = parser.parse_known_args()
    modules = [
        (VarNetModule, "vn"),
        (NOVarnetModule, "no_vn"),
    ]

    for module, model_name in modules:
        if args.model == model_name:
            parser = module.add_model_specific_args(parser)

    # hyperparams
    return parser.parse_args()


def config_logger(run_id, file_logging=False):
    """
    Configures logging to both the console and a file.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Define formatter for log messages
    formatter = logging.Formatter("%(asctime)s \t %(levelname)s \t %(message)s")

    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create a handler for file output
    if file_logging:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        file_handler = logging.FileHandler("logs/" + f"{run_id}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    if file_logging:
        logger.addHandler(file_handler)  # type: ignore

    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        """
        Logs uncaught exceptions with stack traces.
        """
        allow_keyboard_interrupt = False
        if allow_keyboard_interrupt and issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
        logger.warning(
            "Warning: An error occurred\n"
            + "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        )

    sys.excepthook = handle_uncaught_exception


if __name__ == "__main__":
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("RUN_ID: " + run_id)
    args = build_args()
    config_logger(run_id)

    # local logger
    logger = logging.getLogger()
    logger.info("Training started")
    main(run_id, args)
    logger.info("Training completed")

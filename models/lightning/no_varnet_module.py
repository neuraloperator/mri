from argparse import ArgumentParser
from typing import Tuple

import fastmri
import torch
from fastmri import transforms
from models.lightning.mri_module import MriModule
from models.no_varnet import NOVarnet
from type_utils import tuple_type


class NOVarnetModule(MriModule):
    """
    NO-Varnet training module.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        kno_pools: int = 4,
        kno_chans: int = 16,
        kno_radius_cutoff: float = 0.02,
        kno_kernel_shape: Tuple[int, int] = (6, 7),
        radius_cutoff: float = 0.02,
        kernel_shape: Tuple[int, int] = (6, 7),
        in_shape: Tuple[int, int] = (320, 320),
        use_dc_term: bool = True,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        reduction_method: str = "rss",
        skip_method: str = "add",
        **kwargs,
    ):
        """
        Parameters
        ----------
        num_cascades : int
            Number of cascades (i.e., layers) for the variational network.
        pools : int
            Number of downsampling and upsampling layers for the cascade U-Net.
        chans : int
            Number of channels for the cascade U-Net.
        sens_pools : int
            Number of downsampling and upsampling layers for the sensitivity map U-Net.
        sens_chans : int
            Number of channels for the sensitivity map U-Net.
        lr : float
            Learning rate.
        lr_step_size : int
            Learning rate step size.
        lr_gamma : float
            Learning rate gamma decay.
        weight_decay : float
            Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.kno_pools = kno_pools
        self.kno_chans = kno_chans
        self.kno_radius_cutoff = kno_radius_cutoff
        self.kno_kernel_shape = kno_kernel_shape
        self.radius_cutoff = radius_cutoff
        self.kernel_shape = kernel_shape
        self.in_shape = in_shape
        self.use_dc_term = use_dc_term
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.reduction_method = reduction_method
        self.skip_method = skip_method

        self.model = NOVarnet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
            kno_chans=self.kno_chans,
            kno_pools=self.kno_pools,
            kno_radius_cutoff=self.kno_radius_cutoff,
            kno_kernel_shape=self.kno_kernel_shape,
            radius_cutoff=radius_cutoff,
            kernel_shape=kernel_shape,
            in_shape=in_shape,
            use_dc_term=use_dc_term,
            reduction_method=reduction_method,
            skip_method=skip_method,
        )

        self.criterion = fastmri.SSIMLoss()
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.model(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        output = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies
        )

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        loss = self.criterion(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("epoch", int(self.current_epoch), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataloaders = self.trainer.val_dataloaders
        slug = list(dataloaders.keys())[dataloader_idx]

        output = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies
        )

        target, output = transforms.center_crop_to_smallest(batch.target, output)

        loss = self.criterion(
            output.unsqueeze(1),
            target.unsqueeze(1),
            data_range=batch.max_value,
        )

        return {
            "slug": slug,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": loss,
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help=("Number of pooling layers for sense map estimation U-Net in VarNet"),
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--kno_pools",
            default=4,
            type=int,
            help=("Number of pooling layers for KNO"),
        )
        parser.add_argument(
            "--kno_chans",
            default=16,
            type=int,
            help="Number of channels for KNO",
        )
        parser.add_argument(
            "--kno_radius_cutoff",
            default=0.02,
            type=float,
            required=False,
            help="KNO module radius_cutoff",
        )
        parser.add_argument(
            "--kno_kernel_shape",
            default=(6, 7),
            type=tuple_type,
            required=False,
            help="KNO module kernel_shape. Ex: 6,7 (no spaces please)",
        )
        parser.add_argument(
            "--radius_cutoff",
            default=0.02,
            type=float,
            required=False,
            help="DISCO module radius_cutoff",
        )
        parser.add_argument(
            "--kernel_shape",
            default=(6, 7),
            type=tuple_type,
            required=False,
            help="DISCO module kernel_shape. Ex: 6,7 (no spaces please)",
        )
        parser.add_argument(
            "--in_shape",
            default=(640, 320),
            type=tuple_type,
            required=True,
            help="Spatial dimensions of masked_kspace samples. Ex: 320,320 (no spaces)",
        )
        parser.add_argument(
            "--use_dc_term",
            default=True,
            type=bool,
            help="Whether to use the DC term in the unrolled iterative update step",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--reduction_method",
            default="batch",
            type=str,
            choices=["rss", "batch"],
            help="Reduction method used to reduce multi-channel k-space data before inpainting module. Read documentation of KNO for more information.",
        )
        parser.add_argument(
            "--skip_method",
            default="replace",
            type=str,
            choices=["add_inv", "add", "concat", "replace"],
            help="Method for skip connection around inpainting module.",
        )

        return parser

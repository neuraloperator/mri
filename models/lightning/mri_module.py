"""
Modified for use in <TODO: paper name>
- minified and removed extraneous abstractions
- updated to latest version of lightning

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from collections import defaultdict
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchmetrics.metric import Metric

import lightning as L

matplotlib.use("Agg")

from fastmri import evaluate


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(L.LightningModule):
    """
    Abstract super class for deep learning reconstruction models.

    This is a subclass of the LightningModule class from lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step: Define what happens in one step of training
        - validation_step: Define what happens in one step of validation
        - test_step: Define what happens in one step of testing
        - configure_optimizers: Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16):
        """
        Initialize the MRI module.

        Parameters
        ----------
        num_log_images : int, optional
            Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = [1, 2, 3, 4, 5]
        self.val_batch_results = []

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def log_image(self, name, image):
        if self.logger is not None:
            self.logger.log_image(
                key=f"{name}", images=[image], caption=[{self.global_step}]
            )

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # breakpoint()
        val_logs = outputs

        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        for i, fname in enumerate(val_logs["fname"]):
            if i == 0 and batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)
            slice_num = int(val_logs["slice_num"][i].cpu())

            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()
            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        self.val_batch_results.append(
            {
                "slug": val_logs["slug"],
                "val_loss": val_logs["val_loss"],
                "mse_vals": dict(mse_vals),
                "target_norms": dict(target_norms),
                "ssim_vals": dict(ssim_vals),
                "max_vals": max_vals,
            }
        )

    def on_validation_epoch_end(self):
        val_logs = self.val_batch_results

        dataset_metrics = defaultdict(
            lambda: {
                "losses": [],
                "mse_vals": defaultdict(dict),
                "target_norms": defaultdict(dict),
                "ssim_vals": defaultdict(dict),
                "max_vals": dict(),
            }
        )

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            slug = val_log["slug"]
            dataset_metrics[slug]["losses"].append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                dataset_metrics[slug]["mse_vals"][k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                dataset_metrics[slug]["target_norms"][k].update(
                    val_log["target_norms"][k]
                )
            for k in val_log["ssim_vals"].keys():
                dataset_metrics[slug]["ssim_vals"][k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                dataset_metrics[slug]["max_vals"][k] = val_log["max_vals"][k]

        metrics_to_plot = {"psnr": [], "ssim": [], "nmse": []}
        slugs = []

        for slug, metrics_data in dataset_metrics.items():
            mse_vals, target_norms, ssim_vals, max_vals, losses = (
                metrics_data["mse_vals"],
                metrics_data["target_norms"],
                metrics_data["ssim_vals"],
                metrics_data["max_vals"],
                metrics_data["losses"],
            )
            # check to make sure we have all files in all metrics
            assert (
                mse_vals.keys()
                == target_norms.keys()
                == ssim_vals.keys()
                == max_vals.keys()
            )

            # apply means across image volumes
            metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
            metric_values = {
                "nmse": [],
                "ssim": [],
                "psnr": [],
            }  # to store individual values for std
            local_examples = 0

            for fname in mse_vals.keys():
                local_examples = local_examples + 1
                mse_val = torch.mean(
                    torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
                )
                target_norm = torch.mean(
                    torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
                )
                nmse = mse_val / target_norm
                psnr = 20 * torch.log10(
                    torch.tensor(
                        max_vals[fname],
                        dtype=mse_val.dtype,
                        device=mse_val.device,
                    )
                ) - 10 * torch.log10(mse_val)
                ssim = torch.mean(
                    torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
                )

                # Accumulate metric values
                metrics["nmse"] += nmse
                metrics["psnr"] += psnr
                metrics["ssim"] += ssim

                # Store individual metric values for std calculation
                metric_values["nmse"].append(nmse)
                metric_values["psnr"].append(psnr)
                metric_values["ssim"].append(ssim)

            # reduce across ddp via sum
            metrics["nmse"] = self.NMSE(metrics["nmse"])
            metrics["ssim"] = self.SSIM(metrics["ssim"])
            metrics["psnr"] = self.PSNR(metrics["psnr"])

            tot_examples = self.TotExamples(torch.tensor(local_examples))
            val_loss = self.ValLoss(torch.sum(torch.cat(losses)))  # type: ignore
            tot_slice_examples = self.TotSliceExamples(
                torch.tensor(len(losses), dtype=torch.float)
            )

            metrics_to_plot["nmse"].append(
                (
                    (metrics["nmse"] / tot_examples).item(),
                    torch.std(torch.stack(metric_values["nmse"])).item(),
                )
            )
            metrics_to_plot["psnr"].append(
                (
                    (metrics["psnr"] / tot_examples).item(),
                    torch.std(torch.stack(metric_values["psnr"])).item(),
                )
            )
            metrics_to_plot["ssim"].append(
                (
                    (metrics["ssim"] / tot_examples).item(),
                    torch.std(torch.stack(metric_values["ssim"])).item(),
                )
            )
            slugs.append(slug)

            # Log the mean values
            self.log(
                f"{slug}--validation_loss",
                val_loss / tot_slice_examples,
                prog_bar=True,
            )
            for metric, value in metrics.items():
                self.log(f"{slug}--val_metrics_{metric}", value / tot_examples)

            # Calculate and log the standard deviation for each metric
            for metric, values in metric_values.items():
                std_value = torch.std(torch.stack(values))
                self.log(f"{slug}--val_metrics_{metric}_std", std_value)

        # generate graph
        # breakpoint()
        for metric_name, values in metrics_to_plot.items():
            scores = [val[0] for val in values]
            std_devs = [val[1] for val in values]

            plt.figure(figsize=(10, 6))
            plt.bar(slugs, scores, yerr=std_devs, capsize=5)
            plt.xlabel("Dataset Slug")
            plt.ylabel(f"{metric_name.upper()} Score")
            plt.title(f"{metric_name.upper()} per Dataset with Standard Deviation")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            self.log_image(f"summary_plot_{metric_name}", image_array)
            buf.close()
            plt.close()

    def OLD_on_validation_epoch_end(self):
        val_logs = self.val_batch_results

        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname],
                        dtype=mse_val.dtype,
                        device=mse_val.device,
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics_{metric}", value / tot_examples)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser

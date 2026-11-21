import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity
from timm.utils import AverageMeter
from tqdm import tqdm

from .base_method import Base_method
from openstl.models.neurosd_model import ST_E3D_ODE_FNO
from openstl.utils import print_log
from openstl.utils.experiment_log import (
    append_csv,
    memory_now,
    replace_csv,
    reset_gpu_peak,
    sync_device,
    write_json,
)


def terminal_progress(iterable, desc):
    terminal = None

    try:
        try:
            terminal = open(
                "CONOUT$" if os.name == "nt" else "/dev/tty",
                "w",
                buffering=1,
                encoding="utf-8",
            )
        except OSError:
            terminal = None

        with tqdm(
            iterable,
            desc=desc,
            total=len(iterable),
            file=terminal,
            disable=terminal is None,
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        ) as progress:
            yield from progress
    finally:
        if terminal is not None:
            terminal.close()


class ForecastLoss(nn.Module):
    def __init__(self, lambda_ssim):
        super().__init__()

        if lambda_ssim < 0:
            raise ValueError("lambda_ssim must be nonnegative.")

        self.lambda_ssim = lambda_ssim

    def components(self, pred, true):
        mse = F.mse_loss(pred, true)

        if self.lambda_ssim > 0:
            batch_size, frames, channels, height, width = pred.shape

            score = ssim(
                pred.reshape(
                    batch_size * frames,
                    channels,
                    height,
                    width,
                ),
                true.reshape(
                    batch_size * frames,
                    channels,
                    height,
                    width,
                ),
                data_range=1.0,
                size_average=True,
            )

            total = mse + self.lambda_ssim * (1.0 - score)
        else:
            score = mse.new_tensor(float("nan"))
            total = mse

        return total, mse, score

    def forward(self, pred, true):
        return self.components(pred, true)[0]


class NeuroSD(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)

        if args.dist:
            raise NotImplementedError(
                "This implementation supports single-device experiments."
            )

        if args.fp16:
            raise ValueError(
                "This implementation requires FP32 training."
            )

        self.model = ST_E3D_ODE_FNO(
            in_channels=args.in_shape[1],
            hid_channels=args.hid_channels,
            out_frames=args.aft_seq_length,
            modes1=args.modes1,
            modes2=args.modes2,
            ode_method=args.ode_method,
            ode_time_end=args.ode_time_end,
        ).to(device)

        self.model_optim, self.scheduler, self.by_epoch = (
            self._init_optimizer(steps_per_epoch)
        )

        self.criterion = ForecastLoss(args.lambda_ssim)

        if args.selection_metric not in ("mse_pixel", "loss"):
            raise ValueError(
                "selection_metric must be 'mse_pixel' or 'loss'."
            )

        if args.log_every_updates <= 0:
            raise ValueError(
                "log_every_updates must be positive."
            )

        self.best_score = float("inf")
        self.last_train = {}

    def _predict(self, batch_x, batch_y=None, **kwargs):
        return self.model(batch_x)

    def train_one_epoch(
        self,
        runner,
        train_loader,
        epoch,
        num_updates,
        eta=None,
        **kwargs,
    ):
        self.model.train()

        if self.by_epoch:
            self.scheduler.step(epoch)

        meters = {
            key: AverageMeter()
            for key in ("loss", "mse", "ssim", "nfe")
        }

        lr_start = self.current_lr()[0]
        used_lrs = []
        last_grad_norm = None

        sync_device(self.device)
        reset_gpu_peak(self.device)
        start = time.perf_counter()

        for batch_index, (batch_x, batch_y) in enumerate(
            terminal_progress(
                train_loader,
                desc=f"Train {epoch + 1}/{self.args.epoch}",
            )
        ):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            runner.call_hook("before_train_iter")
            self.model_optim.zero_grad(set_to_none=True)

            used_lr = self.current_lr()[0]
            used_lrs.append(used_lr)

            pred = self._predict(batch_x)
            forward_nfe = float(self.model.nfe)

            if pred.shape != batch_y.shape:
                raise RuntimeError(
                    f"Prediction and target shapes differ: "
                    f"{pred.shape} versus {batch_y.shape}."
                )

            loss, mse, score = self.criterion.components(
                pred,
                batch_y,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Nonfinite loss at epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )

            loss.backward()

            if self.clip_mode == "norm" and self.clip_value is not None:
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_value,
                )
                last_grad_norm = float(norm)
            else:
                self.clip_grads(self.model.parameters())
                last_grad_norm = None

            self.model_optim.step()

            if not self.by_epoch:
                self.scheduler.step()

            batch_size = batch_x.size(0)
            finite_ssim = torch.isfinite(score).item()

            meters["loss"].update(loss.item(), batch_size)
            meters["mse"].update(mse.item(), batch_size)
            meters["nfe"].update(forward_nfe, batch_size)

            if finite_ssim:
                meters["ssim"].update(score.item(), batch_size)

            num_updates += 1
            runner._iter += 1
            runner.call_hook("after_train_iter")

            if (
                batch_index % self.args.log_every_updates == 0
                or batch_index + 1 == len(train_loader)
            ):
                append_csv(
                    os.path.join(runner.path, "train_updates.csv"),
                    {
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "global_update": num_updates,
                        "lr_used": used_lr,
                        "loss": loss.item(),
                        "mse_pixel": mse.item(),
                        "ssim_loss_impl": (
                            score.item() if finite_ssim else None
                        ),
                        "grad_norm_before_clip": last_grad_norm,
                        "forward_nfe": forward_nfe,
                    },
                )

        if not used_lrs:
            raise RuntimeError(
                "The training loader produced no batches."
            )

        sync_device(self.device)
        seconds = time.perf_counter() - start

        row = {
            "epoch": epoch + 1,
            "seed": self.args.seed,
            "global_update": num_updates,
            "train_samples_seen": int(meters["loss"].count),
            "lr_start": lr_start,
            "lr_used_min": min(used_lrs),
            "lr_used_max": max(used_lrs),
            "lr_next": self.current_lr()[0],
            "train_loss": meters["loss"].avg,
            "train_mse_pixel": meters["mse"].avg,
            "train_ssim_loss_impl": (
                meters["ssim"].avg
                if meters["ssim"].count
                else None
            ),
            "forward_nfe_mean": meters["nfe"].avg,
            "train_seconds": seconds,
            "train_sequences_per_second": (
                meters["loss"].count / seconds
            ),
            **memory_now(self.device),
        }

        self.last_train = row

        append_csv(
            os.path.join(runner.path, "train_epoch.csv"),
            row,
        )

        print_log(
            f"[Train] epoch={epoch + 1}/{self.args.epoch} "
            f"loss={row['train_loss']:.6f} "
            f"MSE_pixel={row['train_mse_pixel']:.6f} "
            f"LR_used=[{row['lr_used_min']:.3e},"
            f"{row['lr_used_max']:.3e}] "
            f"time={seconds:.2f}s "
            f"NFE={row['forward_nfe_mean']:.1f}"
        )

        return num_updates, meters["loss"], eta

    @torch.no_grad()
    def evaluate_stream(self, loader):
        self.model.eval()

        horizon = self.args.aft_seq_length
        sums = {
            key: np.zeros(horizon, dtype=np.float64)
            for key in ("mse_pixel", "mae_pixel", "ssim", "psnr")
        }

        total_samples = 0
        loss_sum = 0.0
        nfe_sum = 0.0
        pixel_count = None

        sync_device(self.device)
        reset_gpu_peak(self.device)
        start = time.perf_counter()

        for batch_x, batch_y in terminal_progress(
            loader,
            desc="Evaluating",
        ):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            pred = self._predict(batch_x)
            forward_nfe = float(self.model.nfe)

            if pred.shape != batch_y.shape:
                raise RuntimeError(
                    f"Prediction and target shapes differ: "
                    f"{pred.shape} versus {batch_y.shape}."
                )

            if pred.ndim != 5 or pred.shape[1] != horizon:
                raise RuntimeError(
                    "Expected predictions with shape [B,T,C,H,W] "
                    "and the configured prediction horizon."
                )

            if (
                not torch.isfinite(pred).all().item()
                or not torch.isfinite(batch_y).all().item()
            ):
                raise RuntimeError(
                    "Predictions or targets contain NaN or Inf."
                )

            current_pixel_count = int(np.prod(pred.shape[2:]))

            if pixel_count is None:
                pixel_count = current_pixel_count
            elif pixel_count != current_pixel_count:
                raise RuntimeError(
                    "Spatial dimensions or channel counts changed "
                    "during evaluation."
                )

            batch_size = batch_x.size(0)

            loss_sum += (
                self.criterion(pred, batch_y).item() * batch_size
            )
            nfe_sum += forward_nfe * batch_size

            prediction = pred.cpu().numpy().astype(np.float64)
            target = batch_y.cpu().numpy().astype(np.float64)

            difference = prediction - target

            sums["mse_pixel"] += (
                np.square(difference)
                .mean(axis=(2, 3, 4))
                .sum(axis=0)
            )

            sums["mae_pixel"] += (
                np.abs(difference)
                .mean(axis=(2, 3, 4))
                .sum(axis=0)
            )

            image_prediction = np.clip(prediction, 0.0, 1.0)

            image_mse = np.square(
                image_prediction - target
            ).mean(axis=(2, 3, 4))

            with np.errstate(divide="ignore"):
                sums["psnr"] += (
                    -10.0 * np.log10(image_mse)
                ).sum(axis=0)

            for b in range(batch_size):
                for t in range(horizon):
                    channel_scores = [
                        structural_similarity(
                            target[b, t, c],
                            image_prediction[b, t, c],
                            data_range=1.0,
                            win_size=7,
                        )
                        for c in range(target.shape[2])
                    ]
                    sums["ssim"][t] += np.mean(channel_scores)

            total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError(
                "The evaluation loader produced no samples."
            )

        sync_device(self.device)
        seconds = time.perf_counter() - start

        steps = []

        for t in range(horizon):
            row = {
                "step": t + 1,
                **{
                    key: float(values[t] / total_samples)
                    for key, values in sums.items()
                },
            }

            row["mse"] = row["mse_pixel"] * pixel_count
            row["mae"] = row["mae_pixel"] * pixel_count
            steps.append(row)

        summary = {
            key: float(np.mean([row[key] for row in steps]))
            for key in (
                "mse_pixel",
                "mae_pixel",
                "mse",
                "mae",
                "ssim",
                "psnr",
            )
        }

        summary.update(
            {
                "loss": loss_sum / total_samples,
                "num_samples": total_samples,
                "forward_nfe_mean": nfe_sum / total_samples,
                "evaluation_seconds_including_metrics": seconds,
                **memory_now(self.device),
            }
        )

        return summary, steps

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        summary, steps = self.evaluate_stream(vali_loader)

        epoch = runner._epoch + 1
        selection = self.args.selection_metric
        score = summary[selection]

        if not np.isfinite(score):
            raise RuntimeError(
                "The checkpoint selection metric is nonfinite."
            )

        is_best = score <= self.best_score

        if is_best:
            self.best_score = score

            write_json(
                os.path.join(runner.path, "best_record.json"),
                {
                    "best_epoch": epoch,
                    "selection_metric": selection,
                    "selection_mode": "min",
                    "best_score": score,
                    "seed": self.args.seed,
                    "metrics": summary,
                    "checkpoint_path": os.path.join(
                        runner.path,
                        "checkpoint.pth",
                    ),
                },
            )

            replace_csv(
                os.path.join(runner.path, "best_eval_per_step.csv"),
                steps,
            )

        row = {
            "epoch": epoch,
            "selection_metric": selection,
            "selection_score": score,
            "is_best": is_best,
            **summary,
        }

        append_csv(
            os.path.join(runner.path, "eval_epoch.csv"),
            row,
        )

        append_csv(
            os.path.join(runner.path, "epoch_summary.csv"),
            {
                **self.last_train,
                **{
                    f"eval_{key}": value
                    for key, value in row.items()
                    if key != "epoch"
                },
            },
        )

        log = (
            f"epoch={epoch} "
            f"MSE={summary['mse']:.6f} "
            f"MAE={summary['mae']:.6f} "
            f"SSIM={summary['ssim']:.6f} "
            f"PSNR={summary['psnr']:.4f} "
            f"selection={selection}:{score:.6f} "
            f"best={is_best}"
        )

        if runner.writer:
            for key in ("mse", "mae", "ssim", "psnr", "loss"):
                runner.writer.add_scalar(
                    f"evaluation/{key}",
                    summary[key],
                    epoch,
                )

            if "train_seconds" in self.last_train:
                runner.writer.add_scalar(
                    "runtime/train_seconds",
                    self.last_train["train_seconds"],
                    epoch,
                )

        results = {
            "mse": np.array([summary["mse"]]),
            "loss": np.array([score]),
        }

        return results, log

    def save_final_test(self, runner):
        summary, steps = self.evaluate_stream(
            runner.test_loader
        )

        shared = (
            runner.vali_loader.dataset
            is runner.test_loader.dataset
        )

        write_json(
            os.path.join(runner.path, "test_metrics.json"),
            {
                "seed": self.args.seed,
                "checkpoint": "checkpoint.pth",
                "validation_test_same_dataset_object": shared,
                "independent_final_test": False if shared else None,
                **summary,
            },
        )

        replace_csv(
            os.path.join(runner.path, "test_per_step.csv"),
            steps,
        )

        print_log(f"[Final evaluation] {summary}")

        return summary
import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as cal_ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NavierStokesDataset
from models import NeuroSD

try:
    from pytorch_msssim import ssim as ssim_loss_func
except ImportError:
    print("Warning: pytorch_msssim is not installed. The loss will fall back to MSE only.")
    ssim_loss_func = None


class cfg:
    data_path = "/root/autodl-tmp/data/NavierStockT30.mat"
    checkpoint_dir = "/root/autodl-tmp/t30/checkpoints_neurosd_10_10"

    t_in = 10
    t_out = 10

    in_channels = 1
    hid_channels = 128

    batch_size = 8
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-5
    lambda_s = 0.2

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()

    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred - true) ** 2, axis=(0, 1)).sum()

    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum()


def PSNR(pred, true, min_max_norm=True):
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32)) ** 2)

    if mse == 0:
        return float("inf")

    if min_max_norm:
        return 20.0 * np.log10(1.0 / np.sqrt(mse))

    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def composite_loss(preds, targets):
    mse = torch.nn.MSELoss()(preds, targets)

    if ssim_loss_func is not None:
        b, t, c, h, w = preds.shape

        ssim_val = ssim_loss_func(
            preds.view(-1, c, h, w),
            targets.view(-1, c, h, w),
            data_range=1.0,
            size_average=True
        )

        return mse + cfg.lambda_s * (1.0 - ssim_val)

    return mse


def evaluate_metrics(model, loader, device):
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    num_samples = 0
    num_frames = 0

    print("Running evaluation on the test set...")

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            pred = model(x).cpu().numpy()
            true = y.numpy()

            bs, ts, c, h, w = pred.shape

            num_samples += bs
            num_frames += bs * ts

            total_mse += MSE(pred, true) * bs
            total_mae += MAE(pred, true) * bs

            pred = np.clip(pred, 0, 1)

            for b in range(bs):
                for f in range(ts):
                    p_img = pred[b, f, 0]
                    t_img = true[b, f, 0]

                    ws = 7 if min(p_img.shape) >= 7 else (min(p_img.shape) // 2 * 2 + 1)

                    total_ssim += cal_ssim(
                        p_img,
                        t_img,
                        channel_axis=None,
                        win_size=ws,
                        data_range=1.0
                    )

                    total_psnr += PSNR(p_img, t_img)

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_ssim = total_ssim / num_frames
    avg_psnr = total_psnr / num_frames

    print(
        f"Evaluation Results | "
        f"MSE: {avg_mse:.6f} | "
        f"MAE: {avg_mae:.4f} | "
        f"SSIM: {avg_ssim:.4f} | "
        f"PSNR: {avg_psnr:.2f} dB"
    )

    return avg_mse


def train():
    set_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("NeuroSD Training on Navier-Stokes T30")
    print("=" * 60)
    print(f"Device        : {cfg.device}")
    print(f"Seed          : {cfg.seed}")
    print(f"Input frames  : {cfg.t_in}")
    print(f"Output frames : {cfg.t_out}")
    print(f"Hidden        : {cfg.hid_channels}")
    print(f"Batch size    : {cfg.batch_size}")
    print(f"Epochs        : {cfg.epochs}")
    print(f"Learning rate : {cfg.lr}")
    print(f"Scheduler     : CosineAnnealingLR")
    print(f"Lambda SSIM   : {cfg.lambda_s}")
    print(f"FNO modes     : (2, 3)")
    print("=" * 60)

    train_loader = DataLoader(
        NavierStokesDataset(
            cfg.data_path,
            mode="train",
            t_in=cfg.t_in,
            t_out=cfg.t_out
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        NavierStokesDataset(
            cfg.data_path,
            mode="test",
            t_in=cfg.t_in,
            t_out=cfg.t_out
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    model = NeuroSD(
        in_channels=cfg.in_channels,
        hid_channels=cfg.hid_channels,
        out_frames=cfg.t_out
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs
    )

    best_mse = float("inf")
    best_model_path = os.path.join(cfg.checkpoint_dir, "best_neurosd_10_10.pth")

    for ep in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            pred = model(x)
            loss = composite_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch [{ep + 1}/{cfg.epochs}]")
        print(f"Training Loss: {avg_loss:.6f}")

        current_mse = evaluate_metrics(model, test_loader, cfg.device)

        if current_mse < best_mse:
            best_mse = current_mse
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated: {best_model_path}")
            print(f"Best MSE: {best_mse:.6f}")
        else:
            print(f"No improvement. Current best MSE: {best_mse:.6f}")

    print("\nTraining completed successfully.")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best MSE: {best_mse:.6f}")


if __name__ == "__main__":
    train()
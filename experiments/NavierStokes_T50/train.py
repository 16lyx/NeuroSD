import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NavierStokesDataset
from models import NeuroSD
from metrics import metric as calc_official_metrics

try:
    from pytorch_msssim import ssim as ssim_loss_func
except ImportError:
    ssim_loss_func = None
    print("Warning: pytorch_msssim is not installed. SSIM loss will be skipped.")


class cfg:
    data_path = "/root/autodl-tmp/data/NavierStockT50.mat"
    checkpoint_dir = "/root/checkpoints_t50_neurosd"

    t_in = 10
    t_out = 10

    in_channels = 1
    hid_channels = 128
    modes1 = 2
    modes2 = 3

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


def compute_loss(preds, targets):
    mse_loss = F.mse_loss(preds, targets)

    if ssim_loss_func is not None:
        b, t, c, h, w = preds.shape

        ssim_val = ssim_loss_func(
            preds.view(-1, c, h, w),
            targets.view(-1, c, h, w),
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_val
        total_loss = mse_loss + cfg.lambda_s * ssim_loss

        return total_loss, mse_loss, ssim_val

    return mse_loss, mse_loss, torch.tensor(1.0, device=preds.device)


def evaluate_paper_metrics(model, loader, device):
    model.eval()

    all_preds = []
    all_trues = []

    print("Running evaluation...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing", disable=True):
            x = x.to(device)

            output = model(x).cpu().numpy()
            target = y.numpy()

            all_preds.append(output)
            all_trues.append(target)

    full_pred = np.concatenate(all_preds, axis=0)
    full_true = np.concatenate(all_trues, axis=0)

    eval_res, eval_log = calc_official_metrics(
        pred=full_pred,
        true=full_true,
        mean=None,
        std=None,
        metrics=["mae", "mse", "ssim", "psnr"],
        clip_range=[0, 1],
        spatial_norm=False,
        return_log=True
    )

    mae = eval_res.get("mae", 0.0)
    mse = eval_res.get("mse", 0.0)
    ssim = eval_res.get("ssim", 0.0)
    psnr = eval_res.get("psnr", 0.0)

    print(
        f"Evaluation Results | "
        f"MAE: {mae:.4f} | "
        f"MSE: {mse:.6f} | "
        f"SSIM: {ssim:.4f} | "
        f"PSNR: {psnr:.2f} dB"
    )

    return eval_res


def train():
    set_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("NeuroSD Training on Navier-Stokes T50")
    print("=" * 60)
    print(f"Input frames  : {cfg.t_in}")
    print(f"Output frames : {cfg.t_out}")
    print(f"Batch size    : {cfg.batch_size}")
    print(f"Learning rate : {cfg.lr}")
    print(f"FNO modes     : ({cfg.modes1}, {cfg.modes2})")
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
        out_frames=cfg.t_out,
        modes1=cfg.modes1,
        modes2=cfg.modes2
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
    best_model_path = os.path.join(cfg.checkpoint_dir, "best_neurosd_t50.pth")

    for ep in range(cfg.epochs):
        model.train()

        train_loss_accum = 0.0
        train_mse_accum = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {ep + 1}/{cfg.epochs}",
            disable=True
        )

        for x, y in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad()

            pred = model(x)
            loss, mse_val, ssim_val = compute_loss(pred, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            train_loss_accum += loss.item()
            train_mse_accum += mse_val.item()

            pbar.set_postfix(
                Loss=f"{loss.item():.5f}",
                MSE=f"{mse_val.item():.6f}",
                SSIM=f"{ssim_val.item():.4f}"
            )

        scheduler.step()

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_mse = train_mse_accum / len(train_loader)

        print(f"\nEpoch [{ep + 1}/{cfg.epochs}]")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Training MSE : {avg_train_mse:.6f}")
        print(f"LR           : {scheduler.get_last_lr()[0]:.2e}")

        eval_res = evaluate_paper_metrics(model, test_loader, cfg.device)

        current_mse = eval_res["mse"]

        if current_mse < best_mse:
            best_mse = current_mse

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(f"Best model updated: {best_model_path}")
            print(f"Best MSE: {best_mse:.6f}")
        else:
            print(f"No improvement. Current best MSE: {best_mse:.6f}")

    print("\nTraining completed successfully.")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best MSE: {best_mse:.6f}")


if __name__ == "__main__":
    train()
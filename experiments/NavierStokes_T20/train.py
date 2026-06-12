import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from pytorch_msssim import ssim as ssim_loss_func
except ImportError:
    print("Warning: pytorch_msssim is not installed. The loss will fall back to MSE only.")
    ssim_loss_func = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NavierStokesDataset
from models import NeuroSD
from metrics import metric as calc_official_metrics


class cfg:
    data_path = "/root/autodl-tmp/data/NavierStockT20.mat"
    checkpoint_dir = "/root/autodl-tmp/N_2/two_checkpoints_t20"
    batch_size = 8
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-5
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def composite_loss(preds, targets, lambda_s=0.2):
    mse_val = torch.nn.MSELoss()(preds, targets)

    if ssim_loss_func is not None:
        b, t, c, h, w = preds.shape
        ssim_val = ssim_loss_func(
            preds.view(-1, c, h, w),
            targets.view(-1, c, h, w),
            data_range=1.0,
            size_average=True
        )
        return mse_val + lambda_s * (1.0 - ssim_val)

    return mse_val


def evaluate_paper_metrics(model, loader, device):
    model.eval()

    all_preds = []
    all_trues = []

    print("\nRunning evaluation with MSE, MAE, SSIM, and PSNR...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False, disable=True):
            x = x.to(device)
            output = model(x).cpu().numpy()

            all_preds.append(output)
            all_trues.append(y.numpy())

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

    mse = eval_res["mse"]
    mae = eval_res["mae"]
    ssim = eval_res["ssim"]
    psnr = eval_res["psnr"]

    print("-" * 50)
    print("Test set evaluation results:")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"SSIM : {ssim:.4f}")
    print(f"PSNR : {psnr:.2f} dB")
    print("-" * 50)

    return eval_res


def train():
    set_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    train_loader = DataLoader(
        NavierStokesDataset(cfg.data_path, mode="train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        NavierStokesDataset(cfg.data_path, mode="test"),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4
    )

    model = NeuroSD(hid_channels=128).to(cfg.device)

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
    best_model_path = os.path.join(cfg.checkpoint_dir, "best_neurosd_t20.pth")

    print("=" * 50)
    print("NeuroSD Training on Navier-Stokes T20")
    print("=" * 50)
    print(f"Device          : {cfg.device}")
    print(f"Seed            : {cfg.seed}")
    print(f"Epochs          : {cfg.epochs}")
    print(f"Batch size      : {cfg.batch_size}")
    print(f"Learning rate   : {cfg.lr}")
    print(f"Checkpoint dir  : {cfg.checkpoint_dir}")
    print("=" * 50)

    for ep in range(cfg.epochs):
        model.train()
        train_loss = 0.0

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
            loss = composite_loss(pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        print(f"\nEpoch [{ep + 1}/{cfg.epochs}]")
        print(f"Training Loss: {avg_train_loss:.6f}")

        res = evaluate_paper_metrics(model, test_loader, cfg.device)
        current_mse = res["mse"]

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
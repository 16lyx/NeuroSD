import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_data
from models import NeuroSD
from metrics import metric

try:
    from pytorch_msssim import ssim as ssim_loss_func
except ImportError:
    ssim_loss_func = None


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def composite_loss(preds, targets, lambda_s=0.2):
    mse = nn.MSELoss()(preds, targets)

    if ssim_loss_func:
        b, t, c, h, w = preds.shape
        s_val = ssim_loss_func(
            preds.view(-1, c, h, w),
            targets.view(-1, c, h, w),
            data_range=1.0
        )
        return mse + lambda_s * (1 - s_val)

    return mse


def train():

    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 42,
        "epochs": 200,
        "batch_size": 32,
        "val_batch_size": 32,
        "lr": 1e-4,
        "data_root": "/root/autodl-tmp/data/",
        "save_dir": "checkpoints_taxibj/",
        "pre_seq_length": 4,
        "aft_seq_length": 4,
        "in_channels": 2,
        "hid_channels": 256,
        "eval_interval": 1
    }

    set_seed(config["seed"])

    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    print("=" * 40)
    print(f"{'NeuroSD TaxiBJ Training':^40}")
    print("=" * 40)
    for k, v in config.items():
        print(f"{k:15}: {v}")
    print("=" * 40)

    # --- Data loading ---
    print("Loading dataset...")
    train_loader, val_loader, _ = load_data(
        batch_size=config["batch_size"],
        val_batch_size=config["val_batch_size"],
        data_root=config["data_root"],
        num_workers=4,
        pre_seq_length=config["pre_seq_length"],
        aft_seq_length=config["aft_seq_length"],
        use_augment=True
    )

    # --- Model initialization ---
    model = NeuroSD(
        in_channels=config["in_channels"],
        hid_channels=config["hid_channels"],
        out_frames=config["aft_seq_length"]
    ).to(config["device"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader)
    )

    best_mse = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss_total = 0.0

        train_pbar = tqdm(train_loader, dynamic_ncols=True)
        train_pbar.set_description(f"Epoch [{epoch}/{config['epochs']}]")

        for data, target in train_pbar:
            data = data.to(config["device"])
            target = target.to(config["device"])

            optimizer.zero_grad()

            output = model(data)
            loss = composite_loss(output, target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item()

            train_pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                Lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg_train_loss = train_loss_total / len(train_loader)
        print(f"\nEpoch [{epoch}/{config['epochs']}] Training Loss: {avg_train_loss:.6f}")

        if epoch % config["eval_interval"] == 0:
            model.eval()

            eval_res_all = {
                "mse": [],
                "mae": [],
                "ssim": []
            }

            print(f"Evaluating epoch {epoch}...")
            val_pbar = tqdm(val_loader, desc="Evaluating", leave=False)

            with torch.no_grad():
                for data, target in val_pbar:
                    data = data.to(config["device"])
                    target = target.to(config["device"])

                    output = model(data)

                    res, _ = metric(
                        output.cpu().numpy(),
                        target.cpu().numpy(),
                        metrics=["mse", "mae", "ssim"]
                    )

                    for k in eval_res_all.keys():
                        eval_res_all[k].append(res[k])

            avg_mse = np.mean(eval_res_all["mse"])
            avg_mae = np.mean(eval_res_all["mae"])
            avg_ssim = np.mean(eval_res_all["ssim"])

            print(
                f"Validation Result | "
                f"MSE: {avg_mse:.6f} | "
                f"MAE: {avg_mae:.4f} | "
                f"SSIM: {avg_ssim:.4f}"
            )

            # Save only the best model
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_path = os.path.join(config["save_dir"], "best_neurosd.pth")
                torch.save(model.state_dict(), best_path)

                print(f"New best NeuroSD model saved to: {best_path}")
                print(f"Best MSE: {best_mse:.6f}")

            print("-" * 40)

    print("NeuroSD training completed successfully!")


if __name__ == "__main__":
    train()
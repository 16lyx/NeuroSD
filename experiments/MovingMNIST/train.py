import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from dataset import load_data
from metrics import metric
from models import NeuroSD

try:
    from pytorch_msssim import ssim as ssim_func
except ImportError:
    ssim_func = None
    print("Warning: pytorch_msssim is not installed. SSIM loss will be skipped.")


class CombinedLoss(nn.Module):
    def __init__(self, lambda_ssim=0.2):
        super(CombinedLoss, self).__init__()
        self.lambda_ssim = lambda_ssim

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)

        ssim_loss = torch.tensor(0.0, device=pred.device)
        ssim_val = torch.tensor(1.0, device=pred.device)

        if ssim_func is not None:
            b, t, c, h, w = pred.shape
            pred_flat = pred.reshape(-1, c, h, w)
            target_flat = target.reshape(-1, c, h, w)

            ssim_val = ssim_func(
                pred_flat,
                target_flat,
                data_range=1.0,
                size_average=True
            )

            ssim_loss = 1.0 - ssim_val

        total_loss = mse + self.lambda_ssim * ssim_loss

        return total_loss, mse, ssim_val


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)

            output = model(batch_x)

            preds.append(output.cpu().numpy())
            trues.append(batch_y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    res, _ = metric(
        preds,
        trues,
        metrics=["mse", "mae", "ssim", "psnr"]
    )

    return res


def train():
    config = {
        "dataset": "MovingMNIST",
        "hidden_dim": 256,
        "batch_size": 16,
        "epochs": 200,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "pre_seq_len": 10,
        "aft_seq_len": 10,
        "lambda_ssim": 0.2,
        "modes1": 2,
        "modes2": 3,
        "save_interval": 1,
        "seed": 42,
        "data_root": "/root/autodl-tmp/data/",
        "checkpoint_dir": "checkpoints_mmnist_neurosd"
    }

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("NeuroSD Training on MovingMNIST")
    print("=" * 50)
    print(f"Dataset       : {config['dataset']}")
    print(f"Device        : {device}")
    print(f"Hidden        : {config['hidden_dim']}")
    print(f"Batch size    : {config['batch_size']}")
    print(f"Epochs        : {config['epochs']}")
    print(f"Learning rate : {config['lr']}")
    print(f"Scheduler     : OneCycleLR")
    print(f"Lambda SSIM   : {config['lambda_ssim']}")
    print(f"FNO modes     : ({config['modes1']}, {config['modes2']})")
    print("=" * 50)

    train_loader, _, test_loader = load_data(
        batch_size=config["batch_size"],
        val_batch_size=config["batch_size"],
        data_root=config["data_root"],
        data_name="mnist",
        pre_seq_length=config["pre_seq_len"],
        aft_seq_length=config["aft_seq_len"],
        in_shape=[config["pre_seq_len"], 1, 64, 64],
        num_workers=4,
        use_augment=False,
        drop_last=True
    )

    model = NeuroSD(
        in_channels=1,
        hid_channels=config["hidden_dim"],
        out_frames=config["aft_seq_len"],
        modes1=config["modes1"],
        modes2=config["modes2"]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"]
    )

    criterion = CombinedLoss(lambda_ssim=config["lambda_ssim"])

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    best_mse = float("inf")
    best_model_path = os.path.join(
        config["checkpoint_dir"],
        "best_neurosd_mmnist.pth"
    )

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config['epochs']}",
            disable=True
        )

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_x)

            loss, mse_loss, ssim_val = criterion(output, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_epoch_loss += loss.item()

            pbar.set_postfix(
                Total=f"{loss.item():.4f}",
                MSE=f"{mse_loss.item():.5f}",
                SSIM=f"{ssim_val.item():.3f}"
            )

        avg_train_loss = total_epoch_loss / len(train_loader)

        print(f"\nEpoch [{epoch}/{config['epochs']}]")
        print(f"Training Loss: {avg_train_loss:.6f}")

        if epoch % config["save_interval"] == 0 or epoch == config["epochs"]:
            metrics_res = evaluate(model, test_loader, device)

            print(
                f"Evaluation Results | "
                f"MSE: {metrics_res['mse']:.4f} | "
                f"MAE: {metrics_res['mae']:.4f} | "
                f"SSIM: {metrics_res['ssim']:.4f} | "
                f"PSNR: {metrics_res['psnr']:.4f}"
            )

            current_mse = metrics_res["mse"]

            if current_mse < best_mse:
                best_mse = current_mse

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics_res
                    },
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
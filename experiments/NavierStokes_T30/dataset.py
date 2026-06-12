import torch
import numpy as np
import h5py
import scipy.io as sio
from torch.utils.data import Dataset


class NavierStokesDataset(Dataset):
    def __init__(self, file_path, mode="train", t_in=10, t_out=10, train_ratio=0.8):
        try:
            with h5py.File(file_path, "r") as f:
                data = np.array(f["u"], dtype=np.float32)

            if data.ndim == 4:
                data = data.transpose(3, 0, 1, 2)

        except OSError:
            mat_data = sio.loadmat(file_path)
            data = mat_data["u"].astype(np.float32)

            if data.ndim == 4:
                data = data.transpose(0, 3, 1, 2)

        self.max_val = data.max()
        self.min_val = data.min()

        data = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        total_samples = data.shape[0]
        split_idx = int(total_samples * train_ratio)

        if mode == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

        self.t_in = t_in
        self.t_out = t_out

        print(
            f"{mode} dataset loaded | "
            f"samples: {len(self.data)} | "
            f"setting: {self.t_in}->{self.t_out}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx]).unsqueeze(1)

        input_seq = seq[:self.t_in]
        target_seq = seq[self.t_in:self.t_in + self.t_out]

        return input_seq, target_seq
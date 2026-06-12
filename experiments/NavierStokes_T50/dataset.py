import torch
import numpy as np
import h5py
import scipy.io as sio
from torch.utils.data import Dataset


class NavierStokesDataset(Dataset):
    def __init__(self, file_path, mode="train", t_in=20, t_out=20):
        try:
            with h5py.File(file_path, "r") as f:
                data = np.array(f["u"], dtype=np.float32)
                if data.ndim == 4:
                    data = data.transpose(3, 0, 1, 2)

        except OSError:
            print("Reading the .mat file with scipy...")
            mat_data = sio.loadmat(file_path)
            data = mat_data["u"].astype(np.float32)

            if data.ndim == 4:
                data = data.transpose(3, 2, 0, 1)

        self.max_val = data.max()
        self.min_val = data.min()

        data = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        if mode == "train":
            self.data = data[:1600]
        else:
            self.data = data[1600:2000]

        self.t_in = t_in
        self.t_out = t_out

        print(
            f"{mode} dataset loaded (T50) | "
            f"samples: {len(self.data)} | "
            f"setting: {self.t_in}->{self.t_out}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx]).unsqueeze(1)
        return seq[:self.t_in], seq[self.t_in:self.t_in + self.t_out]
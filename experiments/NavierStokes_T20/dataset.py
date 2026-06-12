import torch
import numpy as np
import h5py
import scipy.io as sio
from torch.utils.data import Dataset


class NavierStokesDataset(Dataset):
    def __init__(self, file_path, mode='train', t_in=10, t_out=10):
        try:
            with h5py.File(file_path, 'r') as f:
                data = np.array(f['u'], dtype=np.float32)
                if data.ndim == 4:
                    # Adjust the data dimension to [N, T, H, W]
                    data = data.transpose(3, 2, 0, 1)

        except OSError:
            print(f"⚠️ Reading the .mat file with scipy...")
            mat_data = sio.loadmat(file_path)
            data = mat_data['u'].astype(np.float32)
            if data.ndim == 4:
                # Convert from [N, H, W, T] to [N, T, H, W]
                data = data.transpose(0, 3, 1, 2)

        # Record the physical value range for later denormalization in metrics.py
        self.max_val, self.min_val = data.max(), data.min()

        # Normalize the data to [0, 1] during training
        data = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        # Split the dataset into training and testing sets
        if mode == 'train':
            self.data = data[:960]
        else:
            self.data = data[960:1200]

        self.t_in, self.t_out = t_in, t_out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert from [T, H, W] to [T, 1, 64, 64]
        seq = torch.from_numpy(self.data[idx]).unsqueeze(1)
        return seq[:self.t_in], seq[self.t_in: self.t_in + self.t_out]
dataset_parameters = {
    "mmnist": {
        "in_shape": [10, 1, 64, 64],
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "total_length": 20,
        "data_name": "mnist",
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "taxibj": {
        "in_shape": [4, 2, 32, 32],
        "pre_seq_length": 4,
        "aft_seq_length": 4,
        "total_length": 8,
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "navier_stokes_t20": {
        "in_shape": [10, 1, 64, 64],
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "total_length": 20,
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "navier_stokes_t30": {
        "in_shape": [10, 1, 64, 64],
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "total_length": 20,
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "navier_stokes_t50": {
        "in_shape": [10, 1, 64, 64],
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "total_length": 20,
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
}
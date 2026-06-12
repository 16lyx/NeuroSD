# Hyperparameter Settings

The detailed hyperparameter settings used for different datasets are summarized below.

| Dataset | Input size | Hidden | Batch size | Epochs | LR | Scheduler | λ | FNO modes | Optimizer |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TaxiBJ | 4 × 2 × 32 × 32 | 256 | 32 | 200 | 1e-4 | OneCycleLR | 0.2 | (3, 3) | AdamW |
| MovingMNIST | 10 × 1 × 64 × 64 | 256 | 16 | 200 | 1e-3 | OneCycleLR | 0.2 | (2, 3) | AdamW |
| Navier-Stokes T20 | 10 × 1 × 64 × 64 | 256 | 8 | 50 | 1e-4 | Cosine | 0.2 | (2, 3) | AdamW |
| Navier-Stokes T30 | 10 × 1 × 64 × 64 | 128 | 8 | 50 | 1e-4 | Cosine | 0.2 | (2, 3) | AdamW |
| Navier-Stokes T50 | 10 × 1 × 64 × 64 | 128 | 8 | 50 | 1e-4 | Cosine | 0.2 | (2, 3) | AdamW |
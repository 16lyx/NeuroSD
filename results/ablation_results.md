# Ablation Results

This file reports the ablation study of NeuroSD on the Navier-Stokes T20 dataset.

Lower MSE and MAE indicate better numerical prediction accuracy, while higher SSIM and PSNR indicate better structural preservation.

---

## Navier-Stokes T20

Ablation results of NeuroSD on the Navier-Stokes T20 dataset are shown below. The table reports the performance of the model after removing DyMem, the SDO Block, and Neural ODE in terms of MSE, MAE, SSIM, and PSNR.

| Modules | MSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ |
|:---|---:|---:|---:|---:|
| NeuroSD w/o DyMem | 7.90 | 111.39 | 0.8712 | 30.08 |
| NeuroSD w/o SDO Block | 9.22 | 119.89 | 0.8627 | 29.33 |
| NeuroSD w/o ODE | 51.44 | 352.72 | 0.6321 | 20.09 |
| NeuroSD | **6.81** | **101.57** | **0.8892** | **30.81** |

The results indicate that all modules have a significant impact on model performance. Removing the Neural ODE module causes the most obvious degradation, especially in MSE, MAE, SSIM, and PSNR, demonstrating its importance for long-term prediction stability. The SDO Block and DyMem also contribute to improving prediction accuracy and preserving spatial structures.

---

## Visualization

The corresponding ablation visualization is provided in the `figures/ablation/` directory.

| Figure | Description |
|:---|:---|
| `figures/ablation/ablation_metrics_t20_t30.png` | Quantitative ablation comparison on Navier-Stokes T20 and T30 |
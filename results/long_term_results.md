# Long-term Prediction Results

This file reports the long-term prediction results of NeuroSD on the Navier-Stokes datasets under different prediction time steps.

Lower MSE and MAE indicate better numerical prediction accuracy, while higher SSIM and PSNR indicate better structural preservation.

---

## Navier-Stokes T50

Performance results of NeuroSD under different prediction steps on the Navier-Stokes T50 dataset are shown below.

| Time Steps | MSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ |
|:---:|---:|---:|---:|---:|
| 4 | 0.0008 | 1.02 | 0.9999 | 68.61 |
| 10 | 0.0033 | 2.18 | 0.9999 | 63.52 |
| 20 | 0.0286 | 6.90 | 0.9996 | 52.22 |
| 25 | 0.0541 | 9.84 | 0.9994 | 49.09 |

The results show that NeuroSD maintains stable prediction performance on the Navier-Stokes T50 dataset under different prediction horizons. Although the prediction error increases as the number of predicted time steps becomes larger, the SSIM remains consistently high, indicating strong structural preservation in long-term prediction.

---

## Navier-Stokes T30

Supplementary performance results on the Navier-Stokes T30 dataset under different prediction time steps are shown below.

| Time Steps | MSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ |
|:---:|---:|---:|---:|---:|
| 4 | 0.0042 | 2.09 | 0.9998 | 61.62 |
| 10 | 1.1277 | 44.87 | 0.9620 | 37.25 |
| 20 | 5.9186 | 112.61 | 0.8420 | 29.55 |
| 25 | 13.3499 | 169.76 | 0.7633 | 26.22 |

The supplementary results on Navier-Stokes T30 further show that the prediction difficulty increases with longer prediction horizons. Compared with short-term prediction, long-term prediction introduces larger accumulated errors and more obvious structural degradation.

---

## Visualization

The corresponding long-term visualizations are provided in the `figures/long_term/` directory.

| Figure | Description |
|:---|:---|
| `figures/long_term/ns_t50_different_horizons.png` | Visualization of NeuroSD on Navier-Stokes T50 under 4, 10, 20, and 25 prediction steps |
| `figures/long_term/ns_t50_model_comparison.png` | Qualitative comparison between NeuroSD and baseline models on Navier-Stokes T50 |
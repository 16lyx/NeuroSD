method = "neurosd"
dataname = "navier_stokes_t30"

in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

ns_data_file = "data/Navier-StokesT30/NavierStockT30.mat"
ns_axis_order = "THWN"

hid_channels = 128
modes1 = 3
modes2 = 5

ode_method = "rk4"
ode_time_end = 1.0
encoder_shape_probe = False

epoch = 50
batch_size = 8
val_batch_size = 8

opt = "adamw"
lr = 1e-4
weight_decay = 1e-5
filter_bias_and_bn = False

sched = "torch_cosine"
min_lr = 0.0
warmup_epoch = 0

clip_grad = 5.0
clip_mode = "norm"

lambda_ssim = 0.2
selection_metric = "mse_pixel"

use_augment = False
use_prefetcher = False
drop_last = False
num_workers = 4
fp16 = False

seed = 42
deterministic = True

log_step = 1
log_every_updates = 50

profile_warmup = 10
profile_repeats = 30
no_display_method_info = True

metric_clip_prediction = True

research_notes = {
    "experiment_purpose": "Navier-Stokes T30, 10-to-10",
    "baseline_or_proposed": "proposed",
    "ablation": "full_model",
    "data_variable": "u",
    "source_axis_order": "THWN",
    "training_trajectories": [0, 8000],
    "evaluation_trajectories": [8000, 10000],
    "index_convention": "zero-based, start-inclusive, stop-exclusive",
    "validation_reuses_test": True,
    "input_frame_indices": [0, 10],
    "target_frame_indices": [10, 20],
    "window_start": 0,
    "window_count_per_trajectory": 1,
    "normalization": (
        "Min-max statistics computed from all stored frames "
        "of training trajectories only."
    ),
    "normalization_fit_trajectories": [0, 8000],
    "normalization_application": (
        "Apply the same training-set statistics to training "
        "and test trajectories."
    ),
    "normalization_epsilon": 1e-8,
    "normalization_change_from_scattered_code": (
        "Replace whole-file statistics with training-only statistics."
    ),
    "scheduler": "PyTorch CosineAnnealingLR",
    "gradient_clipping": 5.0,
    "augmentation": False,
    "checkpoint_selection": "minimum evaluation elementwise MSE",
    "evaluation_interval_epochs": 1,
    "ode_time_definition": (
        "Initial time 0; future nodes k/T_out for k=1,...,T_out."
    ),
    "ode_step_size": 0.1,
    "expected_forward_nfe": 40,
    "metric_scale": "normalized vorticity",
    "mse_mae_prediction_clipping": "none",
    "mse_mae_reduction": (
        "Sum over channels and spatial dimensions, "
        "then average over samples and prediction steps."
    ),
    "ssim_psnr_prediction_clipping": "[0,1]",
    "ssim_psnr_data_range": 1.0,
    "physics_metric_scale": (
        "Original vorticity scale where specified in the manuscript."
    ),
}
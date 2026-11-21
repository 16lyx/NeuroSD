method = "neurosd"
dataname = "navier_stokes_t50"


ns_data_file = (
    "data/Navier-StokesT50/NavierStockT50.mat"
)
ns_axis_order = "THWN"
ns_normalization_frames = 20

in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10
total_length = pre_seq_length + aft_seq_length

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

metrics = ["mse", "mae", "ssim", "psnr"]

log_step = 1
log_every_updates = 50

profile_warmup = 10
profile_repeats = 30
no_display_method_info = True

metric_clip_prediction = True

research_notes = {
    "experiment_purpose": "Navier-Stokes T50, 10-to-10",
    "baseline_or_proposed": "proposed",
    "ablation": "full_model",
    "data_variable": "u",
    "source_axis_order": "THWN",
    "index_convention": "zero-based, start-inclusive, stop-exclusive",
    "training_trajectories": [0, 1600],
    "validation_trajectories": [2000, 2200],
    "test_trajectories": [1600, 2000],
    "validation_reuses_test": False,
    "input_frame_indices": [0, pre_seq_length],
    "target_frame_indices": [pre_seq_length, total_length],
    "window_start": 0,
    "window_count_per_trajectory": 1,
    "normalization": (
        "Min-max statistics fitted on training trajectories "
        "[0:1600] and frames [0:20], then shared by "
        "training, validation and test sets."
    ),
    "normalization_fit_trajectories": [0, 1600],
    "normalization_fit_frames": [0, ns_normalization_frames],
    "normalization_change_from_scattered_code": (
        "Training-only statistics instead of whole-file statistics."
    ),
    "training_epochs": epoch,
    "training_loss": "MSE + 0.2 * (1 - SSIM)",
    "scheduler": "PyTorch CosineAnnealingLR",
    "learning_rate_warmup_epochs": warmup_epoch,
    "gradient_clipping": clip_grad,
    "augmentation": use_augment,
    "checkpoint_selection": (
        "Minimum independent validation elementwise MSE."
    ),
    "evaluation_interval_epochs": 1,
    "ode_time_definition": (
        "Initial time 0; future nodes k/T_out for k=1,...,T_out."
    ),
    "metric_scale": "training-minmax-normalized vorticity",
    "mse_mae_prediction_clipping": "none",
    "mse_mae_reduction": (
        "Sum over channels and spatial dimensions, "
        "then average over samples and prediction steps."
    ),
    "ssim_psnr_prediction_clipping": "[0,1]",
    "ssim_psnr_data_range": 1.0,
}
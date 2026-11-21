method = "neurosd"
dataname = "navier_stokes_t20"

in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

ns_data_file = (
    "data/Navier-StokesT20/NavierStockT20.mat"
)
ns_axis_order = None

hid_channels = 256
modes1 = 2
modes2 = 3

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

clip_grad = None
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
metric_clip_prediction = True

log_step = 1
log_every_updates = 50

profile_warmup = 10
profile_repeats = 30
no_display_method_info = True

research_notes = {
    "experiment_purpose": "Navier-Stokes T20 main experiment, 10-to-10",
    "baseline_or_proposed": "proposed",
    "ablation": "full_model",
    "data_variable": "u",
    "physical_variable_meaning": "vorticity",
    "simulation_viscosity": 1e-5,
    "simulation_forcing": None,
    "physical_sampling_interval": None,
    "source_axis_order": None,
    "training_trajectories": [0, 960],
    "evaluation_trajectories": [960, 1200],
    "index_convention": "zero-based, start-inclusive, stop-exclusive",
    "input_frame_indices": [0, 10],
    "target_frame_indices": [10, 20],
    "window_start": 0,
    "window_count_per_trajectory": 1,
    "normalization": (
        "Min-max statistics fitted on training trajectories "
        "and shared by training and test data."
    ),
    "normalization_change_from_scattered_code": (
        "Training-only statistics instead of whole-file statistics."
    ),
    "training_epochs": epoch,
    "training_loss": "MSE + 0.2 * (1 - SSIM)",
    "scheduler": "PyTorch CosineAnnealingLR",
    "learning_rate_warmup_epochs": warmup_epoch,
    "gradient_clipping": False,
    "augmentation": use_augment,
    "checkpoint_selection": "minimum evaluation elementwise MSE",
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
    "physics_metric_scale": (
        "Original vorticity scale where specified in the manuscript."
    ),
    "hyperparameter_search_description": None,
}
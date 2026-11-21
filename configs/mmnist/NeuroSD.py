method = "neurosd"
dataname = "mmnist"
data_name = "mnist"

in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

hid_channels = 128
modes1 = 2
modes2 = 3

ode_method = "rk4"
ode_time_end = 1.0

epoch = 200
batch_size = 16
val_batch_size = 16

opt = "adamw"
lr = 1e-3
weight_decay = 0.01
filter_bias_and_bn = False

sched = "onecycle"
final_div_factor = 1e4

clip_grad = 1.0
clip_mode = "norm"

lambda_ssim = 0.2
selection_metric = "mse_pixel"

seed = 42
deterministic = True

use_augment = False
use_prefetcher = False
fp16 = False
drop_last = False
num_workers = 4

metrics = ["mse", "mae", "ssim", "psnr"]
metric_clip_prediction = True

log_step = 1
log_every_updates = 50

profile_warmup = 10
profile_repeats = 30
no_display_method_info = True

research_notes = {
    "experiment_purpose": "MovingMNIST main experiment, 10-to-10",
    "baseline_or_proposed": "proposed",
    "ablation": "full_model",
    "hyperparameter_search_description": None,
    "physical_sampling_interval": None,
    "ode_time_definition": (
        "Initial time 0; future nodes k/T_out for k=1,...,T_out."
    ),
    "training_data": "online-generated MovingMNIST",
    "evaluation_data": "fixed mnist_test_seq.npy",
    "normalization": "pixel values divided by 255",
    "metric_scale": "normalized pixel values",
    "mse_mae_prediction_clipping": "none",
    "mse_mae_reduction": (
        "Sum over channels and spatial dimensions, "
        "then average over samples and prediction steps."
    ),
    "ssim_psnr_prediction_clipping": "[0,1]",
    "ssim_psnr_data_range": 1.0,
    "checkpoint_selection": "minimum evaluation elementwise MSE",
    "augmentation": use_augment,
    "baseline_implementation_version": None,
    "changes_from_reference_implementation": None,
}
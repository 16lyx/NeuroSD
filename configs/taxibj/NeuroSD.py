method = "neurosd"
dataname = "taxibj"



hid_channels = 256
modes1 = 3
modes2 = 3
in_shape = [4, 2, 32, 32]
pre_seq_length = 4
aft_seq_length = 4
total_length = 8
lambda_ssim = 0.2


ode_method = "rk4"
ode_time_end = 1.0
encoder_shape_probe = False

epoch = 200
batch_size = 32
val_batch_size = 32

opt = "adamw"
lr = 1e-4
weight_decay = 1e-4
filter_bias_and_bn = False

sched = "onecycle"
final_div_factor = 1e4

clip_grad = 5.0
clip_mode = "norm"

lambda_ssim = 0.2
selection_metric = "mse_pixel"

use_augment = False
use_prefetcher = False
drop_last = False

num_workers = 4
seed = 42
deterministic = True
fp16 = False

metrics = ["mse", "mae", "ssim", "psnr"]
metric_clip_prediction = True

log_step = 1
log_every_updates = 50

profile_warmup = 10
profile_repeats = 30
no_display_method_info = True

research_notes = {
    "experiment_purpose": "TaxiBJ main experiment, 4-to-4",
    "baseline_or_proposed": "proposed",
    "ablation": "full_model",
    "training_data": "dataset.npz: X_train, Y_train",
    "evaluation_data": "dataset.npz: X_test, Y_test",
    "training_sequences": 20461,
    "test_sequences": 500,
    "validation_reuses_test": True,
    "normalization": "(stored_value + 1) / 2",
    "metric_scale": (
        "Linearly normalized traffic flow values, "
        "without inverse transformation to traffic counts."
    ),
    "augmentation": use_augment,
    "training_drop_last": True,
    "evaluation_drop_last": drop_last,
    "training_epochs": epoch,
    "training_loss": "MSE + 0.2 * (1 - SSIM)",
    "scheduler": "OneCycleLR",
    "gradient_clipping": clip_grad,
    "evaluation_interval_epochs": 1,
    "checkpoint_selection": "minimum evaluation elementwise MSE",
    "ode_time_definition": (
        "Initial time 0; future nodes k/T_out for k=1,...,T_out."
    ),
    "physical_sampling_interval": None,
    "mse_mae_prediction_clipping": "none",
    "mse_mae_reduction": (
        "Sum over channels and spatial dimensions, "
        "then average over samples and prediction steps."
    ),
    "ssim_psnr_prediction_clipping": "[0,1]",
    "ssim_psnr_data_range": 1.0,
    "hyperparameter_search_description": None,
    "evaluation_change_from_scattered_code": (
        "Sample-weighted aggregation instead of equal batch averaging."
    ),
}
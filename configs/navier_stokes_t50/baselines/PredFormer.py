# =========================
# 方法与数据
# =========================
method = 'PredFormer'
dataname = 'navier_stokes_t50'

data_root = '/root/autodl-tmp/修_A/main/data'
ns_data_file = (
    '/root/autodl-tmp/修_A/main/'
    'data/Navier-StokesT50/NavierStockT50.mat'
)
ns_axis_order = 'THWN'

in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# =========================
# 官方 PredFormer-TSST 网络配置
# =========================
model_config = {
    'height': 64,
    'width': 64,
    'num_channels': 1,

    'pre_seq': 10,
    'after_seq': 10,

    'patch_size': 8,
    'dim': 256,
    'heads': 8,
    'dim_head': 32,

    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,

    'scale_dim': 4,
    'depth': 1,
    'Ndepth': 6,
}

# =========================
# NS-50 训练设置
# =========================
epoch = 50
batch_size = 8
val_batch_size = 8

opt = 'adamw'
lr = 3e-4
weight_decay = 1e-2
filter_bias_and_bn = False

sched = 'onecycle'
final_div_factor = 1e4
warmup_epoch = 0

clip_grad = 1.0
clip_mode = 'norm'

# 官方 PredFormer 方法直接使用 nn.MSELoss。
# 不添加 NeuroSD 的 SSIM 复合损失。

# =========================
# 数据加载与运行
# =========================
use_augment = False
use_prefetcher = False
drop_last = False
num_workers = 4

seed = 42
deterministic = True
fp16 = False

log_step = 1
no_display_method_info = True
metrics = ['mse', 'mae', 'ssim', 'psnr']

# 若当前公共训练入口支持该参数，则按像素 MSE 选模。
selection_metric = 'mse_pixel'

# =========================
# 实验记录
# =========================
research_notes = {
    'experiment_purpose': 'NS-50 baseline comparison',
    'baseline_or_proposed': 'baseline',
    'model': 'PredFormer-TSST',
    'source_repo': 'https://github.com/yyyujintang/PredFormer',

    'training_loss': 'MSE',
    'prediction': 'direct 10-to-10',

    'train_trajectory_slice': [0, 1600],
    'val_trajectory_slice': [2000, 2200],
    'test_trajectory_slice': [1600, 2000],

    'normalization_fit_scope': (
        'training trajectories, first 20 frames'
    ),
    'checkpoint_selection': 'minimum validation MSE',

    'training_budget_epochs': 50,
    'hyperparameters_status': (
        'official TSST architecture; '
        'NS-50 starting training configuration, not a verified optimum'
    ),
}

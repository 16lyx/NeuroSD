method = 'ConvLSTM'
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 4
layer_norm = 0
# training
lr = 5e-4
batch_size = 16
sched = 'onecycle'


# =========================
# NS-30：沿用当前数据处理
# =========================
dataname = 'navier_stokes_t30'

data_root = '/root/autodl-tmp/修_A/main/data'
ns_data_file = (
    '/root/autodl-tmp/修_A/main/'
    'data/Navier-StokesT30/NavierStockT30.mat'
)
ns_axis_order = 'THWN'

in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

epoch = 50
batch_size = 8
val_batch_size = 8

use_augment = False
use_prefetcher = False
drop_last = False
num_workers = 4
fp16 = False

seed = 42
deterministic = True

# 保留官方配置中的学习率、网络结构和采样参数。
# 这里显式指定优化器基础设置。
opt = 'adam'
weight_decay = 0.0
filter_bias_and_bn = False
clip_grad = None
clip_mode = 'norm'
warmup_epoch = 0

# 供后续接入现有日志和公共评估使用。
selection_metric = 'mse_pixel'
metric_clip_prediction = True

log_step = 1
log_every_updates = 50
no_display_method_info = True

profile_warmup = 10
profile_repeats = 30

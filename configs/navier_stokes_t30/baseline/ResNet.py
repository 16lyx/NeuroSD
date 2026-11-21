method = 'resnet'

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
grad_accum_steps = 1

opt = 'adam'
lr = 1e-3
weight_decay = 0.0
filter_bias_and_bn = False
sched = 'onecycle'
final_div_factor = 1e4
warmup_epoch = 0
min_lr = 0.0

clip_grad = 1.0
clip_mode = 'norm'
lambda_ssim = 0.0
selection_metric = 'mse_pixel'
metric_clip_prediction = True

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

unet_channels = 64

vit_patch_size = 16
vit_dim = 768
vit_depth = 12
vit_heads = 12
vit_mlp_dim = 3072
vit_dropout = 0.0
vit_attention_dropout = 0.0

cno_layers = 3
cno_res = 1
cno_neck = 6
cno_channels = 32
cno_filter_impl = 'ref'

fno_modes = 12
fno_width = 20

# MMVP 官方 MMNIST 网络结构，迁移到 NS-30。
downsample_setting = '2,2,2'
hid_S = 32
hid_T = 192
rrdb_encoder_num = 2
rrdb_decoder_num = 2
rrdb_enhance_num = 2
use_direct_predictor = True

batch_size = 8
val_batch_size = 8
research_notes = {'experiment_purpose': 'NS30 ResNet, 10-to-10', 'baseline_or_proposed': 'baseline', 'training_loss': 'full future pixel MSE; no SSIM term', 'initialization': 'random; trained on NS30', 'training_trajectories': [0, 8000], 'evaluation_trajectories': [8000, 10000], 'validation_reuses_test': True, 'normalization': 'whole-file minmax', 'training_recipe': 'controlled NS30 adaptation; not a published NS30 optimal configuration', 'effective_batch_size': 8, 'gradient_accumulation': False}

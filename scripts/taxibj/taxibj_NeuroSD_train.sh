#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/修_A/main"
cd "$PROJECT_ROOT"

SEED=42
BATCH_SIZE=32
EPOCHS=200
LR=1e-4
CONFIG_FILE="configs/taxibj/NeuroSD.py"

export PYTHONHASHSEED="$SEED"

python - "$CONFIG_FILE" <<'PY'
import sys
from openstl.utils import load_config

config = load_config(sys.argv[1])

expected = {
    'hid_channels': 256,
    'modes1': 3,
    'modes2': 3,
    'pre_seq_length': 4,
    'aft_seq_length': 4,
    'lambda_ssim': 0.2,
}

for key, value in expected.items():
    actual = config.get(key)
    if actual != value:
        raise ValueError(
            f'Configuration mismatch: expected {key}={value}, '
            f'got {actual}.'
        )

if list(config.get('in_shape', [])) != [4, 2, 32, 32]:
    raise ValueError('Expected in_shape=[4,2,32,32].')

print('TaxiBJ main-experiment configuration verified.')
PY

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M-%S")

EX_NAME="taxibj/${CURRENT_TIME}_NeuroSD_h256_modes3x3_bs${BATCH_SIZE}_lr${LR}_wd1e-4_clip5_AdamW_OneCycle_${EPOCHS}ep_seed${SEED}"

RES_DIR="${PROJECT_ROOT}/work_dirs"
RUN_DIR="${RES_DIR}/${EX_NAME}"

mkdir -p "$RUN_DIR"
cp -- "${BASH_SOURCE[0]}" "${RUN_DIR}/launch.sh"

python -u tools/train.py \
    --config_file "$CONFIG_FILE" \
    --method neurosd \
    --dataname taxibj \
    --data_root "${PROJECT_ROOT}/data" \
    --res_dir "$RES_DIR" \
    --ex_name "$EX_NAME" \
    --device cuda \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$BATCH_SIZE" \
    --epoch "$EPOCHS" \
    --lr "$LR" \
    --opt adamw \
    --weight_decay 1e-4 \
    --sched onecycle \
    --clip_grad 5.0 \
    --clip_mode norm \
    --seed "$SEED" \
    --deterministic \
    --use_augment \
    --log_step 1 \
    --num_workers 4 \
    2>&1 | tee "${RUN_DIR}/console.log"
#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/修_A/main"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

SEED=42
BATCH_SIZE=8
EPOCHS=50
LR=1e-4

export PYTHONHASHSEED="$SEED"

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M-%S")

EX_NAME="navier_stokes_t30/${CURRENT_TIME}_NeuroSD_h128_modes3x5_lambda0.2_10to10_bs${BATCH_SIZE}_lr${LR}_AdamW_cosine_${EPOCHS}ep_seed${SEED}"

CONFIG_FILE="configs/navier_stokes_t30/NeuroSD.py"

RES_DIR="${PROJECT_ROOT}/work_dirs"
RUN_DIR="${RES_DIR}/${EX_NAME}"

mkdir -p "$RUN_DIR"

cp -- "${BASH_SOURCE[0]}" "${RUN_DIR}/launch.sh"
cp -- "$CONFIG_FILE" "${RUN_DIR}/config_used.py"

python -u tools/train.py \
    --config_file "$CONFIG_FILE" \
    --method neurosd \
    --dataname navier_stokes_t30 \
    --data_root "${PROJECT_ROOT}/data" \
    --res_dir "$RES_DIR" \
    --ex_name "$EX_NAME" \
    --device cuda \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$BATCH_SIZE" \
    --pre_seq_length 10 \
    --aft_seq_length 10 \
    --total_length 20 \
    --epoch "$EPOCHS" \
    --lr "$LR" \
    --opt adamw \
    --weight_decay 1e-5 \
    --sched torch_cosine \
    --min_lr 0 \
    --warmup_epoch 0 \
    --clip_grad 5.0 \
    --clip_mode norm \
    --seed "$SEED" \
    --deterministic \
    --log_step 1 \
    --num_workers 4 \
    2>&1 | tee "${RUN_DIR}/console.log"
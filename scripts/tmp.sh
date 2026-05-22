#!/bin/sh
set -e

# 设置环境变量
export OMP_NUM_THREADS=1

# 指定用于训练的 GPU，可以写成 "cuda:0 cuda:1 cuda:2" 或 "0 1 2"
GPUS="cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6"

CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
for GPU in $GPUS; do
    GPU_ID=${GPU#cuda:}
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID
    else
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,$GPU_ID
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done

export CUDA_VISIBLE_DEVICES

# 执行 Python 程序
torchrun --nproc_per_node="${NUM_GPUS}" main_pretrain.py \
    --accum_iter 2 \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --epochs 400 \
    --warmup_epochs 20 \
    --mask_ratio 0.75 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --vis_freq 50 \
    --vis_num_images 8 \
    --norm_pix_loss \
    --data_path ".datasets/intraoral" \
    --output_dir "exp/pretrain_scratch" \
    --log_dir "exp/pretrain_scratch"

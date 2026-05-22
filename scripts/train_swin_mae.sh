#!/bin/sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Set physical GPU ids here. Both "cuda:1 cuda:2" and "1 2" are supported.
GPUS=${GPUS:-"cuda:3 cuda:4 cuda:5 cuda:6"}
CONFIG=${CONFIG:-"config.yaml"}

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

echo "Using physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"

torchrun --nproc_per_node="${NUM_GPUS}" \
    train.py \
    --config "${CONFIG}" \
    "$@"

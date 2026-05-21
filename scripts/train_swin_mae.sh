#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_swin_mae.sh <gpus> [train.py args...]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/train_swin_mae.sh cuda:1 cuda:2 --batch_size 32 --epochs 400"
  echo "  bash scripts/train_swin_mae.sh 1,2 --batch_size 32 --data_path .datasets/intraoral"
  echo "  bash scripts/train_swin_mae.sh 0 --batch_size 16 --epochs 1"
  exit 1
fi

gpu_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    cuda:*|[0-9]*)
      gpu_args+=("$1")
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ ${#gpu_args[@]} -eq 0 ]]; then
  echo "No GPU id was provided."
  exit 1
fi

gpu_list="$(
  IFS=,
  echo "${gpu_args[*]}"
)"
gpu_list="${gpu_list//cuda:/}"
gpu_list="${gpu_list// /,}"
gpu_list="${gpu_list//,,/,}"
gpu_list="${gpu_list#,}"
gpu_list="${gpu_list%,}"

num_gpus=$(awk -F',' '{print NF}' <<< "${gpu_list}")
master_port="${MASTER_PORT:-29500}"

echo "Using physical GPUs: ${gpu_list}"
echo "DDP processes: ${num_gpus}"

CUDA_VISIBLE_DEVICES="${gpu_list}" torchrun \
  --nproc_per_node="${num_gpus}" \
  --master_port="${master_port}" \
  train.py "$@"

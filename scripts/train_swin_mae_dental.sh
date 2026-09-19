#!/bin/sh
set -e

# Swin-MAE 牙科 SSL 预训练（病例级 split，§9.5.4）
# 用法（在 baselines/Swin-MAE 下执行；从任意目录运行也可，路径自动解析）:
#   ./scripts/train_swin_mae_dental.sh                                        # 官方 Swin-Small(22k) 权重初始化
#   ./scripts/train_swin_mae_dental.sh --resume exp/swin_mae_dental_small_448/checkpoint-050.pth   # 中断恢复
#   ./scripts/train_swin_mae_dental.sh --checkpoint_encoder <adapted-ckpt>   #  自定义编码器初始化
#
# 编码器变体（2026-09-20 由 Swin-T 改为 Swin-Small，与 ours 的 swins 预设对齐）：
#   本脚本显式传 --model swin_mae_s，与 INIT_CKPT 的官方
#   swin_small_patch4_window7_224_22k.pth 成对使用；config.yaml 的 MODEL.MODEL 也已是
#   swin_mae_s（这里显式传参是为了即使 config.yaml 因 cwd 找不到而回落到 argparse 默认值
#   —— --model swin_mae + depths (2,2,2,2) —— 也仍然是正确架构）。
#   ⚠️ Swin-T 与 Swin-Small 的 embed_dim(96) / num_heads 完全相同，错配不会报 shape 错，
#   load_model 会静默丢 block（tiny ckpt 喂 swin_mae_s = stage2 只填 6/18 个 block）。
#   要换回 tiny：追加 --model swin_mae_t --checkpoint_encoder .../swin_tiny_patch4_window7_224.pth
#   （命令行参数在最后，会覆盖脚本里的值）。
#
# 分辨率说明：默认 INPUT_SIZE=448，与 ours（full_canvas 448）统一，dense 特征网格
#   14x14（stride 32）。官方 swin_small_patch4_window7_224_22k.pth 可直接初始化：
#   - Swin 编码器用相对位置编码（window 7，bias 表 13x13）与 patch_embed（4x4），
#     均与输入分辨率无关；
#   - MAE 的 pos_embed 是固定正弦编码（requires_grad=False），initialize_weights 会
#     按 448 的 num_patches 自动重生成 112x112，不依赖 checkpoint。
#   需要 224 时：INPUT_SIZE=224 ./scripts/train_swin_mae_dental.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 目录定位（脚本位于 <root>/baselines/Swin-MAE/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWINMAE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SWINMAE_ROOT}/../.." && pwd)"

# 指定用于训练的 GPU，可以写成 "cuda:0 cuda:1 cuda:2" 或 "0 1 2"
GPUS="${GPUS:-2 3 4 5 6 7}"

# 训练输入分辨率（与 ours full_canvas=448 统一）；patch token 数随分辨率平方增长，
# 448 下 batch_size 已相应调小（224 时为 64）
INPUT_SIZE="${INPUT_SIZE:-448}"

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

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。显式设置 DATA_ROOT / SPLIT_JSON 时以环境变量为准，不回退。
resolve_dental_path() {
    # $1 = 相对 .datasets 层名的后缀（"" 或 "/train_test_seed0_test0p2.json"）
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "/train_test_seed0_test0p2.json")}"
# Swin-Small + 448 用独立输出目录：目录名带 small，与旧的 exp/swin_mae_dental_448
# （Swin-T 产物，被 9 个 compare/outputs/*/manifest.json 引用）彻底隔离，避免新架构
# 覆盖历史 checkpoint。
OUTPUT_DIR="${OUTPUT_DIR:-${SWINMAE_ROOT}/exp/swin_mae_dental_small_448}"
# 必须与下面的 --model swin_mae_s 成对：tiny/small 的 embed_dim 相同，错配会静默丢 block
INIT_CKPT="${INIT_CKPT:-${WORKSPACE_ROOT}/.checkpoints/intraoral/swin_small_patch4_window7_224_22k.pth}"

# 额外透传参数（如 --resume ...、--category_sampling ...、--split test）
EXTRA_ARGS="$@"

cd "${SWINMAE_ROOT}"
echo "[train_swin_mae_dental] GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[train_swin_mae_dental] DATA_ROOT=${DATA_ROOT}"
echo "[train_swin_mae_dental] SPLIT_JSON=${SPLIT_JSON}"
echo "[train_swin_mae_dental] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train_swin_mae_dental] MODEL=swin_mae_s (Swin-Small) INIT_CKPT=${INIT_CKPT}"

torchrun --nproc_per_node="${NUM_GPUS}" train.py \
    --input_size "${INPUT_SIZE}" \
    --batch_size 16 \
    --epochs 200 \
    --warmup_epochs 20 \
    --mask_ratio 0.75 \
    --lr 1e-4 \
    --min_lr 1e-7 \
    --weight_decay 0.05 \
    --save_freq 50 \
    --vis_freq 100 \
    --vis_num_images 8 \
    --norm_pix_loss \
    --data_path "${DATA_ROOT}" \
    --split_json "${SPLIT_JSON}" \
    --split train \
    --categories "full,tooth,sextant,group" \
    --model swin_mae_s \
    --checkpoint_encoder "${INIT_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

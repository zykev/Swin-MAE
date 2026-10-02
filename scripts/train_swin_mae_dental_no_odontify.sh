#!/bin/sh
set -e

# Swin-MAE 牙科 SSL 预训练 —— no-odontify 病例级 split
#（自有采集者 + AIDC 纳入训练；odontify 外部集整体排除）
#
# 独立脚本：自带全部训练超参与 GPU / 端口配置，直接调 torchrun。
#
# 用法（在 baselines/Swin-MAE 下执行；从任意目录运行也可，路径自动解析）:
#   ./scripts/train_swin_mae_dental_no_odontify.sh
#   ./scripts/train_swin_mae_dental_no_odontify.sh --resume exp/swin_mae_dental_small_448_no_odontify/checkpoint-50.pth
#   ./scripts/train_swin_mae_dental_no_odontify.sh --checkpoint_encoder <adapted-ckpt>
#
# 可覆盖环境变量:
#   GPUS        默认 6 卡 GPU 2-7（分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀）
#               GPUS="2 3" / GPUS="2,3" / GPUS="cuda:2 cuda:3" 三者等价
#   INPUT_SIZE  默认 448（与 ours full_canvas=448 统一）
#   DATA_ROOT   默认 <root>/.datasets/intraoral（挂载点缺失时回退 intraoral1）
#   SPLIT_JSON  默认 .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   OUTPUT_DIR  默认 baselines/Swin-MAE/exp/swin_mae_dental_small_448_no_odontify
#   INIT_CKPT   默认 .checkpoints/intraoral/swin_small_patch4_window7_224_22k.pth
#   MASTER_PORT 默认 29532（torchrun 默认 29500 常被其它任务占用）
#
# 编码器变体（Swin-Small，与 ours 的 swins 预设对齐）：
#   本脚本显式传 --model swin_mae_s，与 INIT_CKPT 的官方
#   swin_small_patch4_window7_224_22k.pth 成对使用；config.yaml 的 MODEL.MODEL 也已是
#   swin_mae_s（这里显式传参是为了即使 config.yaml 因 cwd 找不到而回落到 argparse 默认值
#   —— --model swin_mae + depths (2,2,2,2) —— 也仍然是正确架构）。
#   ⚠️ Swin-T 与 Swin-Small 的 embed_dim(96) / num_heads 完全相同，错配不会报 shape 错，
#   load_model 会静默丢 block（tiny ckpt 喂 swin_mae_s = stage2 只填 6/18 个 block）。
#   要换回 tiny：追加 --model swin_mae_t --checkpoint_encoder .../swin_tiny_patch4_window7_224.pth
#   （命令行参数在最后，会覆盖脚本里的值）。
#
# 分辨率说明：默认 448，dense 特征网格 14x14（stride 32）。官方
#   swin_small_patch4_window7_224_22k.pth 可直接初始化：
#   - Swin 编码器用相对位置编码（window 7，bias 表 13x13）与 patch_embed（4x4），
#     均与输入分辨率无关；
#   - MAE 的 pos_embed 是固定正弦编码（requires_grad=False），initialize_weights 会
#     按 448 的 num_patches 自动重生成 112x112，不依赖 checkpoint。
#   需要 224 时：INPUT_SIZE=224 ./scripts/train_swin_mae_dental_no_odontify.sh
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 目录定位（脚本位于 <root>/baselines/Swin-MAE/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWINMAE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SWINMAE_ROOT}/../.." && pwd)"

# 指定用于训练的 GPU：分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀
#   GPUS="2 3"   GPUS="2,3"   GPUS="cuda:2 cuda:3"   GPUS="2, 3"
GPUS="${GPUS:-2 3 4 5 6 7}"

# 解析 GPUS -> torchrun 用的 CUDA_VISIBLE_DEVICES 列表与卡数 NUM_GPUS
# set -f：解析期间关闭通配符展开（GPUS 里误写 * ? [ 时不会被 glob 成文件名）。
CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
set -f
for GPU in $(printf '%s\n' "${GPUS}" | tr ',;' '  '); do
    GPU_ID=${GPU#cuda:}
    case "${GPU_ID}" in
        ''|*[!0-9]*)
            echo "[train_swin_mae_dental_no_odontify] ERROR: GPUS 中的 '${GPU}' 不是合法 GPU 序号" >&2
            echo "  合法写法: GPUS=\"2 3\" / GPUS=\"2,3\" / GPUS=\"cuda:2 cuda:3\"" >&2
            exit 1
            ;;
    esac
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES},${GPU_ID}
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done
set +f
if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "[train_swin_mae_dental_no_odontify] ERROR: GPUS 为空，应形如 GPUS=\"2 3\" 或 GPUS=\"2,3\"" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES

# 分布式 rendezvous 端口：避开常被占用的 29500
MASTER_PORT="${MASTER_PORT:-29532}"

# 训练输入分辨率（与 ours full_canvas=448 统一）；patch token 数随分辨率平方增长，
# 448 下 batch_size 已相应调小（224 时为 64）
INPUT_SIZE="${INPUT_SIZE:-448}"

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。显式设置 DATA_ROOT / SPLIT_JSON 时以环境变量为准，不回退。
resolve_dental_path() {
    # $1 = 相对 .datasets 层名的后缀（"" 或 "/train_test_seed0_test0p2_no_odontify.json"）
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "/train_test_seed0_test0p2_no_odontify.json")}"
# Swin-Small + 448 + no-odontify 用独立输出目录：不覆盖常规 split 产物，
# 也不与旧的 Swin-T 产物 exp/swin_mae_dental_448（9 个 compare manifest 的引用对象）混淆
OUTPUT_DIR="${OUTPUT_DIR:-${SWINMAE_ROOT}/exp/swin_mae_dental_small_448_no_odontify}"
# 必须与下面的 --model swin_mae_s 成对：tiny/small 的 embed_dim 相同，错配会静默丢 block
INIT_CKPT="${INIT_CKPT:-${WORKSPACE_ROOT}/.checkpoints/intraoral/swin_small_patch4_window7_224_22k.pth}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[train_swin_mae_dental_no_odontify] ERROR: split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

# 额外透传参数（如 --resume ...、--category_sampling ...、--split test）
EXTRA_ARGS="$@"

cd "${SWINMAE_ROOT}"
echo "[train_swin_mae_dental_no_odontify] GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS} MASTER_PORT=${MASTER_PORT}"
echo "[train_swin_mae_dental_no_odontify] DATA_ROOT=${DATA_ROOT}"
echo "[train_swin_mae_dental_no_odontify] SPLIT_JSON=${SPLIT_JSON}"
echo "[train_swin_mae_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train_swin_mae_dental_no_odontify] MODEL=swin_mae_s (Swin-Small) INIT_CKPT=${INIT_CKPT}"

# ckpt 保存频率：--save_freq 10 → 每 10 个 epoch 存一次（train.py:
# (epoch + 1) % save_freq == 0 or epoch + 1 == epochs）→ 50 epoch 下产出
# checkpoint-10/20/30/40/50.pth（文件名不补零，数字是 epoch+1）。
# --vis_freq 保持 50：只影响重建可视化频率，与 ckpt 无关。
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" train.py \
    --input_size "${INPUT_SIZE}" \
    --batch_size 16 \
    --epochs 50 \
    --warmup_epochs 5 \
    --mask_ratio 0.75 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --save_freq 10 \
    --vis_freq 50 \
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

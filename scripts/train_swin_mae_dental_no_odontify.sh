#!/bin/sh
set -e

# Swin-MAE 牙科 SSL 预训练 —— 新公开 split（自有采集者 + AIDC，排除 odontify）
#
# 本脚本是 scripts/train_swin_mae_dental.sh 的「预设包装」：只预设下面 4 个环境变量，
# 训练超参（200 epoch / lr 1e-4 / batch 16 / INPUT_SIZE=448 /
# categories full,tooth,sextant,group / --model swin_mae_s + 官方 Swin-Small(22k) 权重初始化 …）
# 全部沿用原脚本，不在两处重复声明，避免以后改超参时两边漂移。
# （原脚本已于 2026-09-20 由 Swin-T 切换为 Swin-Small，本包装自动跟随。）
#
#   1) SPLIT_JSON  -> <root>/.datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#                     新的病例级 split（把 AIDC 纳入训练、排除 odontify 外部集）
#   2) DATA_ROOT   -> <root>/.datasets/intraoral（intraoral -> intraoral1 回退不变）
#   3) OUTPUT_DIR  -> exp/swin_mae_dental_small_448_no_odontify
#                     与旧的 exp/swin_mae_dental_448 完全隔离：旧产物是 Swin-T 结果、
#                     被 9 个 compare/outputs/*/manifest.json 引用，绝不能覆盖；
#                     目录名带 small 以标示这次是 Swin-Small 架构。
#   4) MASTER_PORT -> 29532（与原脚本默认 29501 错开，便于 4 个 baseline 并行起）
#
# 用法（从任意目录；参数原样透传给原脚本）:
#   bash baselines/Swin-MAE/scripts/train_swin_mae_dental_no_odontify.sh
#   bash baselines/Swin-MAE/scripts/train_swin_mae_dental_no_odontify.sh \
#       --resume exp/swin_mae_dental_small_448_no_odontify/checkpoint-050.pth
#
# 环境变量优先级仍然是「显式传入 > 本脚本预设」，例如:
#   GPUS="0 1" OUTPUT_DIR=/tmp/swinmae_try \
#       bash baselines/Swin-MAE/scripts/train_swin_mae_dental_no_odontify.sh
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)

# 目录定位（脚本位于 <root>/baselines/Swin-MAE/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWINMAE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SWINMAE_ROOT}/../.." && pwd)"
ORIGINAL_SCRIPT="${SCRIPT_DIR}/train_swin_mae_dental.sh"

# 新 split 的文件名（相对 .datasets/<collector-root> 的后缀）
NEW_SPLIT_SUFFIX="/train_test_seed0_test0p2_no_odontify.json"

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。与原脚本同一套规则。
resolve_dental_path() {
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

export DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
export SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "${NEW_SPLIT_SUFFIX}")}"
# 新 split 用独立输出目录，避免覆盖 compare/ manifest 还在引用的旧产物
export OUTPUT_DIR="${OUTPUT_DIR:-${SWINMAE_ROOT}/exp/swin_mae_dental_small_448_no_odontify}"
export MASTER_PORT="${MASTER_PORT:-29532}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[train_swin_mae_dental_no_odontify] ERROR: 新 split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

echo "[train_swin_mae_dental_no_odontify] DATA_ROOT=${DATA_ROOT}"
echo "[train_swin_mae_dental_no_odontify] SPLIT_JSON=${SPLIT_JSON}"
echo "[train_swin_mae_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train_swin_mae_dental_no_odontify] MASTER_PORT=${MASTER_PORT}"
echo "[train_swin_mae_dental_no_odontify] -> exec ${ORIGINAL_SCRIPT} $*"

exec sh "${ORIGINAL_SCRIPT}" "$@"

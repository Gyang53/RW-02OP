#!/bin/bash
# 在 Newton Viewer 里回放 nezhav1 训练好的策略（Isaac Lab 环境，GPU 物理）
#
# 用法:
#   ./isaac/play_viewer.sh                       # 用默认 checkpoint
#   ./isaac/play_viewer.sh <checkpoint.pt> 16    # 指定 checkpoint 和环境数
#
# 视图操作: 鼠标左键旋转 / 中键平移 / 滚轮缩放；左侧面板可 Pause / Reset Episode。

set -euo pipefail

ISAACLAB=${ISAACLAB:-/home/yangg/data/isaacsim/IsaacLab}
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT=${1:-}
NUM_ENVS=${2:-9}

if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "$ISAACLAB"/logs/rsl_rl/nezha_flat/*/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$CKPT" ]; then
        echo "找不到 checkpoint，请显式传入：./isaac/play_viewer.sh <checkpoint.pt>" >&2
        exit 1
    fi
fi

echo "[play] checkpoint = $CKPT"
echo "[play] num_envs   = $NUM_ENVS"
echo "[play] 关闭窗口或 Ctrl+C 退出"

cd "$ISAACLAB"
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
    PYTHONEXE="$ISAACLAB/.venv/bin/python" \
    ./_isaac_sim/python.sh scripts/reinforcement_learning/play.py \
    --task Isaac-Velocity-Flat-Nezha \
    --rl_library rsl_rl \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CKPT" \
    --visualizer newton_gl

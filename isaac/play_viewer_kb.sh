#!/bin/bash
# 在 Newton Viewer 里回放 nezhav1 训练好的策略，键盘控制运动。
#
# cd /home/yangg/data/isaacsim/IsaacLab
# env -u CONDA_PREFIX PYTHONEXE=./.venv/bin/python ./_isaac_sim/python.sh \
#     <ws>/isaac/play_keyboard.py --checkpoint <ckpt> --num_envs 1 --visualizer newton_gl
#
# 键盘（查看器窗口必须在前台）:
#   UP / DOWN  — 前进 / 停止
#   LEFT / RIGHT — 左转 / 右转
#   释放自动停止

set -euo pipefail

ISAACLAB=${ISAACLAB:-/home/yangg/data/isaacsim/IsaacLab}
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT=${1:-}
NUM_ENVS=${2:-1}

if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "$ISAACLAB"/logs/rsl_rl/nezha_flat/*/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$CKPT" ]; then
        echo "找不到 checkpoint，请显式传入：./isaac/play_viewer_kb.sh <checkpoint.pt>" >&2
        exit 1
    fi
fi

echo "[play] checkpoint = $CKPT"
echo "[play] num_envs   = $NUM_ENVS"
echo "[play] 键盘控制：↑ 前进 / ←→ 转向 / 释放停止"
echo "[play] 关闭窗口或 Ctrl+C 退出"

cd "$ISAACLAB"
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
    PYTHONEXE="$ISAACLAB/.venv/bin/python" \
    ./_isaac_sim/python.sh "$WS"/isaac/play_keyboard.py \
    --task "Isaac-Velocity-Flat-Nezha" \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CKPT" \
    --visualizer newton_gl

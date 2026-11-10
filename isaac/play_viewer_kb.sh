#!/bin/bash
# 在 Newton Viewer 里回放 nezhav1 训练好的策略，键盘全功能控制机器人。
#
# cd /home/yangg/data/isaacsim/IsaacLab
# env -u CONDA_PREFIX PYTHONEXE=./.venv/bin/python ./_isaac_sim/python.sh \
#     <ws>/isaac/play_keyboard.py --checkpoint <ckpt> --num_envs 1 --visualizer newton_gl
#
# 键盘（查看器窗口必须在前台，可叠加）:
#   W / S   — 前进 / 后退
#   A / D   — 左转 / 右转
#   Z / X   — 蹲下 / 站高；C — 中间高度
#   R       — 手动复位全部环境（演示倒下→起身）
#   释放行走键自动停止

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
    --task "${TASK:-Isaac-Velocity-Flat-Nezha}" \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CKPT" \
    --visualizer newton_gl

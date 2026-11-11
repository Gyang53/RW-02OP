#!/bin/bash
# nezhav1 全功能 Newton Viewer 演示（一键）
#
# 用法:
#   ./isaac/demo_all.sh                 # 依次演示 A/B/C（每段结束后按回车继续）
#   ./isaac/demo_all.sh A               # 只跑 A：行走 + 蹲站
#   ./isaac/demo_all.sh B               # 只跑 B：跳跃（0.12 m 障碍）
#   ./isaac/demo_all.sh C               # 只跑 C：倒地 / 起身
#   ./isaac/demo_all.sh D               # 只跑 D：发射式跳跃（G 键）
#
# 演示时的操作（Newton Viewer 窗口必须在前台，点一下窗口再按键）:
#   W/S      前进 / 后退
#   A/D      左转 / 右转
#   Z/X/C    蹲下 0.35 / 站高 0.60 / 中间 0.50
#   R        复位全部环境
#   跳跃段：靠近障碍时自动执行蹲跳原语（遥测里 jump= 会显示 crouch/extend/tuck）
#
# 各段验收看点（终端遥测，每 2 s 一行）:
#   A: cmd=(+1.00,+0.00) 时 vx≈0.97~0.99；Z 后 h_cmd=0.35 且 z≈0.37~0.42
#   B: jump=crouch→extend→tuck 依次出现；tuck 时 vx 会升到 ~1.8
#   C: fallen=N/8 出现后，观察 8 s 是否减少（当前实现：不减少 = 站不起来）

set -euo pipefail

ISAACLAB=${ISAACLAB:-/home/yangg/data/isaacsim/IsaacLab}
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WALK_CKPT=${WALK_CKPT:-$ISAACLAB/logs/rsl_rl/nezha_flat/2026-11-11_14-33-09/model_8748.pt}
GETUP_CKPT=${GETUP_CKPT:-$ISAACLAB/logs/rsl_rl/nezha_flat/2026-11-11_15-20-44/model_11100.pt}
LAUNCH_CKPT=${LAUNCH_CKPT:-$ISAACLAB/logs/rsl_rl/nezha_flat/2026-11-11_17-53-53/model_9350.pt}

run() {
    cd "$ISAACLAB"
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
        PYTHONEXE="$ISAACLAB/.venv/bin/python" \
        ./_isaac_sim/python.sh "$WS/isaac/play_keyboard.py" "$@"
}

segment_A() {
    echo "== A) 行走 + 蹲站（4 个机器人，平地）=="
    echo "   按键: W/S 前后 · A/D 转向 · Z/X/C 蹲/站/中 · R 复位"
    run --task Isaac-Velocity-Flat-Nezha-S06 --num_envs 4 \
        --checkpoint "$WALK_CKPT" --visualizer newton_gl \
        --h_crouch 0.35 --h_stand 0.60 --h_mid 0.50
}

segment_B() {
    echo "== B) 跳跃（0.12 m 固定障碍，自动蹲跳原语）=="
    echo "   看点: 遥测 jump=crouch→extend→tuck；靠近障碍时 obs_dx 变小"
    run --task Isaac-Velocity-Flat-Nezha-S06-JumpTest --num_envs 4 \
        --checkpoint "$WALK_CKPT" --visualizer newton_gl \
        --jump --jump_dist 0.5 \
        --h_crouch 0.35 --h_stand 0.60 --h_mid 0.50
}

segment_C() {
    echo "== C) 倒地 / 起身（8 个机器人，50% 倒地起步）=="
    echo "   看点: fallen=N/8 是否在 8 s 内减少（当前实现: 不减少）"
    run --task Isaac-Velocity-Flat-Nezha-S06-GetUpTest --num_envs 8 \
        --checkpoint "$GETUP_CKPT" --visualizer newton_gl \
        --h_crouch 0.35 --h_stand 0.60 --h_mid 0.50
}

segment_D() {
    echo "== D) 发射式跳跃（G 键 = 模拟发射机构，0.6 m 高度，含 10 kg 载荷）=="
    echo "   按键: G 发射 · W/S/A/D 行走 · Z/X/C 蹲站 · R 复位"
    echo "   看点: 按下 G 后 z 由 ~0.50 升到 ~1.08（抬升 0.59 m），落地 fallen=0"
    run --task Isaac-Velocity-Flat-Nezha-S06-Launch --num_envs 4 \
        --checkpoint "$LAUNCH_CKPT" --visualizer newton_gl \
        --launch --h_crouch 0.35 --h_stand 0.60 --h_mid 0.50
}

case "${1:-ALL}" in
    A) segment_A ;;
    B) segment_B ;;
    C) segment_C ;;
    D) segment_D ;;
    ALL)
        segment_A
        echo; read -r -p "A 段结束，回车继续 B 段（Ctrl+C 退出）..." _
        segment_B
        echo; read -r -p "B 段结束，回车继续 C 段（Ctrl+C 退出）..." _
        segment_C
        ;;
    *) echo "用法: $0 [A|B|C|D|ALL]" >&2; exit 1 ;;
esac

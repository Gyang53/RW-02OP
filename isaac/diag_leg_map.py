"""标定：在 Isaac 里实测 (hip, knee) → 轮相对机身的位置。

为什么需要：之前用 MJCF 解析几何推的 hip↔knee 映射与 Isaac/USD 的实际运动学不符
（实测 knee+0.63 / hip−0.30 时，轮 link 相对 base 的竖直距离只有 0.139 m，
而腿长 0.425 m ⇒ 腿斜了约 71°）。所有基于解析模型的轨迹都必须重算。

做法：对每个 (hip, knee) 样本，用动作命令把腿 PD 保持在该姿态（15 步），
然后实测「轮 link − base」的 (Δx, Δz)，以及实际到达的关节角。
输出可直接用于反查"竖直伸展"轨迹（要求 Δx≈0）。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_leg_map.py --visualizer none --num_envs 1
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=15)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEG_SCALE = 0.25


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    dev = env.unwrapped.device
    jn = robot.joint_names
    kidx = jn.index("knee_L_Joint")
    hidx = jn.index("hip_L_Joint")
    bidx_base = robot.body_names.index("base_link")
    bidx_wheel = robot.body_names.index("wheel_L_Link")

    knees = [-1.40, -1.00, -0.60, -0.20, 0.20, 0.60]
    hips = [-0.90, -0.45, 0.00, 0.45, 0.90, 1.00]

    print("=" * 78)
    print(f"{'cmd hip':>8} {'cmd knee':>9} | {'act hip':>8} {'act knee':>9} | {'Δx':>7} {'Δz':>7} | {'竖直长度':>9}")
    print("-" * 78)
    rows = []
    for knee_c in knees:
        for hip_c in hips:
            act = torch.zeros(env.action_space.shape, device=dev)
            act[:, kidx] = knee_c / LEG_SCALE
            act[:, hidx] = hip_c / LEG_SCALE
            for _ in range(args_cli.settle):
                env.step(act)
            m = robot.data.body_pos_w
            d = (m[:, bidx_wheel] - m[:, bidx_base]).mean(dim=0)
            hip_a = robot.data.joint_pos[:, hidx].mean().item()
            knee_a = robot.data.joint_pos[:, kidx].mean().item()
            rows.append((hip_a, knee_a, d[0].item(), d[2].item()))
            print(f"{hip_c:>8.2f} {knee_c:>9.2f} | {hip_a:>8.2f} {knee_a:>9.2f} | "
                  f"{d[0].item():>7.3f} {d[2].item():>7.3f} | {abs(d[2].item()):>9.3f}")
    print("-" * 78)
    # 对每个实测 knee，找 Δx 最接近 0 的样本（即腿竖直）
    print("按实测 knee 分组的『最竖直』样本（Δx≈0）：")
    by_knee = {}
    for hip_a, knee_a, dx, dz in rows:
        k = round(knee_a, 2)
        if k not in by_knee or abs(dx) < abs(by_knee[k][1]):
            by_knee[k] = (hip_a, dx, dz)
    for k in sorted(by_knee):
        hip_a, dx, dz = by_knee[k]
        print(f"  knee={k:+.2f} -> hip={hip_a:+.2f}  Δx={dx:+.3f}  竖直腿长={abs(dz):.3f}")
    print("=" * 78)
    env.close()
    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        import sys

        sys.stdout.flush()
        simulation_app.close()
    raise SystemExit(code)

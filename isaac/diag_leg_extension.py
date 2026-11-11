"""隔离诊断：在 Isaac 里单看「腿伸展 → 机身抬升」，排除策略/障碍/原语干扰。

做法：把所有环境固定到同一个蹲姿、锁住轮子、只给腿一个饱和伸展指令，
记录 base_z / 竖直速度 / 膝角的响应。若机身抬不起来，说明是执行器/腿模型问题；
若正常抬起，说明问题在接触时机或障碍交互。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_leg_extension.py --visualizer none --num_envs 8
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-GetUpTest")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--crouch_knee", type=float, default=-1.40)
parser.add_argument("--extend_knee", type=float, default=0.663)
parser.add_argument("--hold_steps", type=int, default=40, help="伸展保持的环境步数")
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
    kidx = [jn.index("knee_L_Joint"), jn.index("knee_R_Joint")]
    hidx = [jn.index("hip_L_Joint"), jn.index("hip_R_Joint")]
    n = args_cli.num_envs

    def set_pose(knee, hip, z):
        q = robot.data.default_joint_pos.clone()
        for i in kidx:
            q[:, i] = knee
        for i in hidx:
            q[:, i] = hip
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        pose = robot.data.default_root_state.clone()
        pose[:, 2] = z
        pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        robot.write_root_pose_to_sim(pose[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev))

    # 蹲姿（几何：knee=-1.40 时机身约 0.34）
    set_pose(args_cli.crouch_knee, 0.40, 0.34)
    # ⚠ 稳定阶段必须**持续命令蹲姿**：动作 0 意味着关节目标 0（站立），
    # 会在 30 步内把蹲姿顶回去（实测膝角从 -1.40 被拉到 -0.09）。
    crouch_act = torch.zeros(env.action_space.shape, device=dev)
    for i in kidx:
        crouch_act[:, i] = args_cli.crouch_knee / LEG_SCALE
    for i in hidx:
        crouch_act[:, i] = 0.40 / LEG_SCALE
    for _ in range(30):
        env.step(crouch_act)
    z_before = robot.data.root_pos_w[:, 2].mean().item()
    print(f"[legdiag] 蹲姿稳定后 base_z = {z_before:.3f} m")

    # 饱和伸展：膝 +0.663、髋 +0.40、轮 0（锁定）
    act = torch.zeros(env.action_space.shape, device=dev)
    for i in kidx:
        act[:, i] = args_cli.extend_knee / LEG_SCALE
    for i in hidx:
        act[:, i] = 0.40 / LEG_SCALE

    z_max, vz_max = z_before, -9.9
    knee_log = []
    for step in range(args_cli.hold_steps):
        env.step(act)
        z = robot.data.root_pos_w[:, 2]
        vz = robot.data.root_lin_vel_w[:, 2]
        z_max = max(z_max, z.mean().item())
        vz_max = max(vz_max, vz.mean().item())
        knee_log.append(robot.data.joint_pos[:, kidx[0]].mean().item())

    print("=" * 60)
    print(f"[legdiag] 伸展 {args_cli.hold_steps} 步（每步 {env.unwrapped.step_dt*1000:.1f} ms）")
    print(f"[legdiag] 起始 base_z = {z_before:.3f}")
    print(f"[legdiag] base_z 峰值 = {z_max:.3f}   抬升 = {z_max - z_before:+.3f} m")
    print(f"[legdiag] 竖直速度峰值 = {vz_max:+.3f} m/s")
    print(f"[legdiag] 膝角轨迹: 起 {knee_log[0]:+.2f} -> 末 {knee_log[-1]:+.2f}（目标 {args_cli.extend_knee:+.2f}）")
    print(f"[legdiag] 判定: {'✅ 腿伸展能抬起机身' if (z_max - z_before) > 0.08 else '❌ 抬不起来（执行器/腿模型问题）'}")
    print("=" * 60)
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

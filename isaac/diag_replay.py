"""参考动作实验：录下策略自身的「蹲 → 站」轨迹，再按不同速度回放。

动机（见 docs 6B.5 系列）：静态 keyframe 必然漂姿态；而策略高度指令 h 0.35→0.60
本身是**动力学有效**的竖直位移（z 0.373→0.602，全长 g_z≈−1.00）。若把这条有效轨迹
按时间压缩回放，末端可能出现正的竖直速度——那就是可用的"起跳"。

流程：
1. 策略在 h=0.35 收敛（120 步），记录关节角；
2. 把 h 切到 0.60，记录 120 步的关节角序列（含轮速动作）→ 参考轨迹；
3. 复位后按 speed ∈ {1, 2, 3, 4} 倍速回放该关节序列（作为位置目标），
   测量 base_z 与竖直速度。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_replay.py --visualizer none --checkpoint <walk_ckpt>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--h_low", type=float, default=0.35)
parser.add_argument("--h_high", type=float, default=0.60)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEG_SCALE = 0.25
WHEEL_SCALE = 10.0


class MlpPolicy:
    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mean = ck["obs_normalizer._mean"].numpy().astype(np.float64).reshape(-1)
        self.var = ck["obs_normalizer._var"].numpy().astype(np.float64).reshape(-1)
        self.w = [ck[f"mlp.{i}.weight"].numpy().T for i in (0, 2, 4, 6)]
        self.b = [ck[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]

    def __call__(self, obs):
        x = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(3):
            x = x @ self.w[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x @ self.w[3] + self.b[3]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.base_velocity.heading_command = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    # 该 gym 包装层没有 seed()（OrderEnforcing），改由 parse_env_cfg 的 seed 控制
    try:
        env.unwrapped.seed(0)
    except Exception:  # noqa: BLE001
        pass
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    hcmd = env.unwrapped.command_manager.get_command("height")
    dev = env.unwrapped.device
    jn = robot.joint_names
    leg_idx = [i for i, n in enumerate(jn) if ("abad" in n or "hip" in n or "knee" in n)]
    wheel_idx = [jn.index("wheel_L_Joint"), jn.index("wheel_R_Joint")]

    def step_policy(h: float):
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
        hcmd[:, 0] = h
        nonlocal obs
        if isinstance(obs, dict):
            obs = obs["policy"]
        act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32, device=dev)
        obs, _, _, _, _ = env.step(act)

    # 1) 蹲姿收敛
    for _ in range(120):
        step_policy(args_cli.h_low)
    z_low = robot.data.root_pos_w[:, 2].mean().item()
    print(f"[replay] 蹲姿收敛: z={z_low:.3f}")

    # 2) 录参考轨迹（h 切到 h_high）
    traj = []
    for _ in range(120):
        step_policy(args_cli.h_high)
        traj.append(robot.data.joint_pos[0, leg_idx].cpu().numpy().copy())
    z_high = robot.data.root_pos_w[:, 2].mean().item()
    print(f"[replay] 参考轨迹录完: z {z_low:.3f} → {z_high:.3f}（{len(traj)} 步）")

    # 3) 倍速回放
    print("=" * 66)
    print(f"{'speed':>6} {'z_end':>8} {'z_max':>8} {'vz_end':>8} {'vz_max':>8}")
    print("-" * 66)
    for speed in (1.0, 2.0, 3.0, 4.0):
        env.reset()
        for _ in range(20):
            env.step(torch.zeros(env.action_space.shape, device=dev))
        # 回到蹲姿（用策略快速收敛到 h_low，保证起点一致）
        for _ in range(120):
            step_policy(args_cli.h_low)
        z0 = robot.data.root_pos_w[:, 2].mean().item()
        vz_max, z_max = -9.9, z0
        n = len(traj)
        i = 0.0
        while i < n - 1:
            k = int(i)
            q = traj[k]
            a = torch.zeros(env.action_space.shape, device=dev)
            for j, idx in enumerate(leg_idx):
                a[:, idx] = torch.tensor(q[j] / LEG_SCALE, device=dev)
            a[:, wheel_idx[0]] = 0.0
            a[:, wheel_idx[1]] = 0.0
            env.step(a)
            z = robot.data.root_pos_w[:, 2].mean().item()
            vz = robot.data.root_lin_vel_w[:, 2].mean().item()
            z_max = max(z_max, z)
            vz_max = max(vz_max, vz)
            i += speed
        z_end = robot.data.root_pos_w[:, 2].mean().item()
        vz_end = robot.data.root_lin_vel_w[:, 2].mean().item()
        print(f"{speed:>6.1f} {z_end:>8.3f} {z_max:>8.3f} {vz_end:>+8.3f} {vz_max:>+8.3f}")
    print("=" * 66)
    print("[replay] 若某倍速下 vz_end > 0 且 z_max 明显高于站立(0.602)，说明压缩有效轨迹可产生弹射")
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

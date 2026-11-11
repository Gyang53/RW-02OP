"""读取策略在各个高度指令下的稳定关节角（用于给跳跃/起身原语提供真实蹲姿）。

背景：诊断里发现手写的蹲姿（knee −0.92 / hip +0.40）几乎不降低机身
（z 0.565 → 0.576），而策略按 Z 键能把 z 降到 0.37–0.42。说明策略用的是另一组关节角，
必须实测取出，不能凭 FK 猜。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_policy_pose.py --visualizer none \
      --checkpoint <walk_ckpt>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--heights", default="0.35,0.45,0.50,0.60")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEG = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint"]


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
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    hcmd = env.unwrapped.command_manager.get_command("height")
    jn = robot.joint_names
    leg_idx = [jn.index(n) for n in LEG]

    print("=" * 78)
    print(f"{'h_cmd':>7} {'z':>7} {'g_z':>7} {'abad_L':>8} {'hip_L':>8} {'knee_L':>8} {'knee_R':>8}")
    print("-" * 78)
    for h in [float(x) for x in args_cli.heights.split(",")]:
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
        hcmd[:, 0] = h
        for _ in range(120):  # 2.4 s，等姿态收敛
            hcmd[:, 0] = h
            if isinstance(obs, dict):
                obs = obs["policy"]
            act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32,
                                  device=env.unwrapped.device)
            obs, _, _, _, _ = env.step(act)
        q = robot.data.joint_pos[:, leg_idx].mean(dim=0).cpu().numpy()
        z = robot.data.root_pos_w[:, 2].mean().item()
        gz = robot.data.projected_gravity_b[:, 2].mean().item()
        knee_r = robot.data.joint_pos[:, jn.index("knee_R_Joint")].mean().item()
        print(f"{h:>7.2f} {z:>7.3f} {gz:>+7.2f} {q[0]:>+8.3f} {q[1]:>+8.3f} {q[2]:>+8.3f} {knee_r:>+8.3f}")
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

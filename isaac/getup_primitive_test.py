"""验证起身原语：在 V5 任务（50% 倒地起步）里注入仰卧起身原语，测站起率。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/getup_primitive_test.py --visualizer none \
      --num_envs 32 --steps 1200 --checkpoint <ckpt> [--use_primitive 0]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-V5")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--use_primitive", type=int, default=1)
parser.add_argument("--stand_z", type=float, default=0.55)
parser.add_argument("--wheel_action", type=float, default=2.0, help="翻转相位轮速动作（符号需实测）")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from getup_primitive import GetUpPrimitive  # noqa: E402


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
    env_cfg.episode_length_s = 30.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]

    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    hcmd = env.unwrapped.command_manager.get_command("height")
    dev = env.unwrapped.device
    dt = env.unwrapped.step_dt

    # 判定哪些环境是倒地起步（复位后空跑 10 步）
    for _ in range(10):
        env.step(torch.zeros(env.action_space.shape, device=dev))
    z0 = robot.data.root_pos_w[:, 2].cpu().numpy()
    fallen = z0 < 0.45

    prim = GetUpPrimitive(args_cli.num_envs, dev, stand_z=args_cli.stand_z,
                          wheel_flip_action=args_cli.wheel_action)
    if args_cli.use_primitive:
        prim.trigger(torch.as_tensor(fallen, device=dev))

    reached = np.zeros(args_cli.num_envs, dtype=bool)
    z_max = z0.copy()
    for i in range(args_cli.steps):
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
        hcmd[:, 0] = 0.50
        if isinstance(obs, dict):
            obs = obs["policy"]
        act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32, device=dev)
        if args_cli.use_primitive and prim.active.any():
            act = torch.where(prim.active.unsqueeze(-1), prim.actions(), act)
        obs, _, term, trunc, _ = env.step(act)
        if isinstance(obs, dict):
            obs = obs["policy"]
        z = robot.data.root_pos_w[:, 2].cpu().numpy()
        z_max = np.maximum(z_max, z)
        reached |= (z > args_cli.stand_z) & fallen
        prim.advance(dt, robot.data.root_pos_w[:, 2])

    n_fallen = int(fallen.sum())
    n_up = int((reached & fallen).sum())
    print("=" * 64)
    print(f"[getup-prim] task={args_cli.task} envs={args_cli.num_envs} steps={args_cli.steps}")
    print(f"[getup-prim] 模式 = {'原语' if args_cli.use_primitive else '纯策略(对照)'}")
    print(f"[getup-prim] 倒地起步 = {n_fallen}/{args_cli.num_envs}")
    print(f"[getup-prim] 站起 = {n_up}/{n_fallen}"
          f"  ({n_up / n_fallen * 100:.1f}%)" if n_fallen else "[getup-prim] 无倒地环境")
    if n_fallen:
        print(f"[getup-prim] z_max(倒地) 均值 = {z_max[fallen].mean():.3f} m")
    print("=" * 64)
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
        import sys as _s

        _s.stdout.flush()
        simulation_app.close()
    raise SystemExit(code)

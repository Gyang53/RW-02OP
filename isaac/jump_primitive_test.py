"""在 Isaac 环境里验证跳跃原语：策略行走 + 障碍临近时切换到原语，测越障成功率。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/jump_primitive_test.py --visualizer none \
      --num_envs 16 --steps 400 --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-V4")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--trigger_dist", type=float, default=0.6, help="障碍物距离小于该值时触发跳跃")
parser.add_argument("--vx", type=float, default=1.0)
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
from jump_primitive import JumpPrimitive  # noqa: E402


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
    has_obs = "obstacle" in env.unwrapped.scene.keys()
    if not has_obs:
        print("[jump] 场景无 obstacle，退出")
        return 0
    obstacle = env.unwrapped.scene["obstacle"]

    prim = JumpPrimitive(args_cli.num_envs, env.unwrapped.device)
    dt = env.unwrapped.step_dt

    n = args_cli.steps
    x0 = robot.data.root_pos_w[:, 0].clone()
    dev = env.unwrapped.device
    crossed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=dev)
    airborne_any = np.zeros(args_cli.num_envs, dtype=bool)
    z_max = np.zeros(args_cli.num_envs)
    wheel_air = np.zeros((n, args_cli.num_envs), dtype=bool)
    triggered = np.zeros(args_cli.num_envs, dtype=bool)

    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    names = list(sensor.body_names) if sensor is not None else []
    wl = names.index("wheel_L_Link") if "wheel_L_Link" in names else None
    wr = names.index("wheel_R_Link") if "wheel_R_Link" in names else None

    for i in range(n):
        cmd[:, 0] = args_cli.vx
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
        hcmd[:, 0] = 0.50

        # 障碍距离 → 触发原语
        dx = (obstacle.data.root_pos_w[:, 0] - robot.data.root_pos_w[:, 0]).abs()
        near = (dx < args_cli.trigger_dist) & (dx > -0.2)
        before = prim.active.clone()
        prim.trigger(near & ~crossed)
        triggered = triggered | (prim.active & ~before).cpu().numpy()
        prim.advance(dt)

        if isinstance(obs, dict):
            obs = obs["policy"]
        act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32,
                              device=env.unwrapped.device)
        if prim.active.any():
            prim_act = prim.actions()
            act = torch.where(prim.active.unsqueeze(-1), prim_act, act)

        obs, _, term, trunc, _ = env.step(act)
        if isinstance(obs, dict):
            obs = obs["policy"]

        x = robot.data.root_pos_w[:, 0].cpu().numpy()
        ox = obstacle.data.root_pos_w[:, 0].cpu().numpy()
        crossed |= torch.as_tensor(x > ox, device=dev)
        z = robot.data.root_pos_w[:, 2].cpu().numpy()
        z_max = np.maximum(z_max, z)
        if sensor is not None and wl is not None:
            f = sensor.data.net_forces_w
            if hasattr(f, "torch"):
                f = f.torch
            contact = (torch.linalg.norm(f[:, [wl, wr]], dim=-1) > 1.0).any(dim=1)
            wheel_air[i] = (~contact).cpu().numpy()
            airborne_any |= wheel_air[i]

    print("=" * 64)
    print(f"[jump] task={args_cli.task} envs={args_cli.num_envs} steps={n}")
    print(f"[jump] 触发原语的环境数 = {int(triggered.sum())}/{args_cli.num_envs}")
    crossed_np = crossed.cpu().numpy()
    print(f"[jump] 越过障碍成功率 = {crossed_np.mean() * 100:.1f}%  ({int(crossed_np.sum())}/{args_cli.num_envs})")
    print(f"[jump] 出现双轮离地的环境数 = {int(airborne_any.sum())}/{args_cli.num_envs}")
    print(f"[jump] 双轮离地帧占比 = {wheel_air.mean() * 100:.1f}%")
    print(f"[jump] base_z 最大值 = {z_max.max():.3f} m（站立 ≈ 0.60）")
    dx_end = (obstacle.data.root_pos_w[:, 0] - robot.data.root_pos_w[:, 0]).cpu().numpy()
    print("[jump] 结束时 env→障碍距离（负=已越过）:", np.round(dx_end, 2).tolist())
    adv = (robot.data.root_pos_w[:, 0] - x0).cpu().numpy()
    print("[jump] 逐环境前进距离(m):", np.round(adv, 2).tolist())
    print(f"[jump] 前进超过 {args_cli.trigger_dist * 2:.1f} m 的环境数 = {int((adv > args_cli.trigger_dist * 2).sum())}/{args_cli.num_envs}")
    print(f"[jump] 平均前进距离 = {adv.mean():.2f} m")
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

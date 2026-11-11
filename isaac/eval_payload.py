"""载荷验证：把机身质量强制设到指定值（默认 10 kg 附加），测行走稳定性。

与 eval_skills 的区别：后者用任务默认的随机质量分布；本脚本**固定**载荷，
用于回答"10 kg 载荷下能否稳定平衡行走"。

做法：覆盖任务的 `add_base_mass.mass_distribution_params`（倍数），并关闭质心随机化，
使所有环境都处于目标载荷。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/eval_payload.py --visualizer none \
      --checkpoint <ckpt> --extra_kg 10
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--extra_kg", type=float, default=10.0, help="附加载荷 (kg)")
parser.add_argument("--base_kg", type=float, default=9.921, help="MJCF 里 base_link 质量")
parser.add_argument("--vx", type=float, default=1.0)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


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
    # 强制固定载荷（关闭随机化）
    scale = (args_cli.base_kg + args_cli.extra_kg) / args_cli.base_kg
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (scale, scale)
    env_cfg.events.base_com = None  # 关闭质心随机，只看载荷
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    hcmd = env.unwrapped.command_manager.get_command("height")
    dev = env.unwrapped.device

    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    bnames = list(sensor.body_names) if sensor is not None else []
    wsel = [bnames.index(b) for b in ("wheel_L_Link", "wheel_R_Link") if b in bnames]

    vx_err, falls, contact, z_hist, gz_hist = [], 0, [], [], []
    for i in range(args_cli.steps):
        cmd[:, 0] = args_cli.vx
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
        hcmd[:, 0] = 0.50
        if isinstance(obs, dict):
            obs = obs["policy"]
        act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32, device=dev)
        obs, _, term, trunc, _ = env.step(act)
        vx = robot.data.root_lin_vel_b[:, 0]
        vx_err.append((vx - cmd[:, 0]).abs().mean().item())
        falls += int(term.sum().item())
        z_hist.append(robot.data.root_pos_w[:, 2].mean().item())
        gz_hist.append(robot.data.projected_gravity_b[:, 2].mean().item())
        if sensor is not None and len(wsel) == 2:
            f = sensor.data.net_forces_w
            if hasattr(f, "torch"):
                f = f.torch
            c = (torch.linalg.norm(f[:, wsel], dim=-1) > 1.0).any(dim=1).float().mean().item()
            contact.append(c)

    total_mass = args_cli.base_kg * scale
    print("=" * 62)
    print(f"[payload] 任务={args_cli.task} 附加载荷={args_cli.extra_kg:.1f} kg "
          f"(base {args_cli.base_kg:.2f} → {total_mass:.2f} kg)")
    print(f"[payload] envs={args_cli.num_envs} steps={args_cli.steps} cmd_vx={args_cli.vx}")
    print(f"[payload] mean |vx - cmd| = {np.mean(vx_err):.3f} m/s")
    print(f"[payload] 摔倒(terminated)次数 = {falls} / {args_cli.steps * args_cli.num_envs} 环境步")
    print(f"[payload] base_z 均值/最小 = {np.mean(z_hist):.3f} / {np.min(z_hist):.3f}")
    print(f"[payload] projected_gravity_z 均值/最小 = {np.mean(gz_hist):+.3f} / {np.min(gz_hist):+.3f}")
    if contact:
        print(f"[payload] 双轮接地比例 = {np.mean(contact) * 100:.1f}%")
    ok = (falls == 0) and (abs(np.mean(gz_hist) + 1.0) < 0.15) and (np.mean(vx_err) < 0.25)
    print(f"[payload] 判定 = {'✅ 稳定' if ok else '❌ 不稳定'}")
    print("=" * 62)
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

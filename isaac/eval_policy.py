"""用训练好的 rsl_rl checkpoint 回放并统计指标（headless）。

    cd /home/yangg/data/isaacsim/IsaacLab
    env -u CONDA_PREFIX PYTHONEXE=./.venv/bin/python ./_isaac_sim/python.sh \
        <ws>/isaac/eval_policy.py --visualizer none --num_envs 64 --steps 1000 \
        --checkpoint logs/rsl_rl/nezha_flat/<run>/model_1499.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--fixed_vx", type=float, default=None, help="强制固定指令 vx")
parser.add_argument("--fixed_wz", type=float, default=0.0, help="强制固定指令 wz")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


class MlpPolicy:
    """rsl_rl 的 actor：obs_normalizer -> mlp(ELU) -> 8 维动作（确定性）。"""

    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.obs_dim = int(ck["mlp.0.weight"].shape[1])
        self.mean = ck["obs_normalizer._mean"].numpy().astype(np.float64).reshape(-1)
        self.var = ck["obs_normalizer._var"].numpy().astype(np.float64)
        self.w = [ck[f"mlp.{i}.weight"].numpy().T for i in (0, 2, 4, 6)]
        self.b = [ck[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(3):
            x = x @ self.w[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))  # ELU
        return x @ self.w[3] + self.b[3]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")

    vx_err, wz_err, base_z, falls, done_count = [], [], [], 0, 0
    h_err = []
    for i in range(args_cli.steps):
        if args_cli.fixed_vx is not None:
            command[:, 0] = args_cli.fixed_vx
            command[:, 1] = 0.0
            command[:, 2] = args_cli.fixed_wz
        if isinstance(obs, dict):
            obs = obs["policy"]
        obs_np = obs.detach().cpu().numpy()
        actions = torch.as_tensor(policy(obs_np), dtype=torch.float32, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)

        vel = robot.data.root_lin_vel_b
        ang = robot.data.root_ang_vel_b
        h_cmd = env.unwrapped.command_manager.get_command("height")[:, 0]
        h_err.append((robot.data.root_pos_w[:, 2] - h_cmd).abs().mean().item())
        vx_err.append((vel[:, 0] - command[:, 0]).abs().mean().item())
        wz_err.append((ang[:, 2] - command[:, 2]).abs().mean().item())
        base_z.append(robot.data.root_pos_w[:, 2].mean().item())
        falls += int(terminated.sum().item())
        done_count += terminated.numel()

    print("=" * 60)
    print(f"[eval] steps={args_cli.steps} envs={args_cli.num_envs}")
    print(f"[eval] mean |vx - cmd|   = {np.mean(vx_err):.3f} m/s")
    print(f"[eval] mean |wz - cmd|   = {np.mean(wz_err):.3f} rad/s")
    print(f"[eval] mean base_z       = {np.mean(base_z):.3f} m  (min {np.min(base_z):.3f})")
    print(f"[eval] base_contact rate = {falls / done_count * 100:.1f} %")
    print(f"[eval] mean |z - h_cmd|  = {np.mean(h_err):.3f} m")
    print(f"[eval] final base_z      = {base_z[-1]:.3f} m")
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

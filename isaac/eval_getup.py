"""评测起身能力：从倒地姿态开始，统计有多少环境能站起来。

    cd /home/yangg/data/isaacsim/IsaacLab
    env -u CONDA_PREFIX PYTHONEXE=./.venv/bin/python ./_isaac_sim/python.sh \
        <ws>/isaac/eval_getup.py --visualizer none --num_envs 128 --steps 500 \
        --checkpoint <run>/model_xxxx.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-GetUp")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--stand_z", type=float, default=0.60)
parser.add_argument("--fall_prob", type=float, default=0.5, help="评测时的倒地概率")
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

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(3):
            x = x @ self.w[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x @ self.w[3] + self.b[3]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.events.reset_fallen.params["probability"] = args_cli.fall_prob
    env_cfg.events.reset_fallen.params["z_range"] = (0.30, 0.55)
    env_cfg.events.reset_fallen.params["tilt_range"] = (0.9, 1.4)
    env_cfg.curriculum.fall_prob = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    policy = MlpPolicy(args_cli.checkpoint)

    n = args_cli.num_envs
    # 复位后先空跑 10 步让机体落到地上，再判定哪些是倒地起步
    for _ in range(10):
        env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    z0 = robot.data.root_pos_w[:, 2].cpu().numpy()
    fallen = z0 < 0.45
    reached = np.zeros(n, dtype=bool)
    z_max = z0.copy()

    for _ in range(args_cli.steps):
        if isinstance(obs, dict):
            obs = obs["policy"]
        obs_np = obs.detach().cpu().numpy()
        obs_np = np.nan_to_num(obs_np)
        act = torch.as_tensor(policy(obs_np), dtype=torch.float32, device=env.unwrapped.device)
        obs, _, term, trunc, _ = env.step(act)
        z = robot.data.root_pos_w[:, 2].cpu().numpy()
        z_max = np.maximum(z_max, z)
        reached |= (z > args_cli.stand_z) & fallen

    n_fallen = int(fallen.sum())
    n_up = int((reached & fallen).sum())
    print("=" * 60)
    print(f"[getup] envs={n}  fallen_at_reset={n_fallen}  ({n_fallen / n * 100:.0f}%)")
    if n_fallen:
        print(f"[getup] stood_up = {n_up}/{n_fallen}  ({n_up / n_fallen * 100:.1f}%)")
    print(f"[getup] z_max(fallen) mean={z_max[fallen].mean() if n_fallen else float('nan'):.3f} m")
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

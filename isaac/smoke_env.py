"""Isaac Lab 环境的冒烟测试：建环境、打印机器人结构、跑几步。

    cd /home/yangg/data/isaacsim/IsaacLab
    env -u CONDA_PREFIX PYTHONEXE=./.venv/bin/python ./_isaac_sim/python.sh \
        <ws>/isaac/smoke_env.py --visualizer none --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Velocity-Flat-Nezha"


def main() -> int:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    print(f"[smoke] bodies ({len(robot.body_names)}): {robot.body_names}")
    print(f"[smoke] joints ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"[smoke] action space: {env.action_space}")
    print(f"[smoke] obs space: {env.observation_space}")

    for i in range(args_cli.steps):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, rew, terminated, truncated, extras = env.step(actions)
        if i % 25 == 0:
            root = robot.data.root_pos_w[0]
            print(f"[smoke] step {i:3d} base_z={root[2].item():.3f} rew={rew.mean().item():+.3f} "
                  f"term={terminated.sum().item()} trunc={truncated.sum().item()}")
    print("[smoke] OK")
    env.close()
    return 0


if __name__ == "__main__":
    import traceback

    code = 0
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    raise SystemExit(code)

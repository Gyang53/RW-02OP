"""⚠ 本方法失败，保留作为记录：把 PD 设为 0 后写关节角再单步，机体失稳/穿地，
knee > -0.86 时出现 NaN，best_hip 卡在边界 -1.00，腿长读数是垃圾。

不能在 Isaac 里用"关 PD + 单步"的方式做运动学标定。
可行替代：用 MuJoCo 对同一个缩放模型做正运动学（mj_forward）生成 (hip,knee)→轮位置
的精确表，再用该表生成关节空间参考轨迹喂给 Isaac。

---- 以下为原始说明 ----
"""纯运动学标定：关掉关节 PD，直接写关节角后单步读位姿。

与 diag_leg_map.py 的区别：不靠 PD 跟踪（实测偏差可达 0.4 rad），而是把刚度和阻尼
写成 0，再 `write_joint_state_to_sim` 写入精确关节角，单步物理后读 body 位姿。
单步 5 ms 内重力带来的关节漂移可忽略。

输出：
1. 密集 (hip,knee) → (Δx, Δz) 表（用于拟合）
2. 每个 knee 下「Δx≈0」的 hip（即腿竖直），以及该姿态的竖直腿长
   → 这张表就是"竖直伸展"轨迹的正确来源

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_leg_kinematics.py --visualizer none
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step", type=float, default=0.05, help="网格步长 (rad)")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    dev = unwrapped.device
    jn = robot.joint_names
    kidx = jn.index("knee_L_Joint")
    hidx = jn.index("hip_L_Joint")
    all_ids = torch.arange(len(jn), device=dev, dtype=torch.long)
    bidx_base = robot.body_names.index("base_link")
    bidx_wheel = robot.body_names.index("wheel_L_Link")

    # 关掉所有关节的 PD，避免写入的角度被驱动回去
    n = args_cli.num_envs
    robot.write_joint_stiffness_to_sim_index(
        stiffness=torch.zeros((n, len(jn)), device=dev), joint_ids=all_ids
    )
    robot.write_joint_damping_to_sim_index(
        damping=torch.zeros((n, len(jn)), device=dev), joint_ids=all_ids
    )

    def measure(hip: float, knee: float):
        q = robot.data.default_joint_pos.clone()
        q[:, hidx] = hip
        q[:, kidx] = knee
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        unwrapped.sim.step(render=False)
        m = robot.data.body_pos_w
        d = (m[:, bidx_wheel] - m[:, bidx_base]).to("cpu").numpy()[0]
        return float(d[0]), float(d[2])

    hips = np.arange(-1.0, 1.001, args_cli.step)
    knees = np.arange(-1.46, 0.664, args_cli.step)

    print("=" * 74)
    print("knee    best_hip   Δx(min|Δx|)   竖直腿长   机身高度(轮底触地)")
    print("-" * 74)
    table = []
    for knee in knees:
        best = None
        for hip in hips:
            dx, dz = measure(hip, knee)
            if best is None or abs(dx) < abs(best[1]):
                best = (hip, dx, dz)
        hip, dx, dz = best
        # 轮底触地时 base_z = |dz| + 轮半径
        base_h = abs(dz) + 0.127
        table.append((knee, hip, dx, abs(dz), base_h))
        print(f"{knee:+.2f}   {hip:+.2f}     {dx:+.3f}      {abs(dz):.3f}       {base_h:.3f}")

    print("-" * 74)
    if table:
        stroke = table[-1][4] - table[0][4]
        print(f"竖直伸展行程（最低→最高）= {stroke:.3f} m")
        print("可跳抬升上限 ≈ 行程 × 有效率；越障上限 ≈ 该抬升")
        print("建议轨迹（knee, hip）：")
        for k, h, _, _, bh in table[::4]:
            print(f"  knee={k:+.2f} hip={h:+.2f}  机身高={bh:.3f}")
    print("=" * 74)
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

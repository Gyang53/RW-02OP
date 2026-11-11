"""发射式跳跃的控制框架验证：假设已有发射机构，验证控制链路能否完成 0.6 m 跳跃并安全落地。

背景（docs 附录 D）：0.6 m + 10 kg 需要 177 J 起跳能量，靠腿部伸展不可能（12.3 倍差距），
必须外加发射机构（弹簧/气动，峰值约 8.4 g）。机械未就位前，本脚本用**冲量代理**验证
软件侧是否可行：在触发瞬间直接把竖直速度设为目标值（等价于发射机构做功），
之后完全交给策略做空中姿态保持与落地缓冲。

指标：
- 峰值高度（应接近目标）
- 空中姿态（g_z 是否保持 ≈ −1）
- 落地后是否摔倒（termination / 高度坍塌）
- 落地后能否恢复行走（再跑一段，看 vx 跟踪）

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/jump_launcher_sim.py --visualizer none \
      --checkpoint <ckpt> --height 0.6
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-Payload")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--height", type=float, default=0.6, help="目标跳跃高度 (m)")
parser.add_argument("--extra_kg", type=float, default=10.0)
parser.add_argument("--fwd_speed", type=float, default=0.0, help="起跳时保留的水平速度 (m/s)")
parser.add_argument("--settle_steps", type=int, default=150)
parser.add_argument("--flight_steps", type=int, default=200)
parser.add_argument("--recover_steps", type=int, default=300)
parser.add_argument("--buffer", type=int, default=0,
                    help="1=落地前伸腿缓冲（脚本姿态）。⚠ 实测比不缓冲更差（终止 23 vs 7），"
                         "默认关闭；姿态缓冲应由训练获得，而非脚本 keyframe")
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
    scale = (9.921 + args_cli.extra_kg) / 9.921
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (scale, scale)
    env_cfg.events.base_com = None
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
    n = args_cli.num_envs
    dt = env.unwrapped.step_dt

    jn = robot.joint_names
    idx_leg = [jn.index(x) for x in
               ("abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
                "abad_R_Joint", "hip_R_Joint", "knee_R_Joint")]
    # 伸展姿态（落地前，增大缓冲行程）与吸收姿态（触地后）
    EXTEND = [-0.18, 0.06, 0.00]     # abad, hip, knee（来自策略实测站姿）
    ABSORB = [-0.27, 0.82, -1.46]    # 策略实测深蹲姿
    LEG_SCALE = 0.25

    def pose_actions(pose3):
        a = torch.zeros(env.action_space.shape, device=dev)
        for side in (0, 3):
            for k in range(3):
                a[:, idx_leg[side + k]] = pose3[k] / LEG_SCALE
        return a

    def step(vx_cmd: float = 0.0, override=None):
        nonlocal obs
        cmd[:, 0] = vx_cmd
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
        hcmd[:, 0] = 0.50
        if isinstance(obs, dict):
            obs = obs["policy"]
        act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32, device=dev)
        if override is not None:
            act = override
        obs, _, term, trunc, _ = env.step(act)
        return int(term.sum().item())

    # 1) 稳定站立
    for _ in range(args_cli.settle_steps):
        step(0.0)
    z0 = robot.data.root_pos_w[:, 2].mean().item()
    gz0 = robot.data.projected_gravity_b[:, 2].mean().item()
    print(f"[launch] 起跳前: z={z0:.3f} g_z={gz0:+.2f}")

    # 2) 触发发射：直接给竖直速度（等价于发射机构做功）
    vz = math.sqrt(2 * 9.81 * args_cli.height)
    # 根速度是 6 维（线速度 3 + 角速度 3）
    vel = torch.zeros((n, 6), device=dev)
    vel[:, 0] = args_cli.fwd_speed
    vel[:, 2] = vz
    robot.write_root_velocity_to_sim(vel)
    print(f"[launch] 施加发射冲量: vz={vz:.2f} m/s（目标高度 {args_cli.height:.2f} m）")

    # 3) 飞行 + 落地
    z_max = z0
    gz_min, gz_max = gz0, gz0
    fell = 0
    land_step = None
    absorb_ticks = 0
    for i in range(args_cli.flight_steps):
        z_now = robot.data.root_pos_w[:, 2].mean().item()
        vz_now = robot.data.root_lin_vel_w[:, 2].mean().item()
        override = None
        if args_cli.buffer:
            descending = vz_now < -0.2 and z_now > z0 + 0.03
            if descending and land_step is None:
                override = pose_actions(EXTEND)      # 落地前：伸腿增大吸收行程
            elif land_step is not None and absorb_ticks < 25:
                override = pose_actions(ABSORB)      # 触地后：收腿吸能
                absorb_ticks += 1
        fell += step(args_cli.fwd_speed, override)
        z = robot.data.root_pos_w[:, 2].mean().item()
        gz = robot.data.projected_gravity_b[:, 2].mean().item()
        z_max = max(z_max, z)
        gz_min, gz_max = min(gz_min, gz), max(gz_max, gz)
        if land_step is None and i > 5 and z < z0 + 0.02:
            land_step = i
        if i % 20 == 0 or i == args_cli.flight_steps - 1:
            print(f"[launch]   step {i:3d} z={z:.3f} g_z={gz:+.2f} vx={robot.data.root_lin_vel_b[:, 0].mean().item():+.2f}")

    # 4) 落地后恢复行走
    vx_err = []
    for _ in range(args_cli.recover_steps):
        fell += step(1.0)
        vx_err.append((robot.data.root_lin_vel_b[:, 0] - 1.0).abs().mean().item())
    z_end = robot.data.root_pos_w[:, 2].mean().item()
    gz_end = robot.data.projected_gravity_b[:, 2].mean().item()

    print("=" * 66)
    print(f"[launch] 峰值高度 = {z_max:.3f} m（抬升 {z_max - z0:+.3f}，目标 {args_cli.height:.2f}）")
    print(f"[launch] 空中姿态 g_z 范围 = [{gz_min:+.2f}, {gz_max:+.2f}]（-1=竖直）")
    print(f"[launch] 落地检测步 = {land_step}")
    print(f"[launch] 终止(摔倒)事件 = {fell}")
    print(f"[launch] 落地后恢复: z={z_end:.3f} g_z={gz_end:+.2f} vx误差={np.mean(vx_err):.3f}")
    ok = (z_max - z0 > args_cli.height * 0.85) and (fell == 0) and (abs(gz_end + 1.0) < 0.2)
    print(f"[launch] 判定 = {'✅ 发射-飞行-落地-恢复 全链路通过' if ok else '❌ 未通过'}")
    if args_cli.buffer:
        print("[launch] 注意：脚本缓冲实测更差（见 docs 附录 D.3），建议 --buffer 0")
    print("=" * 66)
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

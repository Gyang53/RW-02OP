"""腿伸展诊断（有效版）：从策略的稳定蹲姿出发，PD 平缓伸展，分离"抬升"与"姿态变化"。

历史教训（都记录在 docs/TRAINING_GUIDE 6B.5e）：
1. 瞬移到极端蹲姿会穿地 → 被物理弹成躺姿，测到的是塌姿读数。
2. FK 求出的"腿竖直"蹲姿（knee −1.40 → hip +0.99）会把机身顶翻（hIp +0.99 过大），
   动力学上不可稳定站立。
3. 因此必须用**策略实测的稳定姿态**作为起点：knee ≈ −0.92、hip ≈ +0.40（z ≈ 0.47），
   并全程校验 projected_gravity_z < −0.8（否则中止，避免再次误测）。

输出：
- 蹲姿与伸展中的 base_z、竖直速度、projected_gravity_z（姿态）、轮接地比例
- 抬升量 vs 姿态变化量（判断能量是否转成了机体旋转）

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_leg_extension.py --visualizer none --num_envs 8
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--crouch_knee", type=float, default=-0.92, help="策略稳定蹲姿的 knee")
parser.add_argument("--crouch_hip", type=float, default=0.40, help="策略稳定蹲姿的 hip")
parser.add_argument("--extend_knee", type=float, default=0.663)
parser.add_argument("--extend_hip", type=float, default=-0.25)
parser.add_argument("--ramp_steps", type=int, default=20, help="伸展用多少个环境步（20 = 0.4 s）")
parser.add_argument("--hold_steps", type=int, default=20)
parser.add_argument("--sweep", action="store_true", help="扫掠 ramp_steps（伸展速度）")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEG_SCALE = 0.25


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    dev = env.unwrapped.device
    jn = robot.joint_names
    kidx = [jn.index("knee_L_Joint"), jn.index("knee_R_Joint")]
    hidx = [jn.index("hip_L_Joint"), jn.index("hip_R_Joint")]
    n = args_cli.num_envs

    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    bnames = list(sensor.body_names) if sensor is not None else []
    wsel = [bnames.index(b) for b in ("wheel_L_Link", "wheel_R_Link") if b in bnames]

    def act_of(knee: float, hip: float) -> torch.Tensor:
        a = torch.zeros(env.action_space.shape, device=dev)
        for i in kidx:
            a[:, i] = knee / LEG_SCALE
        for i in hidx:
            a[:, i] = hip / LEG_SCALE
        return a

    def snap() -> dict:
        gz = robot.data.projected_gravity_b[:, 2].mean().item()
        contact = 1.0
        if sensor is not None and len(wsel) == 2:
            f = sensor.data.net_forces_w
            if hasattr(f, "torch"):
                f = f.torch
            contact = (torch.linalg.norm(f[:, wsel], dim=-1) > 1.0).any(dim=1).float().mean().item()
        return {
            "z": robot.data.root_pos_w[:, 2].mean().item(),
            "vz": robot.data.root_lin_vel_w[:, 2].mean().item(),
            "gz": gz,
            "contact": contact,
        }

    # ---- 1) 站立稳定 ----
    for _ in range(20):
        env.step(torch.zeros(env.action_space.shape, device=dev))
    st = snap()
    print(f"[legdiag] 站立: z={st['z']:.3f} g_z={st['gz']:+.2f} 轮接地={st['contact']:.2f}")

    # ---- 2) PD 缓入策略稳定蹲姿 ----
    for _ in range(60):
        env.step(act_of(args_cli.crouch_knee, args_cli.crouch_hip))
    cr = snap()
    print(f"[legdiag] 蹲姿: z={cr['z']:.3f} g_z={cr['gz']:+.2f} 轮接地={cr['contact']:.2f}")
    if cr["gz"] > -0.8:
        print("[legdiag] ❌ 前置条件不满足：蹲姿不是竖直的（g_z≈0 表示躺倒），本次测量无效，中止。")
        env.close()
        return 2

    def run_once(ramp_steps: int):
        # 复位并重新建立蹲姿
        env.reset()
        for _ in range(20):
            env.step(torch.zeros(env.action_space.shape, device=dev))
        for _ in range(60):
            env.step(act_of(args_cli.crouch_knee, args_cli.crouch_hip))
        base = snap()
        z_m, vz_m, gz_lo, gz_hi, lost = base["z"], -9.9, base["gz"], base["gz"], None
        vz_last = 0.0
        for i in range(ramp_steps + args_cli.hold_steps):
            frac = min(1.0, (i + 1) / ramp_steps)
            knee = args_cli.crouch_knee + frac * (args_cli.extend_knee - args_cli.crouch_knee)
            hip = args_cli.crouch_hip + frac * (args_cli.extend_hip - args_cli.crouch_hip)
            env.step(act_of(knee, hip))
            sm = snap()
            z_m = max(z_m, sm["z"]); vz_m = max(vz_m, sm["vz"])
            gz_lo, gz_hi = min(gz_lo, sm["gz"]), max(gz_hi, sm["gz"])
            if lost is None and sm["contact"] < 0.5:
                lost = i
            if i == ramp_steps - 1:
                vz_last = sm["vz"]
        return base["z"], z_m, vz_m, vz_last, gz_lo, gz_hi, lost

    if args_cli.sweep:
        print("=" * 70)
        print(f"{'ramp_steps':>10} {'时长(ms)':>9} {'抬升(m)':>8} {'vz_peak':>9} {'vz_末伸展':>9} {'失接地步':>9}")
        print("-" * 70)
        for rs in (3, 5, 8, 12, 20):
            z0, zm, vzm, vzl, glo, ghi, lost = run_once(rs)
            print(f"{rs:>10} {rs * 20:>9} {zm - z0:>+8.3f} {vzm:>+9.3f} {vzl:>+9.3f} "
                  f"{(str(lost) if lost is not None else '未失'):>9}")
        print("=" * 70)
        env.close()
        return 0

    # ---- 3) 平缓伸展（knee/hip 同步插值）----
    z_max, vz_max = cr["z"], -9.9
    gz_min, gz_max = cr["gz"], cr["gz"]
    lost_at = None
    for i in range(args_cli.ramp_steps + args_cli.hold_steps):
        frac = min(1.0, (i + 1) / args_cli.ramp_steps)
        knee = args_cli.crouch_knee + frac * (args_cli.extend_knee - args_cli.crouch_knee)
        hip = args_cli.crouch_hip + frac * (args_cli.extend_hip - args_cli.crouch_hip)
        env.step(act_of(knee, hip))
        s = snap()
        z_max = max(z_max, s["z"])
        vz_max = max(vz_max, s["vz"])
        gz_min, gz_max = min(gz_min, s["gz"]), max(gz_max, s["gz"])
        if lost_at is None and s["contact"] < 0.5:
            lost_at = i
        if i % 5 == 0 or i == args_cli.ramp_steps + args_cli.hold_steps - 1:
            print(f"[legdiag]  step {i:2d} knee={knee:+.2f} hip={hip:+.2f} "
                  f"z={s['z']:.3f} vz={s['vz']:+.3f} g_z={s['gz']:+.2f} 轮接地={s['contact']:.2f}")

    end = snap()
    print("=" * 66)
    print(f"[legdiag] 抬升 = {z_max - cr['z']:+.3f} m（{cr['z']:.3f} → {z_max:.3f}）")
    print(f"[legdiag] 竖直速度峰值 = {vz_max:+.3f} m/s")
    print(f"[legdiag] 姿态范围 g_z = [{gz_min:+.2f}, {gz_max:+.2f}]（-1=完全竖直，0=水平）"
          f" → 姿态变化 {abs(gz_max - gz_min):.2f}")
    print(f"[legdiag] 轮失去接地发生在第 {lost_at} 步" if lost_at is not None
          else "[legdiag] 轮全程保持接地")
    print(f"[legdiag] 结束时 z={end['z']:.3f} g_z={end['gz']:+.2f}")
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

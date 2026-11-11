"""验证「蹲下 → 快速起立 → 收腿」跳跃原语在 S06 几何上的物理可行性。

用 MuJoCo 直接仿真（CPU，秒级）。执行器是 MJCF 里的 <motor>（力矩，ctrl 单位 N·m）：
腿用 PD 力矩跟踪 knee 轨迹，轮子用速度环保持滚动。

用法：
  python3 isaac/jump_primitive_check.py --sweep
  python3 isaac/jump_primitive_check.py --crouch -0.9 --extend_ms 100
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="assets/nezha/models/nezhav1/mjcf/mjmodel_s06.xml")
parser.add_argument("--crouch", type=float, default=-0.90, help="蓄力 knee 角 (rad)")
parser.add_argument("--extend", type=float, default=0.663, help="起跳伸展 knee 角 (rad)")
parser.add_argument("--tuck", type=float, default=-1.20, help="腾空收腿 knee 角 (rad)")
parser.add_argument("--kcrouch_ms", type=float, default=400.0)
parser.add_argument("--extend_ms", type=float, default=120.0)
parser.add_argument("--tuck_ms", type=float, default=180.0)
parser.add_argument("--wheel_target", type=float, default=10.0)
parser.add_argument("--kp", type=float, default=120.0)
parser.add_argument("--kd", type=float, default=3.0)
parser.add_argument("--sweep", action="store_true")
args = parser.parse_args()

LEGS_KNEE = ["knee_L_Joint", "knee_R_Joint"]
LEGS_HIP = ["hip_L_Joint", "hip_R_Joint"]
WHEELS = ["wheel_L_Joint", "wheel_R_Joint"]


def run(crouch, extend, extend_ms, kp=120.0, kd=3.0, kcrouch_ms=400.0, tuck=-1.20, tuck_ms=180.0, wheel_target=10.0):
    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def qa(n):
        return m.jnt_qposadr[jid(n)]

    def da(n):
        return m.jnt_dofadr[jid(n)]

    def ctrl_id(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.602
    mujoco.mj_forward(m, d)

    t_c, t_e, t_k = kcrouch_ms / 1000, extend_ms / 1000, tuck_ms / 1000
    T = t_c + t_e + t_k + 1.2

    def knee_traj(t):
        if t < t_c:
            return (t / t_c) * crouch
        t -= t_c
        if t < t_e:
            return crouch + (t / t_e) * (extend - crouch)
        t -= t_e
        if t < t_k:
            return extend + (t / t_k) * (tuck - extend)
        return tuck

    # 起飞判定：knee 伸到 extend 的瞬间
    log = []
    for i in range(int(T / m.opt.timestep)):
        t = i * m.opt.timestep
        kt = knee_traj(t)
        d.ctrl[:] = 0.0
        for n in LEGS_KNEE:
            q, v = d.qpos[qa(n)], d.qvel[da(n)]
            d.ctrl[ctrl_id(n)] = np.clip(kp * (kt - q) - kd * v, -120, 120)
        for n in LEGS_HIP:
            q, v = d.qpos[qa(n)], d.qvel[da(n)]
            d.ctrl[ctrl_id(n)] = np.clip(kp * (0.0 - q) - kd * v, -120, 120)
        for n in WHEELS:
            v = d.qvel[da(n)]
            d.ctrl[ctrl_id(n)] = np.clip(2.0 * (wheel_target - v), -40, 40)
        mujoco.mj_step(m, d)
        if i % 5 == 0:
            wheel_bottom = min(d.xpos[m.body(b).id][2] for b in ("wheel_L_Link", "wheel_R_Link")) - 0.127
            log.append((t, d.qpos[2], d.qvel[2], wheel_bottom, d.qpos[qa("knee_L_Joint")]))

    log = np.array(log)
    return log


def summarize(tag, log):
    stand = 0.602
    z, vz, wb = log[:, 1], log[:, 2], log[:, 3]
    print(
        f"{tag:36s} base_z峰值={z.max():.3f} 抬升={z.max()-stand:+.3f} "
        f"vz_max={vz.max():+.2f} 轮底最高={wb.max():+.3f} "
        f"{'✅可越0.15m' if wb.max() > 0.15 else '❌越不过'}"
    )


if args.sweep:
    print("crouch  extend_ms  kp  -> 结果")
    for crouch in (-0.6, -0.9, -1.2):
        for ems in (80, 120, 180):
            for kp in (80, 160):
                log = run(crouch, 0.663, ems, kp=kp)
                summarize(f"crouch={crouch:+.1f} ext={ems}ms kp={kp}", log)
else:
    log = run(args.crouch, args.extend, args.extend_ms, kp=args.kp, kd=args.kd,
              kcrouch_ms=args.kcrouch_ms, tuck=args.tuck, tuck_ms=args.tuck_ms,
              wheel_target=args.wheel_target)
    summarize(f"crouch={args.crouch:+.2f} ext={args.extend_ms:.0f}ms kp={args.kp:.0f}", log)

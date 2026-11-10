"""在纯 MuJoCo 里跑 Isaac 导出的策略（不经过 ROS），用于验证控制律/观测对齐。

    python3 isaac/sim_mujoco_policy.py --policy isaac/policies/nezha_flat_isaac.pt --seconds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

WS = Path(__file__).resolve().parents[1]
DEFAULT_XML = WS / "assets/nezha/models/nezhav1/mjcf/mjmodel.xml"

ACTUATOR_NAMES = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
]
LEG_SIM = [0, 1, 2, 4, 5, 6]
WHEEL_SIM = [3, 7]

LEG_POS_SCALE, WHEEL_VEL_SCALE = 0.25, 10.0
LEG_KP, LEG_KD, WHEEL_KP, WHEEL_KD = 40.0, 2.0, 0.0, 2.0


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v * (2 * w * w - 1) - 2 * w * np.cross(qv, v) + 2 * qv * np.dot(qv, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--policy_hz", type=float, default=50.0)
    ap.add_argument("--print_every", type=float, default=0.5)
    ap.add_argument("--delay", type=float, default=0.0, help="策略接管前的无控时长 (s)")
    ap.add_argument("--friction", type=float, default=None, help="覆盖接触摩擦系数")
    ap.add_argument("--dt", type=float, default=None, help="物理步长（默认用 MJCF 的 0.001）")
    ap.add_argument("--wheel_condim", type=int, default=None, help="轮子接触 condim")
    ap.add_argument("--wheel_friction", type=str, default=None, help="轮子摩擦 3 分量, 如 1.0,0.01,0.01")
    ap.add_argument("--yaw_trim_ki", type=float, default=0.0, help="偏航积分修正增益 (1/s)")
    ap.add_argument("--yaw_hold_kp", type=float, default=0.0, help="航向保持 P 增益 (rad/s per rad)")
    ap.add_argument("--wheel_half_width", type=float, default=None, help="轮子碰撞圆柱半宽 (m)")
    ap.add_argument("--cmd_tau", type=float, default=0.0, help="指令一阶低通时间常数 (s), 0=阶跃")
    ap.add_argument("--hold_first", type=float, default=0.0, help="先原地站 N 秒再给指令")
    ap.add_argument("--wheel_diff_limit", type=float, default=None, help="左右轮动作差上限")
    ap.add_argument("--wz_dither", type=float, default=0.0, help="wz 指令抖动幅度 (rad/s)")
    ap.add_argument("--wz_dither_period", type=float, default=1.0, help="抖动周期 (s)")
    ap.add_argument("--solref", type=float, default=None, help="接触 solref 时间常数")
    ap.add_argument("--solimp", type=str, default=None, help="接触 solimp a,b,w")
    ap.add_argument("--symmetrize", action="store_true", help="把右腿质量改成与左腿一致")
    ap.add_argument("--wheel_shape", default=None, choices=["capsule", "sphere", "ellipsoid"])
    ap.add_argument("--cone", default=None, help="接触锥模型: pyramidal / elliptic")
    ap.add_argument("--solver", default=None, help="solver: CG / Newton / PGS")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    if args.dt is not None:
        model.opt.timestep = args.dt
    if args.symmetrize:
        for side_l, side_r in [("hip_L_Link", "hip_R_Link"), ("knee_L_Link", "knee_R_Link")]:
            l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side_l)
            r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side_r)
            model.body_mass[r] = model.body_mass[l]
            model.body_inertia[r] = model.body_inertia[l]
    if args.solref is not None:
        model.geom_solref[:, 0] = args.solref
        model.geom_solref[:, 1] = 1.0
    if args.solimp is not None:
        model.geom_solimp[:, :3] = [float(v) for v in args.solimp.split(",")]
    if args.cone is not None:
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if args.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    if args.solver is not None:
        model.opt.solver = {"CG": mujoco.mjtSolver.mjSOL_CG, "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
                            "PGS": mujoco.mjtSolver.mjSOL_PGS}[args.solver]
    data = mujoco.MjData(model)
    policy = torch.jit.load(args.policy, map_location="cpu").eval()

    qadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(8)]
    dadr = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(8)]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if args.friction is not None:
        model.geom_friction[:, 0] = args.friction
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if "wheel" in body.lower():
            if args.wheel_shape is not None:
                model.geom_type[g] = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
                                      "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                                      "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[args.wheel_shape]
            if args.wheel_half_width is not None:
                model.geom_size[g, 1] = args.wheel_half_width
            if args.wheel_condim is not None:
                model.geom_condim[g] = args.wheel_condim
            if args.wheel_friction is not None:
                model.geom_friction[g] = [float(v) for v in args.wheel_friction.split(",")]

    decim = max(1, int(round(1.0 / (args.policy_hz * model.opt.timestep))))
    mujoco.mj_forward(model, data)

    mit = np.zeros((5, 8))  # tau, pos, vel, kp, kd
    yaw_trim = 0.0
    yaw_target = None
    cmd_vx = 0.0
    last_action = np.zeros(8)
    a = np.zeros(8)
    n_steps = int(args.seconds / model.opt.timestep)
    t_next_print = 0.0

    start_step = int(args.delay / model.opt.timestep)
    for k in range(n_steps):
        if k >= start_step and (k - start_step) % decim == 0:
            q = data.qpos[qadr].copy()
            qd = data.qvel[dadr].copy()
            quat = data.qpos[3:7].copy()
            rot = data.xmat[base_id].reshape(3, 3)
            lin_vel = rot.T @ data.qvel[0:3]
            ang_vel = rot.T @ data.qvel[3:6]
            vx_target = args.vx if data.time >= args.hold_first else 0.0
            if args.cmd_tau > 0.0:
                cmd_vx += (vx_target - cmd_vx) * (decim * model.opt.timestep) / args.cmd_tau
            else:
                cmd_vx = vx_target
            # 航向保持：把航向误差转成 wz 指令，交给策略去跟踪
            wz_cmd = args.wz
            if args.wz_dither > 0.0 and abs(args.wz) < 1e-6:
                wz_cmd = args.wz_dither * (1.0 if int(data.time / (0.5 * args.wz_dither_period)) % 2 == 0 else -1.0)
            if args.yaw_hold_kp > 0.0:
                w, qx, qy, qz = quat
                yaw_meas = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                if yaw_target is None:
                    yaw_target = yaw_meas
                if abs(args.wz) > 0.05:
                    yaw_target = yaw_meas
                else:
                    err = (yaw_target - yaw_meas + np.pi) % (2 * np.pi) - np.pi
                    wz_cmd = float(np.clip(args.yaw_hold_kp * err, -0.6, 0.6))
            obs = np.concatenate([
                lin_vel, ang_vel,
                quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])),
                np.array([cmd_vx, 0.0, wz_cmd]),
                q, qd, last_action,
            ]).astype(np.float32)
            with torch.no_grad():
                a = policy(torch.from_numpy(obs[None, :])).numpy()[0]
            # 偏航角速度跟踪的积分修正（消除左右轮差速带来的稳态漂移）
            wz_meas = float(ang_vel[2])
            if args.yaw_trim_ki > 0.0:
                yaw_trim += args.yaw_trim_ki * (wz_meas - args.wz) * (decim * model.opt.timestep)
                yaw_trim = float(np.clip(yaw_trim, -1.0, 1.0))
            a = a.copy()
            a[6] -= yaw_trim
            a[7] += yaw_trim
            if args.wheel_diff_limit is not None:
                diff = float(np.clip(a[6] - a[7], -args.wheel_diff_limit, args.wheel_diff_limit))
                mid = 0.5 * (a[6] + a[7])
                a[6] = mid + 0.5 * diff
                a[7] = mid - 0.5 * diff
            last_action = a
            mit[:] = 0.0
            mit[1, LEG_SIM] = LEG_POS_SCALE * a[0:6]
            mit[3, LEG_SIM] = LEG_KP
            mit[4, LEG_SIM] = LEG_KD
            mit[2, WHEEL_SIM] = WHEEL_VEL_SCALE * a[6:8]
            mit[3, WHEEL_SIM] = WHEEL_KP
            mit[4, WHEEL_SIM] = WHEEL_KD

        # 每个物理步都按 MIT PD 计算力矩
        q = data.qpos[qadr]
        qd = data.qvel[dadr]
        data.ctrl[:] = mit[0] + mit[3] * (mit[1] - q) + mit[4] * (mit[2] - qd)
        mujoco.mj_step(model, data)

        if data.time >= t_next_print:
            rpy = np.degrees(np.array([np.arctan2(2 * (data.qpos[3] * data.qpos[4] + data.qpos[5] * data.qpos[6]),
                                                 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)),
                                       np.arcsin(np.clip(2 * (data.qpos[3] * data.qpos[5] - data.qpos[6] * data.qpos[4]), -1, 1)),
                                       np.arctan2(2 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]),
                                                  1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2))]))
            print(f"t={data.time:5.2f}s z={data.qpos[2]:.3f} xy=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                  f"rpy=({rpy[0]:+6.1f},{rpy[1]:+6.1f},{rpy[2]:+7.1f}) "
                  f"act={np.round(a, 2)}")
            print(f"        q={np.round(data.qpos[qadr], 2)} quat={np.round(data.qpos[3:7], 3)}")
            t_next_print += args.print_every

    print(f"[done] final z={data.qpos[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""用 MuJoCo 正运动学生成「竖直伸展」轨迹表（可靠，不依赖 PD/接触）。

背景：在 Isaac 里做关节运动学标定两次都失败（PD 跟踪偏差 0.4 rad；关 PD 后机体失稳）。
而 MuJoCo 的 `mj_forward` 是纯解析 FK，给定关节角立即给出各 body 位姿，不受动力学影响。

做法：扫 (hip, knee)，对每个 knee 找使「轮子相对机身水平偏移 Δx≈0」的 hip，
并记录此时轮底触地所需的机身高度 = |Δz| + 轮半径。输出轨迹表供原语使用。

用法：
  python3 isaac/gen_vertical_trajectory.py            # 打印表格 + 写 JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import argparse as _ap
_ap_p = _ap.ArgumentParser()
_ap_p.add_argument("--model", default="assets/nezha/models/nezhav1/mjcf/mjmodel_s06.xml")
_ap_p.add_argument("--out", default="isaac/vertical_traj.json")
_ap_args = _ap_p.parse_args()
MODEL = _ap_args.model
OUT = Path(_ap_args.out)
WHEEL_R = 0.127

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)


def jnt(name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


QADDR = {n: m.jnt_qposadr[jnt(n)] for n in ("hip_L_Joint", "knee_L_Joint", "abad_L_Joint")}
BID_BASE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
BID_WHEEL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_L_Link")


def wheel_rel(hip: float, knee: float, abad: float = 0.0) -> tuple[float, float]:
    """返回轮相对机身的 (Δx, Δz)（世界系，机身姿态水平）。"""
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.0  # 任意高度，只看相对量
    d.qpos[QADDR["hip_L_Joint"]] = hip
    d.qpos[QADDR["knee_L_Joint"]] = knee
    d.qpos[QADDR["abad_L_Joint"]] = abad
    mujoco.mj_forward(m, d)
    p_w = d.xpos[BID_WHEEL]
    p_b = d.xpos[BID_BASE]
    return float(p_w[0] - p_b[0]), float(p_w[2] - p_b[2])


hips = np.arange(-1.0, 1.001, 0.02)
knees = np.arange(-1.46, 0.664, 0.02)

rows = []
print("=" * 70)
print(f"{'knee':>7} {'best_hip':>9} {'Δx':>7} {'竖直腿长':>9} {'机身高度':>9}")
print("-" * 70)
for knee in knees:
    best = None
    for hip in hips:
        dx, dz = wheel_rel(hip, knee)
        if best is None or abs(dx) < abs(best[1]):
            best = (hip, dx, dz)
    hip, dx, dz = best
    leg_v = abs(dz)
    body_h = leg_v + WHEEL_R  # 轮底触地时的机身高度
    rows.append({"knee": round(float(knee), 3), "hip": round(float(hip), 3),
                 "dx": round(dx, 4), "leg_vert": round(leg_v, 4), "body_h": round(body_h, 4)})
    if abs((knee * 100) % 10) < 1e-6:  # 每 0.1 rad 打印一次
        print(f"{knee:>7.2f} {hip:>9.2f} {dx:>7.3f} {leg_v:>9.3f} {body_h:>9.3f}")
print("-" * 70)

body_hs = [r["body_h"] for r in rows]
stroke = max(body_hs) - min(body_hs)
print(f"竖直伸展行程 = {stroke:.3f} m（机身高度 {min(body_hs):.3f} → {max(body_hs):.3f}）")
print(f"按此前实测有效率 ≈0.37 估算，可跳抬升上限 ≈ {0.37 * stroke:.3f} m")
print("=" * 70)

OUT.write_text(json.dumps(rows, indent=1))
print(f"[gen] 已写出 {OUT}（{len(rows)} 个采样点）")

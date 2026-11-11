"""按比例缩短 nezhav1 的大腿/小腿，生成新的 MJCF（保留环境地面/相机/灯光）。

只缩放「腿段」：hip→knee（大腿）与 knee→wheel（小腿）的 body 偏移、对应碰撞体、
以及段内惯性；**关节轴、限位、轮半径、机身（base_link）全部不变**。

用法：
  python3 isaac/make_scaled_mjcf.py --scale 0.6 \
      --src assets/nezha/models/nezhav1/mjcf/mjmodel.xml \
      --dst assets/nezha/models/nezhav1/mjcf/mjmodel_s06.xml

注意：
- 输出文件必须保留 worldbody 里的 floor/camera/light，否则 MuJoCo 侧没有地面，
  机器人会直接穿过地面（曾因用 make_clean_mjcf 就地覆盖源文件踩过这个坑）。
- 给 Isaac 用的清理版请另行用 NEZHA_MJCF / NEZHA_CLEAN_NAME 生成到**不同文件名**。
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, default=0.6)
parser.add_argument("--src", default="assets/nezha/models/nezhav1/mjcf/mjmodel.xml")
parser.add_argument("--dst", default="assets/nezha/models/nezhav1/mjcf/mjmodel_s06.xml")
args = parser.parse_args()

S = args.scale
tree = ET.parse(args.src)
root = tree.getroot()


def scaled(txt: str, s: float) -> str:
    x, y, z = map(float, txt.split())
    return f"{x * s:.5f} {y * s:.5f} {z * s:.5f}"


def scale_inertial(body: ET.Element, s: float) -> None:
    """COM 位置 ×s，转动惯量 ×s²（长度平方标度，质量不变）。"""
    ine = body.find("inertial")
    if ine is None:
        return
    ine.set("pos", scaled(ine.get("pos"), s))
    di = [float(v) * s * s for v in ine.get("diaginertia").split()]
    ine.set("diaginertia", " ".join(f"{v:.8f}" for v in di))


for side in ("L", "R"):
    prefix = "rl" if side == "L" else "rr"
    abad = root.find(f".//body[@name='abad_{side}_Link']")
    hip = abad.find(f"body[@name='hip_{side}_Link']")
    knee = hip.find(f"body[@name='knee_{side}_Link']")
    wheel = knee.find(f"body[@name='wheel_{side}_Link']")

    # 大腿段：hip 关节 → knee 关节的偏移
    knee.set("pos", scaled(knee.get("pos"), S))
    # 小腿段：knee 关节 → 轮轴 的偏移
    wheel.set("pos", scaled(wheel.get("pos"), S))

    # 碰撞体：随所属段缩放（位置 + 圆柱半长）
    hc = hip.find(f"geom[@name='hip_{side}_collision']")
    if hc is not None:
        hc.set("pos", scaled(hc.get("pos"), S))
        sz = hc.get("size").split()
        sz[1] = f"{float(sz[1]) * S:.4f}"
        hc.set("size", " ".join(sz))
    kc = knee.find(f"geom[@name='knee_{side}_collision']")
    if kc is not None:
        kc.set("pos", scaled(kc.get("pos"), S))
        sz = kc.get("size").split()
        sz[1] = f"{float(sz[1]) * S:.4f}"
        kc.set("size", " ".join(sz))

    # 惯性：大腿/小腿段缩放；轮子只缩偏移（轮半径与质量不变）
    scale_inertial(hip, S)
    scale_inertial(knee, S)
    wi = wheel.find("inertial")
    if wi is not None:
        wi.set("pos", scaled(wi.get("pos"), S))

    # 视觉 mesh：大腿/小腿网格按比例缩放（MJCF mesh 支持 scale 属性）
    for mesh_name in (f"{prefix}_hip", f"{prefix}_knee"):
        for m in root.findall(f".//mesh[@name='{mesh_name}']"):
            m.set("scale", f"{S} {S} {S}")

# 初始站立高度：knee=0 时轮底触地所需的 base z
# 几何：abad(-0.1138,0.105,-0.1115) + hip(0.0813,0.0328,0) + thigh + shank，轮半径 0.127
import math

thigh_z = -0.30311 * S
shank_z = -0.30199 * S
wheel_bottom_rel = -0.1115 + thigh_z + shank_z - 0.127
base_z = -wheel_bottom_rel
base = root.find(".//body[@name='base_link']")
base.set("pos", f"0 0 {base_z:.3f}")

tree.write(args.dst, encoding="UTF-8", xml_declaration=True)
print(f"[scale] {args.src} -> {args.dst}  scale={S}  base_z={base_z:.3f}")

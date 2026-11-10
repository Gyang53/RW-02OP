"""把 mjmodel.xml 清理成可直接给 Isaac Lab 用的 MJCF。

原始 MJCF 的 worldbody 里除了机器人还有两个额外 body：
  - slope        环境斜坡（固定在世界的障碍物）
  - base_Estimate 可视化参考体（free joint）
它们会被 MJCF importer 当成 articulation 的一部分，必须先删掉。
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import os

SRC = Path(
    os.environ.get(
        "NEZHA_MJCF",
        Path(__file__).resolve().parents[1] / "assets/nezha/models/nezhav1/mjcf/mjmodel.xml",
    )
)
DST = SRC.with_name(
    os.environ.get("NEZHA_CLEAN_NAME", "nezha_isaac.xml")
)
DROP = {"slope", "base_Estimate"}


def main() -> int:
    tree = ET.parse(SRC)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        print("no <worldbody>", file=sys.stderr)
        return 1

    removed = []
    for body in list(worldbody.findall("body")):
        if body.get("name") in DROP:
            worldbody.remove(body)
            removed.append(body.get("name"))

    # worldbody 里的 floor(plane)/camera/light 属于场景，不属于机器人；
    # 转成 USD 后会变成 articulation 里的静态碰撞体，把车弹飞。
    for tag in ("geom", "camera", "light"):
        for node in list(worldbody.findall(tag)):
            worldbody.remove(node)
            removed.append(node.get("name") or tag)

    tree.write(DST, encoding="UTF-8", xml_declaration=True)
    print(f"[clean] {SRC.name} -> {DST.name}, removed: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

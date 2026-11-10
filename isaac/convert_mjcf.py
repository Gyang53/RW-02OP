"""用 Isaac Lab 的 MjcfConverter 把 nezha MJCF 转成带物理的 USD。

    cd /home/yangg/data/isaacsim/IsaacLab
    ./.venv/bin/python <ws>/isaac/convert_mjcf.py --headless

产出：<ws>/assets/nezha/models/nezhav1/mjcf/nezha_isaac/nezha_isaac.usda
      （payloads/Physics/{physics,physX,mujoco}.usda 里带关节/质量/ArtRoot）
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

WS = Path(__file__).resolve().parents[1]
# NEZHA_MJCF 可指向缩放版 MJCF（如 mjmodel_s06.xml），转换产物按 xml 文件名命名：
#   mjmodel.xml     -> nezha_isaac.xml -> nezha_isaac/nezha_isaac.usda
#   mjmodel_s06.xml -> mjmodel_s06.xml 的清理版 -> mjmodel_s06/…usda（不覆盖原模型）
_src_name = os.environ.get("NEZHA_MJCF_NAME", "nezha_isaac.xml")
SRC = WS / "assets/nezha/models/nezhav1/mjcf" / _src_name
USD_DIR = SRC.parent


def main() -> int:
    if not SRC.exists():
        raise FileNotFoundError(f"先运行 isaac/make_clean_mjcf.py 生成 {SRC}")
    cfg = MjcfConverterCfg(
        asset_path=str(SRC),
        usd_dir=str(USD_DIR),
        force_usd_conversion=True,
        collision_from_visuals=True,
        collision_type="Convex Hull",
        self_collision=False,
        fix_base=False,
        robot_type="Wheeled",
        run_asset_transformer=True,
    )
    converter = MjcfConverter(cfg)
    print(f"[convert] usd = {converter.usd_path}")
    _move_articulation_root(Path(converter.usd_path))
    return 0


def _move_articulation_root(usd_path: Path) -> None:
    """把 ArticulationRootAPI 从 base_link 挪到机器人根 prim。

    Isaac Lab 的 ArticulationCfg.prim_path 必须是 articulation root，
    而 MJCF importer 把 root 标在了 base_link 上，spawn 到 {ENV_REGEX_NS}/Robot 后
    路径就对不上了。这里改成标在 USD 的 defaultPrim（/nezha_v1）上。
    """
    physics_layer = usd_path.parent / "payloads/Physics/physics.usda"
    old = (
        'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", '
        '"PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]'
    )
    new = 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]'
    text = physics_layer.read_text()
    if old in text:
        physics_layer.write_text(text.replace(old, new))

    root_layer = usd_path
    text = root_layer.read_text()
    marker = 'def Xform "nezha_v1" (\n    prepend references'
    if marker in text and "PhysicsArticulationRootAPI" not in text.split(marker)[0][-400:]:
        text = text.replace(
            marker,
            'def Xform "nezha_v1" (\n'
            '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]\n'
            '    prepend references',
        )
        root_layer.write_text(text)
    print(f"[convert] articulation root -> {usd_path.name} defaultPrim")


if __name__ == "__main__":
    raise SystemExit(main())

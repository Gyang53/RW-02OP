# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 载荷训练任务：整机 20 → 30 kg（额外 0~10 kg），并随机化载荷质心。

需求来源：用户要求"承载能力至高 10 kg，且稳定平衡运行"。
做法：
- 机身质量乘子 1.25 → **2.0**（base 9.92 kg → 最多 19.8 kg，整机约 30 kg）；
  下限保留 0.85，保证轻载情形仍在训练分布里。
- 载荷质心随机：`com_range` 在 x/y/z 上扩大（x ±0.10、y ±0.10、z −0.05~+0.12），
  模拟货物放在机身不同位置带来的偏心。
- 其余保持与 S06 平地任务一致（轮式行走约束、高度指令 0.35–0.60）。

注意：这改变的是**训练分布**，不改变真机能力；真机 10 kg 载荷下
轮扭矩需求仅 ~2.3 N·m（远低于 90 N·m 上限），承载行走在物理上是宽裕的。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .flat_env_s06_cfg import NezhaFlatS06EnvCfg


@configclass
class NezhaPayloadS06EnvCfg(NezhaFlatS06EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- 载荷质量：整机 20 kg → 最高 ~30 kg ----
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                # 倍数：9.92 kg × 2.0 ≈ 19.8 kg；再加腿/轮 ≈ 30 kg 整机
                "mass_distribution_params": (0.85, 2.0),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )

        # ---- 载荷质心随机（偏心货物） ----
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (-0.05, 0.12)},
            },
        )

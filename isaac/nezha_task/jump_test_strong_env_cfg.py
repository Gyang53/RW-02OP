# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""跳跃验证（强执行器版）：只改腿的 PD 增益/力矩上限，几何不动。

用途：判定「跳不起来」是否纯粹是执行器受限。
S06 站立 0.602 m 已接近全伸 0.629 m，超过站高的行程只有 2.7 cm；
若把 stiffness 40→150、effort 120→250 后弹射能出现明显弹道阶段（base_z > 0.65），
说明瓶颈在执行器而不是几何；否则几何（腿长/行程）才是硬约束。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .jump_test_s06_env_cfg import NezhaJumpTestS06EnvCfg


@configclass
class NezhaJumpTestStrongEnvCfg(NezhaJumpTestS06EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        legs = self.scene.robot.actuators["legs"]
        legs.stiffness = 150.0
        legs.damping = 6.0
        legs.effort_limit_sim = 250.0
        legs.velocity_limit_sim = 40.0

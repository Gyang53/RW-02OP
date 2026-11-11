# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""跳跃验证（真机力矩版）：effort 保持 90 N·m，只提高 PD 刚度。

依据：90 N·m 是电机硬上限（60 A × kt 2.1），但 **PD 刚度是控制参数，可以自由选**。
原来 stiffness=40 时，要让力矩打到 90 必须给超出限位的目标（+2.0 rad），
结果腿在离地前撞关节限位（实测膝角 +1.14 vs 限位 +0.663），Isaac 的限位约束
把冲量吸收掉，base_z 只到 0.604。

本配置：stiffness 40→150、damping 2→8，effort 仍为 90。
伸展目标设到限位 +0.66（不再饱和到 +2.0），使力矩在整个行程内保持高值、
在接近限位时自然衰减，避免撞击限位。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .jump_test_s06_env_cfg import NezhaJumpTestS06EnvCfg


@configclass
class NezhaJumpTestHw90EnvCfg(NezhaJumpTestS06EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        legs = self.scene.robot.actuators["legs"]
        # 保持训练值：高刚度只在跳跃窗口内由测试脚本按环境写入（见 jump_primitive_test.py）
        legs.stiffness = 40.0
        legs.damping = 2.0
        # effort_limit_sim 保持 90（nezha_s06_cfg 已对齐真机），此处显式重申
        legs.effort_limit_sim = 90.0

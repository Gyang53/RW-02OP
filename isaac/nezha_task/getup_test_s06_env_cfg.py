# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""起身原语验证任务：轮电机能力对齐真机（90 N·m）+ 提高速度环增益。

背景：`nezha_s06_cfg.py` 里 wheels 的 `effort_limit_sim=40`、`damping=2.0` 是自设值，
但真机 8 个电机全部是 90 N·m（60 A × kt 2.1）。速度环 damping=2 时，轮子很快达到
目标转速，扭矩随即塌到 0 —— 原语需要的「持续~45 N·m 翻转扭矩」根本给不出来
（实测仰卧起身 z 只到 0.43，两次符号都失败）。

本配置只改轮子：effort 40→90（对齐真机）、damping 2→20（大误差下更快饱和，翻转时
能持续输出接近上限的扭矩）。腿的执行器不动。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .getup_s06_v5_env_cfg import NezhaGetUpS06V5EnvCfg


@configclass
class NezhaGetUpTestS06EnvCfg(NezhaGetUpS06V5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        wheels = self.scene.robot.actuators["wheels"]
        wheels.effort_limit_sim = 90.0   # 真机同款电机上限
        wheels.damping = 20.0            # 速度环增益提高 → 翻转时能维持扭矩

        # ⚠ 物理步长：默认 sim.dt=0.005（5 ms）、decimation=4（控制 50 Hz）。
        # MuJoCo 侧验证起身/弹射用的是 0.5 ms —— 步长差 10 倍，瞬时冲量会被抹平，
        # 这很可能是"MuJoCo 能站起、Isaac 站不起"的主因。
        # 改为 dt=0.00125、decimation=16：控制周期仍 20 ms（50 Hz），物理细化 4 倍。
        self.sim.dt = 0.00125
        self.decimation = 16

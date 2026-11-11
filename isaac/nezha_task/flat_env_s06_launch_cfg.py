# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 发射/落地训练任务：周期性被弹飞 + 学习落地缓冲。

需求背景（用户）：目标跳 0.6 m、承载 10 kg、稳定平衡。
- 0.6 m 需要 177 J 发射能量（现有腿伸展仅 14.4 J）→ 机械需加发射机构；
- 软件侧已验证：给到 3.43 m/s 发射速度后能上升 0.59 m 并保持姿态，
  但落地会摔（7/16），脚本姿态缓冲更差（23/16）→ **落地必须靠训练**。

本任务在载荷任务（`S06-Payload`，整机 20→30 kg）之上加：
1. 每 6~9 s 给 25% 环境施加 2.0~3.6 m/s 的竖直发射冲量（对应 0.2~0.66 m 跳跃）；
2. 奖励：落地后"直立 + 轮子接地 + 竖直速度收敛"；
3. 惩罚：被弹飞后姿态明显倾斜（g_z > −0.5）。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .flat_env_s06_payload_cfg import NezhaPayloadS06EnvCfg
from .mdp import launch as lc


@configclass
class NezhaLaunchS06EnvCfg(NezhaPayloadS06EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 周期性发射（interval 事件），概率与速度范围见 mdp/launch.py
        self.events.launch_impulse = EventTerm(
            func=lc.apply_launch_impulse,
            mode="interval",
            interval_range_s=(6.0, 9.0),
            params={"probability": 0.25, "vz_range": (2.0, 3.6), "vx_range": (-0.6, 0.8)},
        )

        # 落地稳定奖励 + 倾斜惩罚
        self.rewards.landing_stability = RewTerm(
            func=lc.landing_stability_reward, weight=3.0, params={"vz_tol": 1.0}
        )
        self.rewards.launch_tilt = RewTerm(func=lc.launch_tilt_penalty, weight=-2.0)

        # 回合延长，给"发射→飞行→落地→恢复"留时间
        self.episode_length_s = 30.0

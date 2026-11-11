# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 v5：起身专项（在 v4 轮式行走+越障策略之上）。

v3 里起身区只占 12.5% 环境、且与越障共用梯度 → 起身率 0%（z_max 0.529，起不来）。
本任务把分布收窄到「起身」：

- 分区：50% 起身区（每回合都从俯卧/仰卧/倾倒起步）/ 50% 纯行走区（保持轮式步态）。
- 无越障奖励与障碍物（停到 100 m 外），避免与起身动作竞争。
- 起身奖励：track_height 权重 2.0 / std 0.25（倒地时 z≈0.30、目标 0.60，
  std 过小会导致 exp 梯度消失——v1 文档已记录这个坑）。
- 轮式约束项保留：`_walking_window` 要求双轮接地，倒地环境自动免疫，
  不会阻止起身动作；起身完成后回到轮式行走即可继续拿分。
- 终止：nan_state + 翻倒持续（起身回合给 6 s 宽限）。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import torch

from .jump_s06_v4_env_cfg import NezhaJumpS06V4EnvCfg, _respawn_obstacle_jump
from .mdp import mixed_reset, terminations_s06_v2 as term2


@configclass
class NezhaGetUpS06V5EnvCfg(NezhaJumpS06V4EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- 分区：50% 起身 / 50% 行走；障碍物全部停远 ----
        self.events.assign_groups = EventTerm(
            func=_assign_getup_groups, mode="startup", params={}
        )
        self.events.reset_fallen = EventTerm(
            func=mixed_reset.reset_mixed,
            mode="reset",
            params={"z_range": (0.28, 0.42), "tilt_range": (0.9, 1.6), "joint_noise": 0.10, "extended_only": True},
        )
        self.events.respawn_obstacle = EventTerm(
            func=_park_obstacle, mode="reset", params={}
        )

        # ---- 起身奖励：高度跟踪加大权重、放宽容差 ----
        # v1 文档的成功配方：std 必须放大 —— 倒地时 z≈0.2、目标 0.60，
        # std=0.25 时 exp(-(0.4/0.25)^2)=0.077 仍偏小；0.45 时 ≈0.45，梯度可用。
        self.rewards.track_height.weight = 2.0
        self.rewards.track_height.params["std"] = 0.45
        # 越障相关奖励在小场景里没有意义，关掉以免噪声
        self.rewards.jump_takeoff.weight = 0.0
        self.rewards.obstacle_cleared.weight = 0.0
        self.rewards.crouch_approach.weight = 0.0
        self.rewards.air_retract.weight = 0.0
        # 起身稠密奖励：低姿态下的上升速度 + 腿伸展（给探索连续梯度）
        from .mdp import wheel_rewards as wr
        self.rewards.getup_lift = RewTerm(func=wr.getup_lift_reward, weight=3.0,
                                         params={"z_low": 0.45, "knee_ext": 0.3})

        # 倒地回合没有轮子接地，腿静止/abad 惩罚已被行走窗口门控，无需改

        # 起身回合给更长宽限（从倒地到站起需要 >3 s，v1 文档记录 10 s 量级）
        self.terminations.base_dragged = DoneTerm(
            func=term2.base_dragged,
            params={"getup_grace_s": 10.0, "tilt_threshold": 0.6, "sustain_s": 0.3},
        )
        self.episode_length_s = 20.0


def _assign_getup_groups(env, env_ids):
    """startup：i%2==0 为起身区（每回合倒地起步），其余为纯行走区。"""
    n = env.scene["robot"].num_instances
    idx = torch.arange(n, device=env.device)
    env.getup_mask = (idx % 2) == 0
    env.obstacle_mask = torch.zeros(n, dtype=torch.bool, device=env.device)
    env.is_fallen_episode = torch.zeros(n, dtype=torch.bool, device=env.device)


def _park_obstacle(env, env_ids):
    """把障碍物停到 100 m 外（本任务不练越障）。"""
    return _respawn_obstacle_jump(
        env, env_ids, ahead=100.0, h_min=0.04, h_max=0.04, ramp_steps=1, warmup_steps=0
    )

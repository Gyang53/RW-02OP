# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 跳跃验证任务：障碍高度固定 0.22 m（轮半径 0.127 m 的 ~1.7 倍）。

存在意义：V4 的障碍高度是 0.04→0.20 m 课程，实测机器人直接滚过/顶过，
不需要腾空，因而无法验证跳跃原语。本任务把障碍固定在 0.22 m、无 warmup，
形成一个必须腾空才能通过的台阶，用于 A/B 对比：
  A. 纯策略（不注入原语）
  B. 策略 + 跳跃原语（`isaac/jump_primitive.py`）
"""

from __future__ import annotations

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .jump_s06_v4_env_cfg import NezhaJumpS06V4EnvCfg, _respawn_obstacle_jump

OBSTACLE_H = 0.18


@configclass
class NezhaJumpTestS06EnvCfg(NezhaJumpS06V4EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 所有环境都放障碍（便于统计），高度固定 0.22 m、无课程、无 warmup
        self.events.assign_groups = EventTerm(func=_assign_all_obstacle, mode="startup", params={})
        self.events.respawn_obstacle = EventTerm(
            func=_respawn_fixed,
            mode="reset",
            params={"ahead": 1.2, "height": OBSTACLE_H},
        )
        # 验证时不注入倒地起步
        self.events.reset_fallen = EventTerm(func=_reset_clean, mode="reset", params={})

        # ⚠ 固定初始位姿：默认 reset_base 的 yaw 是 (±3.14) 随机、位置 ±0.5 m 随机，
        # 导致一半环境背对障碍（世界 −x 前进），越障统计完全失真（实测平均前进 −3 m）。
        # 这里把 x/y/yaw 与初始速度全部固定为 0，让"障碍在前方 1.2 m"成立。
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # 关掉越障后的自动重摆（会污染越障计量）——改为本任务只做固定摆位
        self.rewards.obstacle_cleared = None
        self.rewards.jump_takeoff.weight = 0.0
        self.rewards.crouch_approach.weight = 0.0
        self.episode_length_s = 30.0


def _assign_all_obstacle(env, env_ids):
    n = env.scene["robot"].num_instances
    env.obstacle_mask = torch.ones(n, dtype=torch.bool, device=env.device)
    env.getup_mask = torch.zeros(n, dtype=torch.bool, device=env.device)
    env.is_fallen_episode = torch.zeros(n, dtype=torch.bool, device=env.device)


def _reset_clean(env, env_ids, probability: float | None = None):
    env.is_fallen_episode[env_ids] = False
    if hasattr(env, "_episode_steps_elapsed"):
        env._episode_steps_elapsed[env_ids] = 0


def _respawn_fixed(env, env_ids, ahead: float = 1.2, height: float = OBSTACLE_H):
    """障碍固定摆在机器人前方 ahead 米，高度固定（无课程）。"""
    return _respawn_obstacle_jump(
        env,
        env_ids,
        ahead=ahead,
        h_min=height,
        h_max=height,
        ramp_steps=1,
        warmup_steps=0,
    )

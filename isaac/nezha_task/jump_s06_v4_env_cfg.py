# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 v4：跳跃专项微调（在 v3 全能策略之上）。

v3 已把轮式行走（双轮接地 98%）、蹲站、起身练稳；剩下最难的是"遇障碍蹲下爆发起跳"。
本任务把分布收窄到「越障」这一件事，避免行走/起身样本稀释跳跃梯度：

- 分区：50% 越障区（障碍在前方 1.2 m）/ 50% 纯行走区（障碍停 100 m 外）。
  行走区保留是为了不遗忘轮式步态（防止只练跳导致行走退化）。
- 不注入倒地起步（越障与起身互斥，起身已在 v3 学会）。
- 障碍高度课程 0.04 → 0.20 m（warmup 结束才开始升温）。
- 起跳奖励加权（3.0 → 6.0）、越障一次性奖励 8.0 → 15.0，
  并保留 v3 的「障碍前蹲低引导」。
- 保留 v3 的全部轮式行走约束（轮接地/轮驱动/腿静止/abad 中立/趴地惩罚），
  跳跃窗口由 `_walking_window` 自动豁免，不会压制起跳动作。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import torch

from .full_s06_v3_env_cfg import NezhaFullS06V3EnvCfg
from .mdp import mixed_reset
from .mdp import wheel_rewards as wr


@configclass
class NezhaJumpS06V4EnvCfg(NezhaFullS06V3EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- 分区：50% 越障 / 50% 行走，无起身区 ----
        self.events.assign_groups = EventTerm(
            func=_assign_jump_groups, mode="startup", params={}
        )
        self.events.reset_fallen = EventTerm(
            func=_reset_no_fallen, mode="reset", params={}
        )
        self.events.respawn_obstacle = EventTerm(
            func=_respawn_obstacle_jump,
            mode="reset",
            params={"ahead": 1.2, "h_min": 0.04, "h_max": 0.20, "ramp_steps": 12000, "warmup_steps": 3000},
        )

        # ---- 跳跃奖励加权 ----
        self.rewards.jump_takeoff.weight = 6.0
        self.rewards.jump_takeoff.params["target_vz"] = 1.2
        self.rewards.obstacle_cleared.weight = 15.0
        self.rewards.obstacle_cleared.params["ahead"] = 1.2
        self.rewards.obstacle_cleared.params["height_range"] = (0.04, 0.20)
        self.rewards.crouch_approach.weight = 1.5
        self.rewards.crouch_approach.params["dist"] = 0.8
        self.rewards.air_retract.weight = 0.8


def _assign_jump_groups(env, env_ids):
    """startup：i%2==0 为越障区，其余为纯行走区；不设起身区。"""
    n = env.scene["robot"].num_instances
    idx = torch.arange(n, device=env.device)
    env.obstacle_mask = (idx % 2) == 0
    env.getup_mask = torch.zeros(n, dtype=torch.bool, device=env.device)
    env.is_fallen_episode = torch.zeros(n, dtype=torch.bool, device=env.device)


def _reset_no_fallen(env, env_ids, probability: float | None = None):
    """复位事件：不注入倒地姿态（仅清标志）。

    `probability` 参数是为评测脚本兼容保留的（eval_skills 会尝试写倒地概率），此处忽略。
    """
    env.is_fallen_episode[env_ids] = False
    if hasattr(env, "_episode_steps_elapsed"):
        env._episode_steps_elapsed[env_ids] = 0


def _respawn_obstacle_jump(
    env,
    env_ids,
    ahead: float = 1.2,
    h_min: float = 0.04,
    h_max: float = 0.20,
    ramp_steps: int = 12000,
    warmup_steps: int = 3000,
):
    """障碍物摆放 + 高度课程（课程从 warmup 结束开始升温）。"""
    obs = env.scene["obstacle"]
    step = int(getattr(env, "_sim_step_counter", 0))
    progress = min(max((step - warmup_steps) / max(ramp_steps, 1), 0.0), 1.0)
    hi = h_min + (h_max - h_min) * progress
    h = torch.empty(len(env_ids), device=env.device).uniform_(h_min, hi)

    origins = env.scene.env_origins[env_ids]
    pose = obs.data.default_root_state[env_ids, :7].clone()
    omask = getattr(env, "obstacle_mask", None)
    active = omask[env_ids] if omask is not None else torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    pose[:, 0] = origins[:, 0] + torch.where(active, torch.full_like(h, ahead), torch.full_like(h, 100.0))
    pose[:, 1] = origins[:, 1]
    pose[:, 2] = torch.where(active, h / 2, torch.full_like(h, 0.08))
    obs.write_root_pose_to_sim(pose, env_ids=env_ids)
    if hasattr(env, "_pre_jump_air"):
        env._pre_jump_air[env_ids] = False
        env._was_behind[env_ids] = True

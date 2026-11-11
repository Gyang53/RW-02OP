# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 混合模式复位：把「行走环境」与「起身环境」在环境维度上永久分开。

问题背景（实测）：
  Full 任务用「每次复位按概率倒地」的方式混训，倒地概率 50% 时，
  纯轮式行走策略（双轮接地 98.5%、腿几乎静止）被摧毁成
  「张腿 + 腿划水 + 抬起机身」的腿式步态（双轮接地 0.2%、腿速度 2.4 rad/s）。
  原因：起身样本占比过高，且倒地回合没有基座接触终止，
  策略发现「用腿推进」在两个分布上都够用，于是放弃了轮子。

本模块的做法：
  - startup 时按固定分组给每个环境打标：每 `stride` 个环境里有 1 个是「起身环境」。
  - 起身环境每次复位都从倒地姿态开始（俯卧/仰卧/倾倒 × 伸腿/收腿）。
  - 行走环境永不倒地开始，保持轮式行走的梯度纯净。
  - 同时写 `env.is_fallen_episode`（bool 张量），供终止函数判断
    「本回合是否由倒地起步」——倒地回合给起身时间，不因基座接触立即终止。
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .events import _quat_from_euler


def assign_mixed_groups(
    env: ManagerBasedRLEnv,
    env_ids,
    stride: int = 4,
):
    """startup 事件：每 stride 个环境里挑 1 个作为起身环境。

    写入 env.getup_mask (num_envs,) bool 与 env.is_fallen_episode 初值。
    """
    n = env.scene["robot"].num_instances
    idx = torch.arange(n, device=env.device)
    # 三区划分（i % stride）：
    #   rem==0          → 越障区（障碍物在前方 1.8 m，练蹲跳）        占比 1/stride
    #   rem==stride-1   → 起身区（每次复位都从倒地姿态开始）          占比 1/stride
    #   其余            → 纯行走区（无障碍、永不倒地）                占比 (stride-2)/stride
    # stride=8 时 = 12.5% 越障 + 12.5% 起身 + 75% 行走：
    # 首轮用过 50% 越障，机器人卡在障碍物上产生大量「翻倒」样本，
    # 轮式行走被稀释（速度误差 0.09 -> 0.55），因此行走区必须占多数。
    st = max(stride, 3)
    rem = idx % st
    env.getup_mask = rem == (st - 1)
    env.obstacle_mask = rem == 0
    env.is_fallen_episode = torch.zeros(n, dtype=torch.bool, device=env.device)


def reset_mixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    z_range: tuple = (0.28, 0.42),
    tilt_range: tuple = (0.9, 1.6),
    joint_noise: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """复位：起身环境放倒地姿态，其余环境正常站立复位。

    与 reset_to_fallen_diverse 的区别：不按概率采样，而是按环境分组，
    保证行走环境始终从站立起步（轮式行走梯度纯净）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "getup_mask"):
        # 未执行 startup 事件时退化为「全部站立」
        env.getup_mask = torch.zeros(asset.num_instances, dtype=torch.bool, device=env.device)
        env.is_fallen_episode = torch.zeros(asset.num_instances, dtype=torch.bool, device=env.device)

    ids = env_ids[env_ids < asset.num_instances]
    if len(ids) == 0:
        return

    mask = env.getup_mask[ids]
    fallen_ids = ids[mask]
    # 记录本回合属性（终止函数据此放行基座接触）
    env.is_fallen_episode[ids] = mask
    if hasattr(env, "_episode_steps_elapsed"):
        env._episode_steps_elapsed[ids] = 0
    if len(fallen_ids) == 0:
        return

    k = len(fallen_ids)
    root_pose = asset.data.default_root_state[fallen_ids, :7].clone()

    # default_root_state 是「环境局部坐标」，写 sim 需要叠加 env_origins；
    # 不叠加会把倒地机器人瞬移到世界原点（实测 min base_z=0.12，整批环境互相踩踏）
    origins = env.scene.env_origins[fallen_ids]
    root_pose[:, 0] = origins[:, 0] + torch.empty(k, device=env.device).uniform_(-0.1, 0.1)
    root_pose[:, 1] = origins[:, 1] + torch.empty(k, device=env.device).uniform_(-0.1, 0.1)
    root_pose[:, 2] = torch.empty(k, device=env.device).uniform_(*z_range)

    # 姿态三分：0=俯卧(pitch +90°) 1=仰卧(pitch -90°) 2=前后倾倒
    mode = torch.randint(0, 3, (k,), device=env.device)
    tilt = torch.empty(k, device=env.device).uniform_(*tilt_range)
    sign = torch.where(torch.rand(k, device=env.device) < 0.5, -1.0, 1.0)
    pitch = torch.where(
        mode == 0,
        torch.full_like(tilt, math.pi / 2),
        torch.where(mode == 1, torch.full_like(tilt, -math.pi / 2), tilt * sign),
    )
    roll = torch.where(mode == 2, tilt * sign * 0.3, torch.zeros_like(tilt))
    yaw = torch.empty(k, device=env.device).uniform_(-math.pi, math.pi)
    root_pose[:, 3:7] = _quat_from_euler(roll, pitch, yaw)

    asset.write_root_pose_to_sim(root_pose, env_ids=fallen_ids)
    asset.write_root_velocity_to_sim(torch.zeros(k, 6, device=env.device), env_ids=fallen_ids)

    # 关节：伸腿（knee 上限，轮远端撑地）/ 收腿（knee≈0，轮贴近机身）
    joint_pos = asset.data.default_joint_pos[fallen_ids].clone()
    extended = torch.rand(k, device=env.device) < 0.5
    knee = torch.where(extended, torch.full_like(joint_pos[:, 2], 0.663), torch.zeros_like(joint_pos[:, 2]))
    joint_pos[:, 2] = knee
    joint_pos[:, 6] = knee
    abad = torch.where(extended, torch.full_like(joint_pos[:, 0], -0.25), torch.zeros_like(joint_pos[:, 0]))
    joint_pos[:, 0] = abad
    joint_pos[:, 4] = abad
    joint_pos += torch.empty_like(joint_pos).uniform_(-joint_noise, joint_noise)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=fallen_ids)

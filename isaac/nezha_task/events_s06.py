# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 起身多样化 + 跳跃相关 MDP 项。"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .events import _quat_from_euler  # 复用四元数工具


def reset_to_fallen_diverse(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    probability: float | None = None,
    fall_prob_max: float = 0.5,
    fall_ramp_steps: int = 6000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_range: tuple = (0.28, 0.42),
    tilt_range: tuple = (0.9, 1.6),
    joint_noise: float = 0.15,
):
    """多样化倒地复位（S06 几何）：俯卧 / 仰卧 / 前后倾倒，配合伸腿或折腿姿态。

    对比旧版 reset_to_fallen（只有 pitch 倾倒 + 固定伸腿）：
    - 姿态三分：俯卧（面朝下）/ 仰卧（背朝下）/ 前后倾倒，随机选择。
      仰卧是原训练分布里没有的姿态——背贴地、腿朝天，必须先折髋把轮子收到
      身体下方（s06 几何下 hip=-0.5 即轮先触地，比原始几何的 -1.0 容易得多）。
    - 关节初始姿态两分：伸腿（knee=+上限，轮子远端撑地）与收腿（knee=0 附近，
      轮子贴近机身），覆盖两种真实倒地形态。
    - z_range 缩到 0.28-0.42（s06 贴地机身高 0.12-0.33，留穿透余量）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if probability is None:
        probability = float(getattr(env, "fall_prob_max", fall_prob_max))
        ramp = float(getattr(env, "fall_ramp_steps", fall_ramp_steps))
        step = int(getattr(env, "_sim_step_counter", 0))
        progress = min(max(step / max(ramp, 1), 0.0), 1.0)
        probability *= progress
    if len(env_ids) == 0 or probability <= 0.0:
        return

    ids = env_ids[env_ids < asset.num_instances]
    if len(ids) == 0:
        return

    n = len(ids)
    mask = torch.rand(n, device=env.device) < probability
    fallen_ids = ids[mask]
    if len(fallen_ids) == 0:
        return

    k = len(fallen_ids)
    root_pose = asset.data.default_root_state[fallen_ids, :7].clone()

    root_pose[:, 0] += torch.empty(k, device=env.device).uniform_(-0.1, 0.1)
    root_pose[:, 1] += torch.empty(k, device=env.device).uniform_(-0.1, 0.1)
    root_pose[:, 2] = torch.empty(k, device=env.device).uniform_(*z_range)

    # 姿态三分：0=俯卧(pitch +90° 附近) 1=仰卧(pitch -90° 附近) 2=前后倾倒
    mode = torch.randint(0, 3, (k,), device=env.device)
    tilt = torch.empty(k, device=env.device).uniform_(*tilt_range)
    sign = torch.where(torch.rand(k, device=env.device) < 0.5, -1.0, 1.0)

    pitch = torch.where(mode == 0, torch.full_like(tilt, math.pi / 2),
             torch.where(mode == 1, torch.full_like(tilt, -math.pi / 2), tilt * sign))
    roll = torch.where(mode == 2, tilt * sign * 0.3, torch.zeros_like(tilt))  # 倾倒加少量侧倾
    yaw = torch.empty(k, device=env.device).uniform_(-math.pi, math.pi)
    root_pose[:, 3:7] = _quat_from_euler(roll, pitch, yaw)

    asset.write_root_pose_to_sim(root_pose, env_ids=fallen_ids)
    asset.write_root_velocity_to_sim(torch.zeros(k, 6, device=env.device), env_ids=fallen_ids)

    # 关节姿态两分：伸腿(knee 上限 0.663) / 收腿(knee≈0)
    joint_pos = asset.data.default_joint_pos[fallen_ids].clone()
    extended = torch.rand(k, device=env.device) < 0.5
    knee_ext = torch.where(extended, torch.full_like(joint_pos[:, 2], 0.663), torch.zeros_like(joint_pos[:, 2]))
    joint_pos[:, 2] = knee_ext  # knee_L
    joint_pos[:, 6] = knee_ext  # knee_R
    # abad：伸腿时张开（轮子向外撑地），收腿时中立
    abad = torch.where(extended, torch.full_like(joint_pos[:, 0], -0.25), torch.zeros_like(joint_pos[:, 0]))
    joint_pos[:, 0] = abad  # abad_L
    joint_pos[:, 4] = abad  # abad_R
    joint_pos += torch.empty_like(joint_pos).uniform_(-joint_noise, joint_noise)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=fallen_ids)


def jump_height_bonus(env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
    """跳跃高度奖励：机身 z 超过站立高度的部分给指数奖励。

    鼓励"快速站高"的爆发动作；与 track_height 不同的是只奖不罚——低于站立
    高度不给奖励（蹲由常规 track_height 管），避免干扰正常行走/蹲站。
    """
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    z_stand = 0.60  # s06 站立高度
    # clamp 到 0.4 m：正常跳跃的合理上限；不 clamp 的话抛飞/摔落瞬间 exp 爆炸
    # （曾出现 exp 奖励 5e14，策略学会自爆刷分），彻底毁掉训练。
    excess = (z - z_stand).clamp(min=0.0, max=0.4)
    return torch.exp(excess / std) - 1.0


def wheel_retract_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """腾空收腿奖励：机身离地越高，越鼓励收腿（knee→0）。

    腾空判定：轮子最低点离地 > 3 cm。收腿减小转动惯量、降低触地冲击，
    并让轮子在越过障碍物边缘时不被勾住。
    """
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    # 轮子是否离地：机身高度接近/超过站立高度时认为可能腾空
    airborne = (z > 0.58).float()
    # knee 角度接近 0 = 收腿（knee 范围 -1.46..0.66，站立约 0，蹲时负值）
    knee = (robot.data.joint_pos[:, 2] + robot.data.joint_pos[:, 6]) * 0.5
    retract = torch.exp(-knee.abs() / 0.15)
    return airborne * retract


def obstacle_crossing_bonus(
    env: ManagerBasedRLEnv,
    obstacle_name: str = "obstacle",
    std: float = 0.5,
) -> torch.Tensor:
    """越过障碍物奖励：机身 x 每前进越过障碍物一次给一次奖励。

    障碍物沿 x 方向每 3 m 一个；奖励 = exp(-横向偏差²/std²) × 到达标志。
    用环境步进中 base x 的增长检测"越线"，避免重复计奖用标志位。
    """
    robot = env.scene["robot"]
    obstacle: RigidObject = env.scene[obstacle_name]
    x = robot.data.root_pos_w[:, 0]
    obs_x = obstacle.data.root_pos_w[:, 0]
    # 上一帧还没越过、这一帧越过了
    prev = getattr(env, "_obstacle_crossed", None)
    crossed_now = (x > obs_x) & ~(prev if prev is not None else torch.zeros_like(x, dtype=torch.bool))
    env._obstacle_crossed = x > obs_x
    y = robot.data.root_pos_w[:, 1]
    lateral = torch.exp(-(y ** 2) / std ** 2)
    return crossed_now.float() * lateral


def kneel_walking_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """跪姿行走惩罚：机身直立但底盘盒触地（趴在底盘上滑行）时给负奖励。

    "跪地走路"（膝折叠、底盘拖地、轮子悬空）是此前训练的主要失败模式，
    直接惩罚这个状态组合。
    """
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    # 直立(接近正常站高)但异常低——趴在底盘上
    upright_low = (z > 0.30) & (z < 0.45)
    return upright_low.float()

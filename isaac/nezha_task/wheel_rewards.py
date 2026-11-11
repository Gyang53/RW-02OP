# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 v3：轮式行走约束 + 蹲跳越障奖励。

设计依据（实测数据，见 isaac/diag_locomotion.py）：
  正确轮式行走（ckpt 5749）：双轮接地 98.5%、轮速 7.28≈vx/r、abad≈-0.12、
                             腿 |vel| 0.2~0.5、base_z 0.467
  错误腿式行走（ckpt 11849）：双轮接地 0.2%、abad≈+0.2、腿 |vel| 2.4~2.7、base_z 0.681、vx≈0

因此奖励必须显式奖励「双轮接地 + 轮子转动」、惩罚「腿划水 / 张腿 / 趴底盘」，
并且这些约束只在「正常行走窗口」生效（跳跃窗口与起身回合豁免），否则会压制跳跃能力。
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

WHEEL_R = 0.127
LEG_JOINTS = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"]
ABAD_NEUTRAL = -0.12  # 好策略实测 abad ≈ -0.14 / -0.10


# ----------------------------------------------------------------------------- helpers


def _t(x):
    """ProxyArray / Tensor 统一成 torch.Tensor。"""
    return x.torch if hasattr(x, "torch") else x


def _sensor(env: ManagerBasedRLEnv):
    return env.scene.sensors.get("contact_forces")


def _wheel_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    """(num_envs, 2) 两轮接地布尔。"""
    sensor = _sensor(env)
    if sensor is None:
        n = env.scene["robot"].num_instances
        return torch.ones(n, 2, dtype=torch.bool, device=env.device)
    f = _t(sensor.data.net_forces_w)
    names = list(sensor.body_names)
    idx = [names.index("wheel_L_Link"), names.index("wheel_R_Link")]
    return torch.linalg.norm(f[:, idx], dim=-1) > threshold


def _base_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    sensor = _sensor(env)
    if sensor is None:
        return torch.zeros(env.scene["robot"].num_instances, dtype=torch.bool, device=env.device)
    f = _t(sensor.data.net_forces_w)
    idx = list(sensor.body_names).index("base_link")
    return torch.linalg.norm(f[:, idx], dim=-1) > threshold


def _joint_idx(env: ManagerBasedRLEnv):
    cache = getattr(env, "_cached_joint_idx", None)
    if cache is None:
        names = list(env.scene["robot"].joint_names)
        cache = {
            "legs": [names.index(n) for n in LEG_JOINTS],
            "wheels": [names.index("wheel_L_Joint"), names.index("wheel_R_Joint")],
            "knee": [names.index("knee_L_Joint"), names.index("knee_R_Joint")],
            "abad": [names.index("abad_L_Joint"), names.index("abad_R_Joint")],
        }
        env._cached_joint_idx = cache
    return cache


def _obstacle_ahead(env: ManagerBasedRLEnv, dist: float = 1.2) -> torch.Tensor:
    """障碍物是否在机器人前方 dist 米内（跳跃窗口）。无 obstacle 时全 False。"""
    if "obstacle" not in env.scene.keys():
        return torch.zeros(env.scene["robot"].num_instances, dtype=torch.bool, device=env.device)
    robot = env.scene["robot"]
    obs = env.scene["obstacle"]
    dx = _t(obs.data.root_pos_w)[:, 0] - _t(robot.data.root_pos_w)[:, 0]
    return (dx > -0.2) & (dx < dist)


def _walking_window(env: ManagerBasedRLEnv, cmd_min: float = 0.3) -> torch.Tensor:
    """正常行走窗口：有速度指令、双轮接地、且在障碍物跳跃窗口之外。"""
    cmd = _t(env.command_manager.get_command("base_velocity"))
    moving = (cmd[:, 0].abs() > cmd_min) | (cmd[:, 2].abs() > 0.2)
    grounded = _wheel_contact(env).all(dim=1)
    return moving & grounded & ~_obstacle_ahead(env)


# ----------------------------------------------------------------------------- 奖励


def wheel_contact_reward(env: ManagerBasedRLEnv, cmd_min: float = 0.3) -> torch.Tensor:
    """双轮同时接地 → 1。有速度指令时生效，静止不给奖励（防「站着领奖」）。"""
    cmd = _t(env.command_manager.get_command("base_velocity"))
    moving = (cmd[:, 0].abs() > cmd_min) | (cmd[:, 2].abs() > 0.2)
    return (_wheel_contact(env).all(dim=1) & moving).float()


def wheel_drive_reward(env: ManagerBasedRLEnv, std: float = 2.0) -> torch.Tensor:
    """轮速与 vx/r 的匹配度（无滑移的轮式驱动）。

    只在双轮接地时计；目标 = 速度指令 vx / 轮半径。std=2.0 rad/s 来自好策略实测
    残差 2.05 rad/s（轮速 7.28 vs vx/r 6.73），这个量级下仍能给出可用梯度。
    """
    robot = env.scene["robot"]
    idx = _joint_idx(env)
    cmd = _t(env.command_manager.get_command("base_velocity"))
    w = _t(robot.data.joint_vel)[:, idx["wheels"]].mean(dim=1)
    target = cmd[:, 0] / WHEEL_R
    err = (w - target).abs()
    grounded = _wheel_contact(env).all(dim=1)
    return torch.exp(-err / std) * grounded.float()


def leg_stillness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """行走窗口内腿关节速度平方和（腿应仅作悬架）。

    好策略腿 |vel| 0.2~0.5 rad/s；坏策略 2.4~2.7。系数 1/25 使 0.5 rad/s 时惩罚 ≈0.01，
    2.5 rad/s 时 ≈0.25，形成 ~25 倍差距而不会压垮跳跃（跳跃窗口被排除）。
    """
    robot = env.scene["robot"]
    idx = _joint_idx(env)
    v = _t(robot.data.joint_vel)[:, idx["legs"]]
    return ((v * v).sum(dim=1) / 25.0) * _walking_window(env).float()


def abad_neutral_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """abad 偏离中立位 -0.12 的平方惩罚（抑制「张腿」）。"""
    robot = env.scene["robot"]
    idx = _joint_idx(env)
    q = _t(robot.data.joint_pos)[:, idx["abad"]]
    return (((q - ABAD_NEUTRAL) ** 2).sum(dim=1)) * _walking_window(env).float()


def base_drag_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """基座触地且两轮都离地（趴底盘 / 跪地滑行）→ 1，由负权重惩罚。

    用接触传感器判定，**不用高度阈值**——低趴蹲行（base_z 0.40~0.47、轮子正常接地）
    是合法姿态，上一版按高度判定把它误罚了，直接把策略推向抬腿抬身。
    """
    return (_base_contact(env) & ~_wheel_contact(env).any(dim=1)).float()


def jump_takeoff_reward(env: ManagerBasedRLEnv, target_vz: float = 1.0) -> torch.Tensor:
    """障碍物窗口内的向上起跳速度奖励（clamp 防 exp 爆炸）。

    减去 vz=0 处的基线值，保证「站着不动」得 0 分（否则 exp(-2)=0.135 会被白拿）。
    """
    robot = env.scene["robot"]
    vz = _t(robot.data.root_lin_vel_w)[:, 2]
    excess = vz.clamp(min=0.0, max=2.0)
    shape = torch.exp(-((excess - target_vz) ** 2) / 0.5) - torch.exp(torch.tensor(-(target_vz**2) / 0.5, device=env.device))
    return shape.clamp(min=0.0) * _obstacle_ahead(env, 1.0).float()


def air_retract_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """腾空时收腿（knee→0）奖励：减小触障风险、降低落地冲击。"""
    robot = env.scene["robot"]
    idx = _joint_idx(env)
    airborne = ~_wheel_contact(env).any(dim=1)
    knee = _t(robot.data.joint_pos)[:, idx["knee"]].mean(dim=1)
    return airborne.float() * torch.exp(-knee.abs() / 0.25)


def obstacle_cleared_reward(
    env: ManagerBasedRLEnv,
    ahead: float = 2.5,
    height_range: tuple = (0.12, 0.20),
) -> torch.Tensor:
    """越过障碍物一次性奖励（要求越线前有过腾空，否则只给 0.2 的「蹭过去」奖励）。

    越过后把障碍物重新摆到前方 ahead 米处（每回合可多次领奖，缓解稀疏奖励）。
    """
    robot = env.scene["robot"]
    n = robot.num_instances
    if "obstacle" not in env.scene.keys():
        return torch.zeros(n, device=env.device)
    obs = env.scene["obstacle"]
    x = _t(robot.data.root_pos_w)[:, 0]
    ox = _t(obs.data.root_pos_w)[:, 0]

    # 延迟标志：本回合越线前是否腾空过
    if not hasattr(env, "_pre_jump_air"):
        env._pre_jump_air = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._was_behind = torch.ones(n, dtype=torch.bool, device=env.device)
    airborne = ~_wheel_contact(env).any(dim=1)
    behind = x < ox
    # 在障碍物前方（behind=True）期间的腾空记录
    env._pre_jump_air = torch.where(behind & airborne, torch.ones_like(env._pre_jump_air), env._pre_jump_air)
    crossed = (~behind) & env._was_behind
    reward = torch.where(
        crossed,
        torch.where(env._pre_jump_air, torch.ones_like(x), torch.full_like(x, 0.2)),
        torch.zeros_like(x),
    )
    # 越线后清标志并重摆障碍物
    if crossed.any():
        env._pre_jump_air = torch.where(crossed, torch.zeros_like(env._pre_jump_air), env._pre_jump_air)
        ids = torch.nonzero(crossed, as_tuple=False).flatten()
        pose = _t(obs.data.root_pos_w).clone()
        h = torch.empty(len(ids), device=env.device).uniform_(*height_range)
        pose[ids, 0] = x[ids] + ahead
        pose[ids, 1] = 0.0
        pose[ids, 2] = h / 2
        quat = torch.zeros((len(ids), 4), device=env.device)
        quat[:, 0] = 1.0
        obs.write_root_pose_to_sim(torch.cat([pose[ids], quat], dim=-1), env_ids=ids)
    env._was_behind = behind
    return reward


def crouch_before_obstacle(env: ManagerBasedRLEnv, dist: float = 1.0) -> torch.Tensor:
    """接近障碍物时奖励蹲低（给「蹲→爆发」提供相位引导）。"""
    robot = env.scene["robot"]
    z = _t(robot.data.root_pos_w)[:, 2]
    low = torch.exp(-((z - 0.38).clamp(min=0.0) ** 2) / 0.01)
    return low * _obstacle_ahead(env, dist).float()

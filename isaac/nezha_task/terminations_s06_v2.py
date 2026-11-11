# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 v2 终止条件：把「靠腿走」从物理上排除掉。

实测失败模式（ckpt 11849）：双轮同时接地 0.2%、base_z 0.681（被腿撑高）、
腿部关节速度 2.4~2.7 rad/s —— 策略用腿撑地跳动前进。正确策略（ckpt 5749）：
双轮接地 98.5%、腿速度 0.2~0.5、base_z 0.467。

因此：
1. `wheels_airborne`：有速度指令时，两轮同时离地持续超过阈值 → 终止。
   跳跃窗口（障碍物附近）豁免，起身回合在起身阶段豁免。
2. `base_dragged`：基座有接触且两轮都无接触（趴底盘 / 跪地）→ 终止。
   起身回合从倒地开始，给 `getup_grace_steps` 步的宽限时间。
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _t(x):
    """ProxyArray / Tensor 统一成 torch.Tensor。"""
    return x.torch if hasattr(x, "torch") else x


def _wheel_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    """两轮接触布尔张量 (num_envs, 2)。"""
    sensor = env.scene.sensors.get("contact_forces")
    if sensor is None:
        # 无传感器时退化为「视为接地」，避免误终止
        n = env.scene["robot"].num_instances
        return torch.ones(n, 2, dtype=torch.bool, device=env.device)
    f = sensor.data.net_forces_w
    if hasattr(f, "torch"):
        f = f.torch
    names = list(sensor.body_names)
    idx = [names.index("wheel_L_Link"), names.index("wheel_R_Link")]
    return torch.linalg.norm(f[:, idx], dim=-1) > threshold


def _base_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    sensor = env.scene.sensors.get("contact_forces")
    if sensor is None:
        return torch.zeros(env.scene["robot"].num_instances, dtype=torch.bool, device=env.device)
    f = sensor.data.net_forces_w
    if hasattr(f, "torch"):
        f = f.torch
    idx = list(sensor.body_names).index("base_link")
    return torch.linalg.norm(f[:, idx], dim=-1) > threshold


def _obstacle_near(env: ManagerBasedRLEnv, dist: float = 0.6) -> torch.Tensor:
    """机器人是否接近障碍物（跳跃窗口）。无 obstacle 时返回全 False。"""
    if "obstacle" not in env.scene.keys():
        return torch.zeros(env.scene["robot"].num_instances, dtype=torch.bool, device=env.device)
    robot = env.scene["robot"]
    obs = env.scene["obstacle"]
    dx = (obs.data.root_pos_w[:, 0] - robot.data.root_pos_w[:, 0]).abs()
    dy = (obs.data.root_pos_w[:, 1] - robot.data.root_pos_w[:, 1]).abs()
    return (dx < dist) & (dy < 0.8)


def wheels_airborne(
    env: ManagerBasedRLEnv,
    grace_s: float = 0.25,
    cmd_threshold: float = 0.3,
    jump_window: float = 0.6,
):
    """有速度指令时双轮持续离地 → 终止（跳跃窗口与起身回合豁免）。"""
    contact = _wheel_contact(env)
    both_air = ~contact.any(dim=1)
    cmd = env.command_manager.get_command("base_velocity")
    if hasattr(cmd, "torch"):
        cmd = cmd.torch
    moving = cmd[:, 0].abs() > cmd_threshold
    exempt = _obstacle_near(env, jump_window)

    fallen_ep = getattr(env, "is_fallen_episode", None)
    if fallen_ep is not None:
        # 起身回合：起身阶段（基座高度低）豁免
        low = env.scene["robot"].data.root_pos_w[:, 2] < 0.50
        exempt = exempt | (fallen_ep & low)

    active = both_air & moving & ~exempt
    # 计时器：连续离地超过 grace_s 才终止
    if not hasattr(env, "_wheel_air_steps"):
        env._wheel_air_steps = torch.zeros(env.scene["robot"].num_instances, dtype=torch.long, device=env.device)
    grace_steps = max(int(grace_s / env.step_dt), 1)
    env._wheel_air_steps = torch.where(
        active, env._wheel_air_steps + 1, torch.zeros_like(env._wheel_air_steps)
    )
    return env._wheel_air_steps > grace_steps


def base_dragged(
    env: ManagerBasedRLEnv,
    getup_grace_s: float = 3.0,
    tilt_threshold: float = 0.6,
    sustain_s: float = 0.3,
):
    """翻倒判定：机身明显倾斜（|g_z|<0.6，即倾角 >53°）且（基座触地 或 双轮离地）持续 sustain_s → 终止。

    不能只用「基座接触」判定——障碍物 0.10~0.18 m、轮半径 0.127 m，机器人撞上去
    会短暂压住底盘但仍能继续，用基座接触当终止会让每次越障都直接结束回合
    （实测 base_dragged 终止率 85%、回合长度 375 步、速度跟踪退化到 0.64）。
    真正该终止的是「翻倒回不来」，因此用倾角做判据。
    起身回合（倒地起步）给 getup_grace_s 宽限。
    """
    robot = env.scene["robot"]
    gz = _t(robot.data.projected_gravity_b)[:, 2]  # 直立时 ≈ -1，翻倒时接近 0
    tilted = gz > -tilt_threshold
    stuck = tilted & (_base_contact(env) | ~_wheel_contact(env).any(dim=1))

    if not hasattr(env, "_tilt_steps"):
        env._tilt_steps = torch.zeros(robot.num_instances, dtype=torch.long, device=env.device)
    env._tilt_steps = torch.where(stuck, env._tilt_steps + 1, torch.zeros_like(env._tilt_steps))

    fallen_ep = getattr(env, "is_fallen_episode", None)
    if fallen_ep is not None:
        if not hasattr(env, "_episode_steps_elapsed"):
            env._episode_steps_elapsed = torch.zeros(robot.num_instances, dtype=torch.long, device=env.device)
        env._episode_steps_elapsed += 1
        grace_steps = max(int(getup_grace_s / env.step_dt), 1)
        allowed = fallen_ep & (env._episode_steps_elapsed < grace_steps)
    else:
        allowed = torch.zeros(robot.num_instances, dtype=torch.bool, device=env.device)

    sustain_steps = max(int(sustain_s / env.step_dt), 1)
    return (env._tilt_steps > sustain_steps) & ~allowed

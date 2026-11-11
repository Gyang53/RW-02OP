# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""发射/落地相关的 MDP 项：给训练注入"被弹飞"事件，让策略学会落地缓冲。

背景（docs 附录 D.3）：
- 用冲量代理验证过：给到 3.43 m/s 发射速度后，机身上升 0.59 m 且空中姿态可保持；
- 但落地有 7/16 环境摔倒；用**脚本姿态**做缓冲反而更差（23/16）
  → 落地稳健性只能靠训练获得。

因此：训练中周期性给一部分环境施加随机竖直冲量（模拟发射机构），
并奖励"落地后保持直立 + 轮子接地 + 速度稳定"。
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def _wheel_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    sensor = env.scene.sensors.get("contact_forces")
    if sensor is None:
        return torch.ones(env.scene["robot"].num_instances, dtype=torch.bool, device=env.device)
    f = _t(sensor.data.net_forces_w)
    names = list(sensor.body_names)
    idx = [names.index("wheel_L_Link"), names.index("wheel_R_Link")]
    return torch.linalg.norm(f[:, idx], dim=-1).any(dim=-1) > threshold


def apply_launch_impulse(
    env: ManagerBasedRLEnv,
    env_ids,
    probability: float = 0.25,
    vz_range: tuple = (2.0, 3.6),
    vx_range: tuple = (-0.6, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """周期性给一部分环境施加发射冲量（模拟弹簧/气动发射）。

    竖直速度按目标跳跃高度换算：h = vz²/(2g) → vz=3.43 对应 0.6 m。
    水平速度保留/随机化，模拟"带前进速度起跳"。
    """
    robot = env.scene[asset_cfg.name]
    ids = env_ids[env_ids < robot.num_instances]
    if len(ids) == 0:
        return
    pick = torch.rand(len(ids), device=env.device) < probability
    ids = ids[pick]
    if len(ids) == 0:
        return
    k = len(ids)
    lin = _t(robot.data.root_lin_vel_w)[ids].clone()  # (k,3) 世界系线速度
    ang = _t(robot.data.root_ang_vel_w)[ids].clone()  # (k,3) 世界系角速度
    lin[:, 2] = torch.empty(k, device=env.device).uniform_(*vz_range)
    lin[:, 0] = lin[:, 0] + torch.empty(k, device=env.device).uniform_(*vx_range)
    lin[:, 1] = 0.0
    ang.zero_()  # 起跳瞬间不带自旋，降低落地难度
    robot.write_root_velocity_to_sim(torch.cat([lin, ang], dim=-1), env_ids=ids)
    # 记录"本次落地待评估"，供奖励项使用
    if not hasattr(env, "_launch_pending"):
        env._launch_pending = torch.zeros(robot.num_instances, dtype=torch.bool, device=env.device)
    env._launch_pending[ids] = True


def landing_stability_reward(env: ManagerBasedRLEnv, vz_tol: float = 1.0) -> torch.Tensor:
    """落地稳定奖励：直立 + 轮子接地 + 竖直速度已收敛（说明已"站稳"）。

    只在被发射过的环境上生效（避免干扰普通行走），且落地站稳后清标志。
    """
    robot = env.scene["robot"]
    n = robot.num_instances
    pending = getattr(env, "_launch_pending", None)
    if pending is None:
        return torch.zeros(n, device=env.device)

    gz = _t(robot.data.projected_gravity_b)[:, 2]          # -1 = 竖直
    vz = _t(robot.data.root_lin_vel_w)[:, 2]
    upright = (gz < -0.9).float()
    grounded = _wheel_contact(env).float()
    settled = (vz.abs() < vz_tol).float()
    airborne = (vz.abs() >= vz_tol).float()

    # 空中：不奖不罚；落地站稳：满分；落地但未站稳：部分分
    reward = pending.float() * ((airborne * 0.2) + (settled * upright * grounded * 1.0))

    # 站稳即认为本次落地结束
    done = pending & (settled > 0.5) & (grounded > 0.5) & (upright > 0.5)
    if done.any():
        env._launch_pending = env._launch_pending & ~done
    return reward


def launch_tilt_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """被发射过的环境若姿态明显倾斜（g_z > -0.5）→ 惩罚，促使空中/落地保持直立。"""
    robot = env.scene["robot"]
    pending = getattr(env, "_launch_pending", None)
    if pending is None:
        return torch.zeros(robot.num_instances, device=env.device)
    gz = _t(robot.data.projected_gravity_b)[:, 2]
    return (pending & (gz > -0.5)).float()

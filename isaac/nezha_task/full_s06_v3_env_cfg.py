# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 v3 全能任务：**轮式**行走 + 蹲行/站立行走 + 蹲跳越障 + 多样化起身。

与 v1（S06-Full）的关键差异：
1. 轮式行走被显式约束：双轮接地奖励 + 轮速/速度匹配奖励 + 腿静止惩罚 +
   abad 中立惩罚 + 趴底盘惩罚，并且「双轮持续离地且要求移动」直接终止。
   （v1 实测双轮接地仅 0.2%，策略靠腿跳动前进。）
2. 行进/起身在环境维度分组（mixed_reset）：1/4 环境固定从倒地起步，
   其余环境的行走梯度保持纯净——v1 用「每回合 50% 概率倒地」把行走能力练废了。
3. 跳跃只在障碍物窗口被奖励（jump_takeoff / air_retract / obstacle_cleared），
   不会退化成原地蹦。
"""

from __future__ import annotations

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from .flat_env_s06_cfg import NezhaFlatS06EnvCfg
from .mdp import mixed_reset, terminations_s06_v2 as term2, wheel_rewards as wr
from .mdp.events import hide_slope
from .mdp.terminations import base_state_nan


@configclass
class NezhaFullS06V3EnvCfg(NezhaFlatS06EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- 轮子执行器：保持与预训练策略（ckpt 5749）相同的动力学 ----
        # 曾试过 damping 2.0->8.0 + effort 40->60「加强速度环」，结果预训练策略直接失稳
        # （双轮接地 98.5% -> 51.5%，vx 0.85 -> 0.25）。轮速滞后由 wheel_drive 奖励微调补偿。
        # 如需真正加强，应从头训练而不是在预训练权重上改动力学。

        # ---- 场景：去掉单边板（接触发散源），放一个可移动障碍物 ----
        # env_spacing 默认 2.5 m，而障碍物 1.6 m 宽 + 摆在 +2.5 m 会落到相邻环境格子里，
        # 把邻居机器人压在地上（实测 base_dragged 终止率 71%、回合长度 1 s）。
        # 改为 4 m 间距 + 1.2 m 宽障碍 + 前方 1.5 m，保证完全落在本环境格内。
        self.scene.env_spacing = 4.0
        self.scene.plank = None
        self.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.10, 1.2, 0.16),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, 0.08)),
        )
        self.events.disable_slope = EventTerm(func=hide_slope, mode="startup")

        # ---- 行进 / 起身 分组复位（关键结构修正）----
        self.events.assign_groups = EventTerm(
            func=mixed_reset.assign_mixed_groups,
            mode="startup",
            params={"stride": 8},  # 12.5% 越障 + 12.5% 起身 + 75% 纯行走
        )
        self.events.reset_fallen = EventTerm(
            func=mixed_reset.reset_mixed,
            mode="reset",
            params={"z_range": (0.28, 0.42), "tilt_range": (0.9, 1.6), "joint_noise": 0.15},
        )
        # 障碍物复位：摆到机器人前方 2.5 m，高度按课程 0.10 → 0.18 m 升温
        self.events.respawn_obstacle = EventTerm(
            func=_respawn_obstacle,
            mode="reset",
            params={"ahead": 1.8, "h_min": 0.06, "h_max": 0.18, "ramp_steps": 10000, "warmup_steps": 30000},
        )

        # ---- 终止：NaN / 双轮长时间离地 / 趴底盘 ----
        self.terminations.nan_state = DoneTerm(func=base_state_nan)
        self.terminations.base_contact = None  # 由 base_dragged 用接触判定替代（高度阈值会误伤低趴姿）
        self.terminations.wheels_airborne = DoneTerm(
            func=term2.wheels_airborne,
            params={"grace_s": 0.25, "cmd_threshold": 0.3, "jump_window": 0.6},
        )
        self.terminations.base_dragged = DoneTerm(
            func=term2.base_dragged,
            params={"getup_grace_s": 3.0, "tilt_threshold": 0.6, "sustain_s": 0.3},
        )
        # 动作限幅 ⚠ 关键教训：JointVelocityActionCfg 的 clip 作用在**缩放后的目标**上，
        # 即 rad/s 而不是动作值。原写法 wheel_vel.clip=±3.0 把轮速上限锁在 3 rad/s
        # = 0.38 m/s，而速度指令要到 1.0 m/s —— 策略只好用腿补速度，
        # 于是学会了「张腿 + 腿划水」的假步态（实测双轮接地 0.2%、腿速 2.5 rad/s）。
        # 正确做法：轮速限幅按物理量给（±60 rad/s = 7.6 m/s，远高于动作空间实际使用值）；
        # 腿的位置目标限幅 ±8 rad 也远超关节限位，只起防数值发散作用。
        self.actions.joint_pos.clip = {".*": (-8.0, 8.0)}
        self.actions.wheel_vel.clip = {".*": (-60.0, 60.0)}

        # ---- 奖励 ----
        # 速度跟踪保持主导（轮式驱动是达成手段，由下面的轮子项保证）
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # 高度：站立精度与蹲行/起身折中（好策略站立 0.467，指令范围 0.35~0.60）
        self.rewards.track_height.weight = 1.0
        self.rewards.track_height.params["std"] = 0.12

        # 轮式行走约束
        self.rewards.wheel_contact = RewTerm(func=wr.wheel_contact_reward, weight=2.0)
        self.rewards.wheel_drive = RewTerm(func=wr.wheel_drive_reward, weight=1.0, params={"std": 2.0})
        self.rewards.leg_stillness = RewTerm(func=wr.leg_stillness_penalty, weight=-1.0)
        self.rewards.abad_neutral = RewTerm(func=wr.abad_neutral_penalty, weight=-1.0)
        self.rewards.base_drag = RewTerm(func=wr.base_drag_penalty, weight=-4.0)

        # 蹲跳越障
        self.rewards.jump_takeoff = RewTerm(func=wr.jump_takeoff_reward, weight=3.0, params={"target_vz": 1.0})
        self.rewards.air_retract = RewTerm(func=wr.air_retract_reward, weight=0.5)
        self.rewards.obstacle_cleared = RewTerm(
            func=wr.obstacle_cleared_reward, weight=8.0, params={"ahead": 1.8, "height_range": (0.06, 0.18)}
        )
        self.rewards.crouch_approach = RewTerm(func=wr.crouch_before_obstacle, weight=0.8, params={"dist": 1.0})

        # 上一版遗留的「按高度判定跪地」奖励项去掉（会误伤低趴蹲行）
        self.rewards.kneel_penalty = None

        self.commands.height.range = (0.35, 0.60)
        self.episode_length_s = 20.0


def _respawn_obstacle(
    env,
    env_ids,
    ahead: float = 1.8,
    h_min: float = 0.06,
    h_max: float = 0.18,
    ramp_steps: int = 10000,
    warmup_steps: int = 30000,
):
    """复位时把障碍物摆到机器人前方，高度按仿真步数线性升温（课程）。"""
    robot = env.scene["robot"]
    obs = env.scene["obstacle"]
    step = int(getattr(env, "_sim_step_counter", 0))
    progress = min(max(step / max(ramp_steps, 1), 0.0), 1.0)
    hi = h_min + (h_max - h_min) * progress
    h = torch.empty(len(env_ids), device=env.device).uniform_(h_min, hi)

    # 障碍物要摆到「该环境自己的世界坐标」上：env_origin + ahead
    origins = env.scene.env_origins[env_ids]
    pose = obs.data.default_root_state[env_ids, :7].clone()
    # 只有「越障区」环境把障碍物摆在前面；纯行走区/起身区停到 100 m 外（不干扰）
    omask = getattr(env, "obstacle_mask", None)
    active = omask[env_ids] if omask is not None else torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    # 两阶段课程：warmup 期内障碍物全部停远，先把轮式行走与起身练稳；
    # warmup 之后再激活越障区（否则机器人频繁卡在障碍前，污染行走梯度）。
    if step < warmup_steps:
        active = torch.zeros_like(active)
    pose[:, 0] = origins[:, 0] + torch.where(
        active, torch.full_like(h, ahead), torch.full_like(h, 100.0)
    )
    pose[:, 1] = origins[:, 1]
    pose[:, 2] = torch.where(active, h / 2, torch.full_like(h, 0.08))
    obs.write_root_pose_to_sim(pose, env_ids=env_ids)
    # 越障标志随重置清零
    if hasattr(env, "_pre_jump_air"):
        env._pre_jump_air[env_ids] = False
        env._was_behind[env_ids] = True

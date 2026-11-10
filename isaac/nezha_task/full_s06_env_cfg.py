# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1-s06 全能任务：行走 + 蹲站 + 多样化起身 + 跳跃越障。

单策略覆盖全部技能，场景含周期性矮障碍物（0.12-0.18 m），策略需要
"蹲下蓄力 → 快速站高 + 收腿"跃过。键盘对应 W/S/A/D + Z/X/C + R。
"""

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

import isaaclab_tasks.core.velocity.mdp as mdp

from .flat_env_s06_cfg import NezhaFlatS06EnvCfg
from .mdp import events_s06
from .mdp.events import hide_slope
from .mdp.terminations import base_state_nan


@configclass
class NezhaFullS06EnvCfg(NezhaFlatS06EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- 场景：去掉单边板（起身接触发散源），放周期障碍物 ----
        self.scene.plank = None
        # 障碍物：高 0.15 m 的横条，每环境错位摆放；策略前方 1.5 m 处第一块
        # （Isaac 场景所有环境共享同一布局；靠 x 间隔防止环境间互撞）
        self.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.10, 1.6, 0.15),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.075)),
        )
        # 场景斜板埋掉（起身接触发散源）
        self.events.disable_slope = EventTerm(func=hide_slope, mode="startup")

        # ---- 倒地多样化复位：俯卧/仰卧/倾倒 + 伸腿/收腿，概率 0→0.5 缓升 ----
        self.events.reset_fallen = EventTerm(
            func=events_s06.reset_to_fallen_diverse,
            mode="reset",
            params={"fall_prob_max": 0.5, "fall_ramp_steps": 6000},
        )
        # NaN 终止 + 动作限幅（起身阶段防接触求解发散）
        self.terminations.nan_state = DoneTerm(func=base_state_nan)
        self.actions.joint_pos.clip = {".*": (-3.0, 3.0)}
        self.actions.wheel_vel.clip = {".*": (-3.0, 3.0)}
        # 倒地时 base_link 贴地，不能作为终止条件
        self.terminations.base_contact = None

        # ---- 奖励 ----
        # 高度跟踪：站立精度与起身/跳跃梯度折中
        self.rewards.track_height.weight = 1.2
        self.rewards.track_height.params["std"] = 0.20
        # 跳跃：腾空高度奖励 + 腾空收腿奖励
        self.rewards.jump_height = RewTerm(
            func=events_s06.jump_height_bonus, weight=2.0, params={"std": 0.08}
        )
        self.rewards.wheel_retract = RewTerm(
            func=events_s06.wheel_retract_bonus, weight=0.5
        )
        # 越障：越过障碍物一次性奖励（障碍物沿 x 每 3 m 重摆——用 reset 事件实现周期化）
        self.rewards.obstacle_cross = RewTerm(
            func=events_s06.obstacle_crossing_bonus, weight=5.0, params={"std": 0.5}
        )
        # 跪姿惩罚：直立但趴底盘
        self.rewards.kneel_penalty = RewTerm(
            func=events_s06.kneel_walking_penalty, weight=-2.0
        )
        # 速度跟踪保持主导
        self.rewards.track_lin_vel_xy_exp.weight = 1.5

        # ---- 事件：每回合把障碍物放到机器人前方（ x + 2.5 m），越过后再放下一块 ----
        self.events.respawn_obstacle = EventTerm(
            func=_respawn_obstacle_ahead,
            mode="reset",
            params={"ahead": 2.5, "height_range": (0.12, 0.20)},
        )

        # 高度指令：0.35（深蹲蓄力）— 0.60（站立）；跳跃由奖励引导而非指令
        self.commands.height.range = (0.35, 0.60)
        self.episode_length_s = 20.0


def _respawn_obstacle_ahead(env, env_ids, ahead: float = 2.5, height_range=(0.12, 0.20)):
    """复位时把障碍物摆到机器人前方 ahead 米处，高度随机。"""
    robot = env.scene["robot"]
    obstacle = env.scene["obstacle"]  # RigidObject（AssetBaseCfg 经 scene 构建后是 RigidObject 实例）
    rx = robot.data.default_root_state[env_ids, 0]
    ox = rx + ahead
    h = torch.empty(len(env_ids), device=env.device).uniform_(*height_range)
    pos = obstacle.data.default_root_state[env_ids].clone()
    pos[:, 0] = ox
    pos[:, 1] = 0.0
    pos[:, 2] = h / 2
    obstacle.write_root_pose_to_sim(pos[:, :7], env_ids=env_ids)
    env._obstacle_height = h
    env._obstacle_crossed = torch.zeros(len(robot.data.root_pos_w), dtype=torch.bool, device=env.device)

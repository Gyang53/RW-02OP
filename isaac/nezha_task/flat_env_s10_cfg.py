# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nezhav1 轮足机器人速度跟踪环境（平地，0.6 缩放腿）。

腿缩短 0.6 后：站立 ~0.60 m、蹲 0.335 m、全伸 0.629 m。
高度指令范围相应改为 (0.35, 0.60)。
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .mdp.commands import UniformHeightCommandCfg
from .mdp.rewards import track_height_exp
from .nezha_s10_cfg import LEG_JOINTS, NEZHA_S10_CFG as NEZHA_CFG, WHEEL_JOINTS


@configclass
class NezhaActionsCfg:
    """腿位置 + 轮速度（与 rl/ 里的 CPU 环境一致）。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[LEG_JOINTS], scale=0.25, use_default_offset=True
    )
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[WHEEL_JOINTS], scale=10.0)


@configclass
class NezhaCommandsCfg:
    """只用前进速度 + 偏航角速度（双轮不能横移）。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    # 机身高度指令（蹲下 / 站高），范围来自腿的可达高度
    height = UniformHeightCommandCfg(resampling_time_range=(10.0, 10.0), range=(0.55, 0.85))


@configclass
class NezhaFlatS10EnvCfg(LocomotionVelocityRoughEnvCfg):
    """平地速度跟踪。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 物理后端：CPU 上 Newton/MJWarp 更稳（GPU 也可用，二者都跑 CUDA）----
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 256
        newton_mjwarp.solver_cfg.nconmax = 128
        self.sim.physics.default = newton_mjwarp

        # ---- 场景 ----
        self.scene.robot = NEZHA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # ---- 地形：单边板（只有一侧轮子会踩上去，练单腿稳定性）----
        # 板子沿 x 方向，y 偏移到右轮路径上；机器人前进时右轮上板、左轮在地面
        self.scene.plank = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/plank",
            spawn=sim_utils.CuboidCfg(
                size=(1.8, 0.18, 0.06),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1, 0.105, 0.03)),
        )

        # ---- 动作 ----
        self.actions = NezhaActionsCfg()

        # ---- 指令 ----
        self.commands = NezhaCommandsCfg()

        # ---- 观测：去掉 height_scan，末尾追加高度指令（保持与 MuJoCo/部署相同的 37 维布局）----
        self.observations.policy.height_scan = None
        self.observations.policy.height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "height"}
        )

        # ---- 奖励 ----
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.track_height = RewTerm(
            func=track_height_exp, weight=0.75, params={"command_name": "height", "std": 0.1}
        )

        # ---- 终止 ----
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # ---- 事件：body 名改成 USD 里的 base_link ----
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        # 迁移到 MuJoCo 的关键：接触摩擦/质量都要在范围内变化，否则策略过拟合 PhysX
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.4, 0.4), "y": (-0.3, 0.3)}

        # 复位关节：默认角全是 0，用 scale 没意义；改成 absolute offset 扰动 + 轮子相位随机
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (-0.08, 0.08),
                "velocity_range": (-0.3, 0.3),
            },
        )

        # ---- 课程 ----
        self.curriculum.terrain_levels = None

        # ---- 常驻设置 ----
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005

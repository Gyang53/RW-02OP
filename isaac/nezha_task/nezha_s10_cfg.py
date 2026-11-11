"""nezhav1 轮足机器人 Isaac Lab 资产配置。

USD 由 isaac/make_clean_mjcf.py + isaac/convert_mjcf.py 从 MJCF 转换得到：
  mjmodel.xml -> nezha_isaac.xml -> nezha_isaac.usda（带关节/质量/碰撞）
可用环境变量 NEZHA_USD 覆盖路径。
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

NEZHA_USD = os.environ.get(
    "NEZHA_USD",
    "/home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController"
    "/assets/nezha/models/nezhav1/mjcf/mjmodel_s10/mjmodel_s10.usda",
)

LEG_JOINTS = "(abad|hip|knee)_[LR]_Joint"
WHEEL_JOINTS = "wheel_[LR]_Joint"

NEZHA_S10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=NEZHA_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.844),
        joint_pos={
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿：位置 PD（隐式），与 rl/nezha_env.py 的 KP_LEG/KD_LEG 对齐
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[LEG_JOINTS],
            # ⚠ 与真机对齐：wb_motor_config.json 里 max_current 60A × kt 2.1 = 90 N·m
            effort_limit_sim=90.0,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=2.0,
        ),
        # 轮：速度控制（stiffness=0，damping 即速度增益）
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[WHEEL_JOINTS],
            effort_limit_sim=40.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)
"""nezhav1 轮足机器人 原始腿长版（×1.0，用于验证"加长腿是否能起跳"）。

腿长缩短后站立高度从 0.88 降到 ~0.60（knee=0 轮底触地），
初始高度与高度指令范围按几何重算：蹲 0.335 m ↔ 站 0.629 m。
USD 来自 assets/.../mjcf/mjmodel_s06.xml 转换（isaac/convert_mjcf.py + NEZHA_MJCF_NAME）。
"""

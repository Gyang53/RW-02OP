#!/usr/bin/python3
"""在 mjsim(MuJoCo/ROS) 里跑 Isaac Lab 训练出来的策略，并用键盘(kb_joy)控制。

启动顺序（重要：先起本节点，再起 sim_node，机器人一落地就被接管）：
    source install/setup.bash
    python3 isaac/policy_node_isaac.py --ros-args -p policy:=isaac/policies/nezha_flat_isaac.pt
    ros2 run mjsim sim_node --ros-args -p model:=nezhav1
    ros2 run mjsim kb_joy            # W/S 前后，A/D 转向，Z/X 蹲下/起身（按住），Space 停

观测 37 维（与 Isaac 训练完全一致）：
    base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | cmd(3)
    | joint_pos(8) | joint_vel(8) | last_action(8) | height_cmd(1)
    height_cmd = 绝对机身高度 (m)，h 由键盘 Z/X 积分得到（joy.axes[2]）
动作 8 维 -> sim_node 的 5n MIT 指令 [tau, pos, vel, kp, kd]×8：
    腿  [abad_L, hip_L, knee_L, abad_R, hip_R, knee_R]: pos = 0.25a, kp=40, kd=2
    轮  [wheel_L, wheel_R]                             : vel = 10a,  kp=0,  kd=2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, Joy
from std_msgs.msg import Float32MultiArray

OBS_DIM = 37
ACT_DIM = 8


# sim_node 的 actuator 顺序
ACTUATOR_NAMES = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
]
# Isaac action 顺序 -> sim actuator 下标
ISAAC_LEG_ORDER = [0, 1, 2, 4, 5, 6]     # abad_L, hip_L, knee_L, abad_R, hip_R, knee_R
ISAAC_WHEEL_ORDER = [3, 7]               # wheel_L, wheel_R

LEG_POS_SCALE = 0.25
WHEEL_VEL_SCALE = 10.0
LEG_KP, LEG_KD = 40.0, 2.0
WHEEL_KP, WHEEL_KD = 0.0, 2.0


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """机体系下的向量：q=(w,x,y,z) 单帧。"""
    w, x, y, z = q
    qv = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * (2.0 * np.dot(qv, v))
    return a - b + c


class IsaacPolicyNode(Node):
    def __init__(self):
        super().__init__("isaac_policy_node")
        self.declare_parameter("policy", "")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("vx_scale", 0.3)
        self.declare_parameter("wz_scale", 0.4)
        self.declare_parameter("cmd_alpha", 0.2)   # 键盘指令低通
        self.declare_parameter("height_min", 0.45)   # 机身目标高度下限 (m)
        self.declare_parameter("height_max", 0.85)   # 机身目标高度上限 (m)
        self.declare_parameter("height_rate", 0.15)  # joy.axes[2] 满量程时的高度变化率 (m/s)
        self.declare_parameter("debug", False)

        policy_path = self.get_parameter("policy").value
        if not policy_path:
            raise RuntimeError("需要 -p policy:=<导出的 policy.pt>")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        self.get_logger().info(f"policy: {policy_path}")

        self.rate = float(self.get_parameter("rate").value)
        self.vx_scale = float(self.get_parameter("vx_scale").value)
        self.wz_scale = float(self.get_parameter("wz_scale").value)
        self.cmd_alpha = float(self.get_parameter("cmd_alpha").value)
        self.height_min = float(self.get_parameter("height_min").value)
        self.height_max = float(self.get_parameter("height_max").value)
        self.height_rate = float(self.get_parameter("height_rate").value)

        self.q = np.zeros(8)
        self.qd = np.zeros(8)
        self.ang_vel = np.zeros(3)
        self.lin_vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])       # w,x,y,z
        self.cmd = np.zeros(2)                            # vx, wz
        self.cmd_target = np.zeros(2)
        self.height = self.height_max                     # 当前目标机身高度 (m)
        self.height_vel = 0.0                             # joy.axes[2]，-1 蹲 / +1 起
        self.last_action = np.zeros(ACT_DIM)

        self.got_js = False
        self.got_imu = False
        self.got_base = False
        self.debug = bool(self.get_parameter("debug").value)
        self.tick = 0

        self.pub = self.create_publisher(Float32MultiArray, "/mujoco_action", 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(Imu, "/imu/data", self.on_imu, 10)
        self.create_subscription(Float32MultiArray, "/base_state", self.on_base, 10)
        self.create_subscription(Joy, "/joy", self.on_joy, 10)
        self.create_timer(1.0 / self.rate, self.on_timer)
        self.get_logger().info(
            f"{self.rate:.0f} Hz | vx_scale={self.vx_scale} wz_scale={self.wz_scale}"
        )

    # ---------- 输入 ----------
    def on_js(self, msg: JointState):
        idx = {name: i for i, name in enumerate(msg.name)}
        for j, name in enumerate(ACTUATOR_NAMES):
            if name in idx:
                self.q[j] = msg.position[idx[name]]
                self.qd[j] = msg.velocity[idx[name]]
        self.got_js = True

    def on_imu(self, msg: Imu):
        q = msg.orientation
        self.quat[:] = [q.w, q.x, q.y, q.z]
        self.ang_vel[:] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.got_imu = True

    def on_base(self, msg: Float32MultiArray):
        if len(msg.data) >= 6:
            self.lin_vel[:] = msg.data[0:3]
            self.got_base = True

    def on_joy(self, msg: Joy):
        if len(msg.axes) >= 2:
            self.cmd_target[0] = float(msg.axes[1]) * self.vx_scale      # W/S
            self.cmd_target[1] = float(msg.axes[0]) * self.wz_scale      # A/D
        if len(msg.axes) > 2:
            self.height_vel = float(np.clip(msg.axes[2], -1.0, 1.0))     # Z/X 蹲下/起身

    # ---------- 推理 ----------
    def on_timer(self):
        if not (self.got_js and self.got_imu):
            return
        self.cmd += self.cmd_alpha * (self.cmd_target - self.cmd)
        self.height = float(np.clip(
            self.height + self.height_vel * self.height_rate / self.rate,
            self.height_min, self.height_max))

        proj_g = quat_rotate_inverse(self.quat, np.array([0.0, 0.0, -1.0]))
        obs = np.concatenate([
            self.lin_vel,                       # base_lin_vel
            self.ang_vel,                       # base_ang_vel
            proj_g,                             # projected_gravity
            np.array([self.cmd[0], 0.0, self.cmd[1]]),   # velocity_commands
            self.q,                             # joint_pos (default 全 0)
            self.qd,                            # joint_vel
            self.last_action,
            np.array([self.height]),                                  # height_cmd（绝对高度，与训练一致）
        ]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), obs.shape

        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs[None, :])).numpy()[0]
        self.last_action = action

        if self.debug and self.tick % 10 == 0:
            self.get_logger().info(
                f"obs lin={np.round(self.lin_vel,3)} ang={np.round(self.ang_vel,3)} "
                f"g={np.round(proj_g,3)} q={np.round(self.q,2)} h={self.height:.3f} "
                f"a={np.round(action,2)}"
            )
        self.tick += 1

        self.pub.publish(self._to_mit(action))

    def _to_mit(self, action: np.ndarray) -> Float32MultiArray:
        tau = np.zeros(8)
        pos = np.zeros(8)
        vel = np.zeros(8)
        kp = np.zeros(8)
        kd = np.zeros(8)
        # 腿：位置 PD
        pos[ISAAC_LEG_ORDER] = LEG_POS_SCALE * action[0:6]
        kp[ISAAC_LEG_ORDER] = LEG_KP
        kd[ISAAC_LEG_ORDER] = LEG_KD
        # 轮：速度环（kp=0, kd 即速度增益）
        vel[ISAAC_WHEEL_ORDER] = WHEEL_VEL_SCALE * action[6:8]
        kp[ISAAC_WHEEL_ORDER] = WHEEL_KP
        kd[ISAAC_WHEEL_ORDER] = WHEEL_KD

        msg = Float32MultiArray()
        msg.data = np.concatenate([tau, pos, vel, kp, kd]).astype(float).tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IsaacPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

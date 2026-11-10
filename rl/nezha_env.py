"""nezhav1 并行 MuJoCo 环境（与 Isaac Lab 任务同接口，用于热启动微调）。

一个进程内并行 N 个 MjData，控制 50 Hz（物理 1 kHz，decimation=20）。

观测 37 维（与 Isaac-Lab 的 ObservationsCfg 顺序完全一致）：
    base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | cmd(3)
    | joint_pos(8) | joint_vel(8) | last_action(8) | height_cmd_norm(1)
    height_cmd_norm = (height_cmd - 0.65) / 0.25，height_cmd 为机身目标高度 [0.45, 0.85]

动作 8 维（与 Isaac 的 action term 顺序一致）：
    [abad_L, hip_L, knee_L, abad_R, hip_R, knee_R] 位置目标 = 0.25a, kp=40, kd=2
    [wheel_L, wheel_R]                             速度目标 = 10a,  kp=0,  kd=2

奖励与 Isaac 的 RewardsCfg 对齐（每控制步）：
    track_lin_vel_xy_exp 1.5, track_ang_vel_z_exp 0.75, lin_vel_z_l2 -2,
    ang_vel_xy_l2 -0.05, flat_orientation_l2 -2.5, dof_torques_l2 -2e-4,
    dof_acc_l2 -2.5e-7, action_rate_l2 -0.01, dof_pos_limits -0.5,
    track_base_height_exp 0.75
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

WS = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = WS / "assets/nezha/models/nezhav1/mjcf/mjmodel.xml"

ACTUATOR_NAMES = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
]
LEG_SIM = np.array([0, 1, 2, 4, 5, 6])
WHEEL_SIM = np.array([3, 7])

LEG_POS_SCALE, WHEEL_VEL_SCALE = 0.25, 10.0
LEG_KP, LEG_KD, WHEEL_KP, WHEEL_KD = 40.0, 2.0, 0.0, 2.0

# 高度指令归一化：(h - HEIGHT_REF) / HEIGHT_SCALE
HEIGHT_REF, HEIGHT_SCALE = 0.65, 0.25
HEIGHT_TRACK_STD = 0.01

OBS_DIM = 37
ACT_DIM = 8


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q: [..., 4] (w,x,y,z), v: [..., 3] -> 机体系向量"""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., 0:1]
    qv = q[..., 1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * (2.0 * np.sum(qv * v, axis=-1, keepdims=True))
    return a - b + c


def action_to_tau(action: np.ndarray, q: np.ndarray, qd: np.ndarray,
                  lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """动作 -> 力矩（与 isaac/policy_node_isaac.py 完全一致）。"""
    a = np.asarray(action, dtype=np.float64)
    tau = np.zeros_like(a)
    pos = LEG_POS_SCALE * a[..., 0:6]
    vel = WHEEL_VEL_SCALE * a[..., 6:8]
    tau[..., LEG_SIM] = LEG_KP * (pos - q[..., LEG_SIM]) + LEG_KD * (0.0 - qd[..., LEG_SIM])
    tau[..., WHEEL_SIM] = WHEEL_KP * (0.0 - q[..., WHEEL_SIM]) + WHEEL_KD * (vel - qd[..., WHEEL_SIM])
    return np.clip(tau, lo, hi)


@dataclass
class EnvCfg:
    model_path: str = str(DEFAULT_MODEL)
    num_envs: int = 64
    physics_dt: float = 0.001
    decimation: int = 20          # 50 Hz 控制
    episode_s: float = 20.0

    # 复位
    init_z: float = 0.88
    init_z_noise: float = 0.02
    init_q_noise: float = 0.08
    init_qd_noise: float = 0.3

    # 指令（与 Isaac 的 UniformVelocityCommandCfg 一致）
    cmd_vx_range: tuple = (-1.0, 1.0)
    cmd_wz_range: tuple = (-1.0, 1.0)
    cmd_forward_bias: float = 0.0   # >0 时 vx 更偏向前进（地形训练用）
    zero_wz_prob: float = 0.5       # 直行(零偏航)指令的比例
    height_range: tuple = (0.45, 0.85)   # 机身目标高度指令采样范围（米）

    # 终止
    min_base_z: float = 0.45
    max_tilt_cos: float = 0.5       # projected_gravity.z > -0.5 => 倾角 > 60°
    max_base_z: float = 1.3

    # 地形：斜木板
    slope_enabled: bool = True
    slope_x: float = 2.0            # 木板中心距起点
    slope_y: float = 0.0
    slope_angle_range: tuple = (-0.25, -0.08)   # rad，绕 y 轴
    slope_randomize: bool = True                # 每个 episode 重新随机（全局）

    # 观测噪声（Isaac 训练时启用，推理时关闭）
    obs_noise: bool = True
    noise_ang_vel: float = 0.2
    noise_proj_g: float = 0.05
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 1.5

    # 奖励权重
    w_track_lin: float = 1.5
    w_track_ang: float = 0.75
    w_lin_vel_z: float = -2.0
    w_ang_vel_xy: float = -0.05
    w_flat_ori: float = -2.5
    w_torque: float = -2e-4
    w_dof_acc: float = -2.5e-7
    w_action_rate: float = -0.01
    w_pos_limits: float = -0.5
    w_track_height: float = 0.75

    seed: int = 0


class VecNezhaEnv:
    def __init__(self, cfg: EnvCfg | None = None):
        self.cfg = cfg or EnvCfg()
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.model = mujoco.MjModel.from_xml_path(c.model_path)
        self.model.opt.timestep = c.physics_dt
        self.data = [mujoco.MjData(self.model) for _ in range(c.num_envs)]
        self.num_envs = c.num_envs

        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(8)]
        assert names == ACTUATOR_NAMES, names

        self.qpos_adr = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(8)])
        self.dof_adr = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(8)])
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.jnt_lo = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0], 0] for i in range(8)])
        self.jnt_hi = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0], 1] for i in range(8)])
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.slope_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slope")

        self.max_steps = int(round(c.episode_s / (c.physics_dt * c.decimation)))
        self._dt_ctrl = c.physics_dt * c.decimation

        self.cmd = np.zeros((c.num_envs, 3))
        self.height_cmd = np.zeros(c.num_envs)
        self.last_action = np.zeros((c.num_envs, ACT_DIM))
        self.steps = np.zeros(c.num_envs, dtype=np.int64)

        self._q = np.zeros((c.num_envs, ACT_DIM))
        self._qd = np.zeros((c.num_envs, ACT_DIM))
        self._qd_prev = np.zeros((c.num_envs, ACT_DIM))
        self._tau = np.zeros((c.num_envs, ACT_DIM))

        if c.slope_enabled and self.slope_id >= 0:
            self._place_slope()
        elif self.slope_id >= 0:
            self.model.body_pos[self.slope_id][2] = -50.0   # 埋到地下

        self.reset()

    # ---------- 地形 ----------
    def _place_slope(self):
        c = self.cfg
        ang = self.rng.uniform(*c.slope_angle_range)
        self.model.body_pos[self.slope_id] = [c.slope_x, c.slope_y, 0.3]
        # 绕 y 轴旋转
        self.model.body_quat[self.slope_id] = [np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0]

    # ---------- 状态 ----------
    def _read_state(self):
        for i, d in enumerate(self.data):
            self._q[i] = d.qpos[self.qpos_adr]
            self._qd[i] = d.qvel[self.dof_adr]

    def _base_quat(self):
        return np.array([d.qpos[3:7] for d in self.data])

    def _velocities(self):
        lin = np.zeros((self.num_envs, 3))
        ang = np.zeros((self.num_envs, 3))
        for i, d in enumerate(self.data):
            rot = d.xmat[self.base_id].reshape(3, 3)
            lin[i] = rot.T @ d.qvel[0:3]
            ang[i] = rot.T @ d.qvel[3:6]
        return lin, ang

    def _obs(self, lin, ang, proj_g, noise: bool):
        c = self.cfg
        q = self._q
        qd = self._qd
        if noise and c.obs_noise:
            n = self.rng.uniform
            ang = ang + n(-c.noise_ang_vel, c.noise_ang_vel, ang.shape)
            proj_g = proj_g + n(-c.noise_proj_g, c.noise_proj_g, proj_g.shape)
            q = q + n(-c.noise_joint_pos, c.noise_joint_pos, q.shape)
            qd = qd + n(-c.noise_joint_vel, c.noise_joint_vel, qd.shape)
        height_norm = ((self.height_cmd - HEIGHT_REF) / HEIGHT_SCALE)[:, None]
        return np.concatenate([lin, ang, proj_g, self.cmd, q, qd, self.last_action,
                               height_norm], axis=1)

    def get_obs(self) -> np.ndarray:
        lin, ang = self._velocities()
        quat = self._base_quat()
        proj_g = quat_rotate_inverse(quat, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        return self._obs(lin, ang, proj_g, noise=False)

    # ---------- 复位 ----------
    def reset(self, env_ids=None) -> np.ndarray:
        c = self.cfg
        ids = np.arange(self.num_envs) if env_ids is None else np.atleast_1d(env_ids)
        for i in ids:
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[0] = self.rng.uniform(-0.05, 0.05)
            d.qpos[1] = self.rng.uniform(-0.05, 0.05)
            d.qpos[2] = c.init_z + self.rng.uniform(-c.init_z_noise, c.init_z_noise)
            ang = self.rng.uniform(-0.03, 0.03, 3)
            d.qpos[3:7] = euler_to_quat(ang)
            d.qpos[self.qpos_adr] = self.rng.uniform(-c.init_q_noise, c.init_q_noise, 8)
            d.qvel[:] = 0.0
            d.qvel[0:6] = self.rng.uniform(-0.05, 0.05, 6)
            d.qvel[self.dof_adr] = self.rng.uniform(-c.init_qd_noise, c.init_qd_noise, 8)
            for _ in range(20):
                mujoco.mj_forward(self.model, d)
                if d.ncon == 0:
                    break
                pen = min(d.contact[k].dist for k in range(d.ncon))
                if pen >= -1e-6:
                    break
                d.qpos[2] -= pen
            mujoco.mj_forward(self.model, d)

            self._sample_cmd(i)
            self.height_cmd[i] = self.rng.uniform(*c.height_range)
            self.last_action[i] = 0.0
            self.steps[i] = 0
        self._read_state()
        self._qd_prev = self._qd.copy()
        return self.get_obs()

    def _sample_cmd(self, i: int):
        c = self.cfg
        vx = self.rng.uniform(*c.cmd_vx_range)
        if c.cmd_forward_bias > 0:
            vx = vx + c.cmd_forward_bias if vx > -0.9 else vx
        # 一半 episode 给 wz=0（直行），否则策略学不到"零偏航"下的平衡
        wz = 0.0 if self.rng.random() < c.zero_wz_prob else self.rng.uniform(*c.cmd_wz_range)
        self.cmd[i] = [float(np.clip(vx, *c.cmd_vx_range)), 0.0, wz]

    # ---------- 单步 ----------
    def step(self, action: np.ndarray):
        c = self.cfg
        a = np.asarray(action, dtype=np.float64).reshape(self.num_envs, ACT_DIM)
        prev_action = self.last_action.copy()
        self._qd_prev = self._qd.copy()

        tau = action_to_tau(a, self._q, self._qd, self.ctrl_lo, self.ctrl_hi)
        for i in range(self.num_envs):
            d = self.data[i]
            d.ctrl[:] = tau[i]
            for _ in range(c.decimation):
                mujoco.mj_step(self.model, d)

        self._read_state()
        self.last_action = a
        self.steps += 1

        lin, ang = self._velocities()
        quat = self._base_quat()
        proj_g = quat_rotate_inverse(quat, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        obs = self._obs(lin, ang, proj_g, noise=True)

        base_z = np.array([d.qpos[2] for d in self.data])
        dof_acc = (self._qd - self._qd_prev) / self._dt_ctrl

        track_lin = np.exp(-((lin[:, 0] - self.cmd[:, 0]) ** 2 + (lin[:, 1] - self.cmd[:, 1]) ** 2) / 0.25)
        track_ang = np.exp(-((ang[:, 2] - self.cmd[:, 2]) ** 2) / 0.25)
        track_height = np.exp(-((base_z - self.height_cmd) ** 2) / HEIGHT_TRACK_STD)
        limit_excess = np.maximum(self._q - self.jnt_hi, 0) + np.maximum(self.jnt_lo - self._q, 0)

        rew = (
            c.w_track_lin * track_lin
            + c.w_track_ang * track_ang
            + c.w_lin_vel_z * lin[:, 2] ** 2
            + c.w_ang_vel_xy * (ang[:, 0] ** 2 + ang[:, 1] ** 2)
            + c.w_flat_ori * (proj_g[:, 0] ** 2 + proj_g[:, 1] ** 2)
            + c.w_torque * np.sum(tau ** 2, axis=1)
            + c.w_dof_acc * np.sum(dof_acc ** 2, axis=1)
            + c.w_action_rate * np.sum((a - prev_action) ** 2, axis=1)
            + c.w_pos_limits * np.sum(limit_excess ** 2, axis=1)
            + c.w_track_height * track_height
        )

        fall = (base_z < c.min_base_z) | (proj_g[:, 2] > -c.max_tilt_cos) | (base_z > c.max_base_z)
        timeout = self.steps >= self.max_steps
        done = fall | timeout

        if np.any(done):
            ids = np.nonzero(done)[0]
            if c.slope_enabled and c.slope_randomize and self.rng.random() < 0.15:
                self._place_slope()
            self.reset(ids)
            obs = self.get_obs()

        info = {
            "base_z": base_z,
            "height_cmd": self.height_cmd.copy(),
            "vx": lin[:, 0],
            "wz": ang[:, 2],
            "fall": fall.astype(np.float32),
            "timeout": timeout.astype(np.float32),
        }
        return obs, rew.astype(np.float32), done.astype(np.float32), info

    def close(self):
        self.data = []


def euler_to_quat(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])

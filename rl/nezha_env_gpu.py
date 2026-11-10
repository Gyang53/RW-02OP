"""nezhav1 GPU 并行环境（mujoco_warp / Warp）。

物理与网络全在 GPU：一个 nworld=N 的批量 mujoco_warp 世界，控制 50 Hz。

观测/动作/奖励与 rl/nezha_env.py（CPU 版）完全一致，也与 isaac/policy_node_isaac.py 一致：
    观测 37: base_lin_vel | base_ang_vel | projected_gravity | cmd | joint_pos | joint_vel
             | last_action | height_cmd_norm   （height_cmd_norm = (h-0.65)/0.25）
    动作 8 : [abad_L, hip_L, knee_L, abad_R, hip_R, knee_R] 位置×0.25 (kp=40, kd=2)
             [wheel_L, wheel_R] 速度×10 (kp=0, kd=2)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from nezha_env import (
    ACT_DIM, ACTUATOR_NAMES, HEIGHT_REF, HEIGHT_SCALE, HEIGHT_TRACK_STD,
    LEG_KP, LEG_KD, LEG_POS_SCALE, LEG_SIM, OBS_DIM,
    WHEEL_KD, WHEEL_KP, WHEEL_SIM, WHEEL_VEL_SCALE, EnvCfg, euler_to_quat,
)

wp.config.quiet = True


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: (N,4) wxyz, v: (N,3) -> 机体系"""
    w = q[:, 0:1]
    qv = q[:, 1:4]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(qv, v, dim=1) * w * 2.0
    c = qv * (2.0 * (qv * v).sum(dim=1, keepdim=True))
    return a - b + c


class VecNezhaEnvGpu:
    def __init__(self, cfg: EnvCfg | None = None, device: str = "cuda"):
        self.cfg = cfg or EnvCfg()
        c = self.cfg
        wp.init()
        self.device = torch.device(device)
        self.rng = np.random.default_rng(c.seed)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(c.seed)

        self.mjm = mujoco.MjModel.from_xml_path(c.model_path)
        self.mjm.opt.timestep = c.physics_dt
        self.num_envs = c.num_envs
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.put_data(self.mjm, mujoco.MjData(self.mjm), nworld=c.num_envs)

        names = [mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(8)]
        assert names == ACTUATOR_NAMES, names

        self.qpos_adr = torch.as_tensor(
            [self.mjm.jnt_qposadr[self.mjm.actuator_trnid[i, 0]] for i in range(8)],
            dtype=torch.long, device=self.device)
        self.dof_adr = torch.as_tensor(
            [self.mjm.jnt_dofadr[self.mjm.actuator_trnid[i, 0]] for i in range(8)],
            dtype=torch.long, device=self.device)
        self.ctrl_lo = torch.as_tensor(self.mjm.actuator_ctrlrange[:, 0], dtype=torch.float32, device=self.device)
        self.ctrl_hi = torch.as_tensor(self.mjm.actuator_ctrlrange[:, 1], dtype=torch.float32, device=self.device)
        self.jnt_lo = torch.as_tensor(
            [self.mjm.jnt_range[self.mjm.actuator_trnid[i, 0], 0] for i in range(8)],
            dtype=torch.float32, device=self.device)
        self.jnt_hi = torch.as_tensor(
            [self.mjm.jnt_range[self.mjm.actuator_trnid[i, 0], 1] for i in range(8)],
            dtype=torch.float32, device=self.device)
        self.base_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.slope_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "slope")
        self.leg_sim = torch.as_tensor(LEG_SIM, dtype=torch.long, device=self.device)
        self.wheel_sim = torch.as_tensor(WHEEL_SIM, dtype=torch.long, device=self.device)

        self.max_steps = int(round(c.episode_s / (c.physics_dt * c.decimation)))
        self.dt_ctrl = c.physics_dt * c.decimation

        # torch 视图（与 warp 共享显存）
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xmat = wp.to_torch(self.d.xmat)

        self.cmd = torch.zeros((c.num_envs, 3), device=self.device)
        self.height_cmd = torch.zeros(c.num_envs, device=self.device)
        self.last_action = torch.zeros((c.num_envs, ACT_DIM), device=self.device)
        self.steps = torch.zeros(c.num_envs, dtype=torch.long, device=self.device)
        self.qd_prev = torch.zeros((c.num_envs, ACT_DIM), device=self.device)

        self._default_qpos = torch.as_tensor(
            np.concatenate([[0, 0, c.init_z, 1, 0, 0, 0], np.zeros(8), [0, 0.91, 1, 0, 0, 0, 0]]),
            dtype=torch.float32, device=self.device)

        if c.slope_enabled and self.slope_id >= 0:
            self._place_slope()
        elif self.slope_id >= 0:
            self.mjm.body_pos[self.slope_id][2] = -50.0
            self.m = mjw.put_model(self.mjm)

        self.reset()

    def _place_slope(self):
        c = self.cfg
        ang = float(self.rng.uniform(*c.slope_angle_range))
        self.mjm.body_pos[self.slope_id] = [c.slope_x, c.slope_y, 0.3]
        self.mjm.body_quat[self.slope_id] = [np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0]
        self.m = mjw.put_model(self.mjm)

    # ---------- 状态 ----------
    def _rot(self):
        return self.xmat[:, self.base_id].reshape(-1, 3, 3)

    def _velocities(self):
        rot = self._rot()
        lin = torch.einsum("nji,nj->ni", rot, self.qvel[:, 0:3])
        ang = torch.einsum("nji,nj->ni", rot, self.qvel[:, 3:6])
        return lin, ang

    def _obs(self, lin, ang, proj_g, noise: bool):
        c = self.cfg
        q = self.qpos[:, self.qpos_adr]
        qd = self.qvel[:, self.dof_adr]
        if noise and c.obs_noise:
            def u(s, a):
                return (torch.rand(s, device=self.device, generator=self.torch_gen) * 2 - 1) * a
            ang = ang + u(ang.shape, c.noise_ang_vel)
            proj_g = proj_g + u(proj_g.shape, c.noise_proj_g)
            q = q + u(q.shape, c.noise_joint_pos)
            qd = qd + u(qd.shape, c.noise_joint_vel)
        height_norm = ((self.height_cmd - HEIGHT_REF) / HEIGHT_SCALE).unsqueeze(1)
        return torch.cat([lin, ang, proj_g, self.cmd, q, qd, self.last_action, height_norm], dim=1)

    def _projected_gravity(self):
        quat = self.qpos[:, 3:7]
        v = torch.zeros_like(quat[:, :3])
        v[:, 2] = -1.0
        return _quat_rotate_inverse(quat, v)

    def get_obs(self) -> np.ndarray:
        lin, ang = self._velocities()
        return self._obs(lin, ang, self._projected_gravity(), noise=False).cpu().numpy()

    # ---------- 复位 ----------
    @torch.no_grad()
    def reset(self, env_ids=None) -> np.ndarray:
        c = self.cfg
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device)
        n = ids.numel()
        base = self._default_qpos.unsqueeze(0).repeat(n, 1)
        base[:, 0] = (torch.rand(n, device=self.device, generator=self.torch_gen) * 2 - 1) * 0.05
        base[:, 1] = (torch.rand(n, device=self.device, generator=self.torch_gen) * 2 - 1) * 0.05
        base[:, 2] = c.init_z + (torch.rand(n, device=self.device, generator=self.torch_gen) * 2 - 1) * c.init_z_noise
        rpy = (torch.rand(n, 3, device=self.device, generator=self.torch_gen) * 2 - 1) * 0.03
        base[:, 3:7] = torch.as_tensor(
            np.array([euler_to_quat(r.cpu().numpy()) for r in rpy]), dtype=torch.float32, device=self.device)
        base[:, 7:15] = (torch.rand(n, 8, device=self.device, generator=self.torch_gen) * 2 - 1) * c.init_q_noise

        self.qpos[ids] = base
        self.qvel[ids] = 0.0
        self.qvel[ids, 0:6] = (torch.rand(n, 6, device=self.device, generator=self.torch_gen) * 2 - 1) * 0.05
        self.qvel[ids.unsqueeze(1), self.dof_adr.unsqueeze(0)] = (
            torch.rand(n, 8, device=self.device, generator=self.torch_gen) * 2 - 1) * c.init_qd_noise

        # 指令
        vx = torch.rand(n, device=self.device, generator=self.torch_gen) * (c.cmd_vx_range[1] - c.cmd_vx_range[0]) + c.cmd_vx_range[0]
        if c.cmd_forward_bias > 0:
            vx = torch.where(vx > -0.9, vx + c.cmd_forward_bias, vx)
        vx = vx.clamp(*c.cmd_vx_range)
        wz = torch.rand(n, device=self.device, generator=self.torch_gen) * (c.cmd_wz_range[1] - c.cmd_wz_range[0]) + c.cmd_wz_range[0]
        zero = torch.rand(n, device=self.device, generator=self.torch_gen) < c.zero_wz_prob
        wz = torch.where(zero, torch.zeros_like(wz), wz)
        self.cmd[ids, 0] = vx
        self.cmd[ids, 1] = 0.0
        self.cmd[ids, 2] = wz
        self.height_cmd[ids] = (
            torch.rand(n, device=self.device, generator=self.torch_gen) * (c.height_range[1] - c.height_range[0])
            + c.height_range[0])
        self.last_action[ids] = 0.0
        self.steps[ids] = 0
        self.qd_prev[ids] = 0.0
        return self.get_obs()

    # ---------- 单步 ----------
    @torch.no_grad()
    def step(self, action: np.ndarray):
        c = self.cfg
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.num_envs, ACT_DIM)
        prev_action = self.last_action.clone()
        self.qd_prev = self.qvel[:, self.dof_adr].clone()

        q = self.qpos[:, self.qpos_adr]
        qd = self.qvel[:, self.dof_adr]
        tau = torch.zeros_like(a)
        pos = LEG_POS_SCALE * a[:, 0:6]
        vel = WHEEL_VEL_SCALE * a[:, 6:8]
        tau[:, self.leg_sim] = LEG_KP * (pos - q[:, self.leg_sim]) + LEG_KD * (0.0 - qd[:, self.leg_sim])
        tau[:, self.wheel_sim] = WHEEL_KP * (0.0 - q[:, self.wheel_sim]) + WHEEL_KD * (vel - qd[:, self.wheel_sim])
        self.ctrl[:] = torch.clamp(tau, self.ctrl_lo, self.ctrl_hi)

        for _ in range(c.decimation):
            mjw.step(self.m, self.d)

        self.last_action = a
        self.steps += 1

        lin, ang = self._velocities()
        proj_g = self._projected_gravity()
        obs = self._obs(lin, ang, proj_g, noise=True)

        q_now = self.qpos[:, self.qpos_adr]
        qd_now = self.qvel[:, self.dof_adr]
        base_z = self.qpos[:, 2]
        dof_acc = (qd_now - self.qd_prev) / self.dt_ctrl

        track_lin = torch.exp(-((lin[:, 0] - self.cmd[:, 0]) ** 2 + (lin[:, 1] - self.cmd[:, 1]) ** 2) / 0.25)
        track_ang = torch.exp(-((ang[:, 2] - self.cmd[:, 2]) ** 2) / 0.25)
        track_height = torch.exp(-((base_z - self.height_cmd) ** 2) / HEIGHT_TRACK_STD)
        limit_excess = torch.clamp(q_now - self.jnt_hi, min=0) + torch.clamp(self.jnt_lo - q_now, min=0)

        rew = (
            c.w_track_lin * track_lin
            + c.w_track_ang * track_ang
            + c.w_lin_vel_z * lin[:, 2] ** 2
            + c.w_ang_vel_xy * (ang[:, 0] ** 2 + ang[:, 1] ** 2)
            + c.w_flat_ori * (proj_g[:, 0] ** 2 + proj_g[:, 1] ** 2)
            + c.w_torque * (tau ** 2).sum(dim=1)
            + c.w_dof_acc * (dof_acc ** 2).sum(dim=1)
            + c.w_action_rate * ((a - prev_action) ** 2).sum(dim=1)
            + c.w_pos_limits * (limit_excess ** 2).sum(dim=1)
            + c.w_track_height * track_height
        )

        fall = (base_z < c.min_base_z) | (proj_g[:, 2] > -c.max_tilt_cos) | (base_z > c.max_base_z)
        timeout = self.steps >= self.max_steps
        done = fall | timeout

        if done.any():
            ids = torch.nonzero(done).flatten()
            if c.slope_enabled and c.slope_randomize and self.rng.random() < 0.15:
                self._place_slope()
            self.reset(ids)
            obs = self._obs(*self._velocities(), self._projected_gravity(), noise=False)

        info = {
            "base_z": base_z.cpu().numpy(),
            "height_cmd": self.height_cmd.cpu().numpy(),
            "vx": lin[:, 0].cpu().numpy(),
            "wz": ang[:, 2].cpu().numpy(),
            "fall": fall.float().cpu().numpy(),
            "timeout": timeout.float().cpu().numpy(),
        }
        return obs.cpu().numpy(), rew.cpu().numpy(), done.float().cpu().numpy(), info

    def close(self):
        self.d = None

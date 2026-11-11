"""数值诊断：策略到底是"轮子驱动"还是"靠腿走"（headless）。

在一次固定 vx 的 rollout 中逐步统计：轮子是否接地、轮速是否与 vx/r 匹配、
腿关节是否大幅周期摆动、底盘是否拖地。纯只读统计，不改变环境行为。

    cd /home/yangg/data/isaacsim/IsaacLab
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python ./_isaac_sim/python.sh \
        <ws>/isaac/diag_locomotion.py --visualizer none --num_envs 16 --steps 400 \
        --fixed_vx 1.0 --checkpoint logs/rsl_rl/nezha_flat/<run>/model_<n>.pt

判读：
- wheel_contact 接近 0 % + mean|wheel_vel| 接近 0 ⇒ 轮子没在驱动（靠腿走）。
- wheel_contact 高但 |wheel_vel - vx/r| 大 ⇒ 轮子打滑或未按指令转。
- leg_activity 的 std/ptp 大且周期性 ⇒ 腿在主动推进。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-Full")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--fixed_vx", type=float, default=1.0, help="强制固定前进指令 vx（m/s）")
args_cli = parser.parse_args()
if args_cli.steps < 1:
    parser.error("--steps 必须 >= 1")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

WHEEL_RADIUS = 0.127  # m，轮半径（S06 缩放只改腿长，轮子不变）
CONTACT_THRESHOLD = 1.0  # N，接触力范数超过此值视为接地（与 mdp 里的 contact 阈值一致）

LEG_JOINT_NAMES = ("abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint")
WHEEL_JOINT_NAMES = ("wheel_L_Joint", "wheel_R_Joint")
WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
BASE_BODY_NAME = "base_link"


class MlpPolicy:
    """rsl_rl 的 actor：obs_normalizer -> mlp(ELU) -> 8 维动作（确定性）。"""

    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mean = ck["obs_normalizer._mean"].numpy().astype(np.float64).reshape(-1)
        self.var = ck["obs_normalizer._var"].numpy().astype(np.float64)
        self.w = [ck[f"mlp.{i}.weight"].numpy().T for i in (0, 2, 4, 6)]
        self.b = [ck[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(3):
            x = x @ self.w[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))  # ELU
        return x @ self.w[3] + self.b[3]


def _np(t: torch.Tensor) -> np.ndarray:
    """张量转 numpy（拷贝，避免之后被仿真写穿）。"""
    return t.detach().cpu().numpy().copy()


def _sensor_id(sensor, body_name: str) -> int:
    ids, _ = sensor.find_sensors(body_name)
    if not ids:
        _, available = sensor.find_sensors(".*")
        raise RuntimeError(f"接触传感器里找不到 body {body_name!r}，可用: {available}")
    return ids[0]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]

    wheel_body_ids = [_sensor_id(contact_sensor, name) for name in WHEEL_BODY_NAMES]
    base_body_id = _sensor_id(contact_sensor, BASE_BODY_NAME)
    leg_joint_ids, leg_names = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
    wheel_joint_ids, _ = robot.find_joints(list(WHEEL_JOINT_NAMES), preserve_order=True)
    abad_l_col = leg_names.index("abad_L_Joint")
    abad_r_col = leg_names.index("abad_R_Joint")

    wheel_contact, base_contact = [], []  # (N, 2) / (N,)
    wheel_err, wheel_abs, wheel_signed = [], [], []  # (N, 2) |w - vx/r| / |w| / w
    leg_pos, leg_vel_abs = [], []  # (N, 6)
    base_z, vx, vx_err = [], [], []  # (N,)

    for _ in range(args_cli.steps):
        command = env.unwrapped.command_manager.get_command("base_velocity")
        command[:, 0] = args_cli.fixed_vx
        command[:, 1] = 0.0
        cmd_vx = command[:, 0].clone()  # 步内可能被 resample，先快照
        if isinstance(obs, dict):
            obs = obs["policy"]
        actions = torch.as_tensor(policy(_np(obs).astype(np.float64)), dtype=torch.float32, device=env.unwrapped.device)
        obs, _, _, _, _ = env.step(actions)

        forces = contact_sensor.data.net_forces_w
        if forces is None:
            raise RuntimeError("接触传感器没有数据（net_forces_w is None）")
        force_norm = forces.torch.norm(dim=-1)  # (N, B)
        wheel_contact.append(_np(force_norm[:, wheel_body_ids] > CONTACT_THRESHOLD))
        base_contact.append(_np(force_norm[:, base_body_id] > CONTACT_THRESHOLD))

        wheel_vel = robot.data.joint_vel.torch[:, wheel_joint_ids]  # (N, 2)
        wheel_err.append(_np((wheel_vel - cmd_vx[:, None] / WHEEL_RADIUS).abs()))
        wheel_abs.append(_np(wheel_vel.abs()))
        wheel_signed.append(_np(wheel_vel))

        leg_pos.append(_np(robot.data.joint_pos.torch[:, leg_joint_ids]))
        leg_vel_abs.append(_np(robot.data.joint_vel.torch[:, leg_joint_ids].abs()))

        base_z.append(_np(robot.data.root_pos_w.torch[:, 2]))
        vx_now = robot.data.root_lin_vel_b.torch[:, 0]
        vx.append(_np(vx_now))
        vx_err.append(_np((vx_now - cmd_vx).abs()))

    env.close()

    # ---- 汇总 ----
    wheel_contact = np.stack(wheel_contact)  # (S, N, 2)
    base_contact = np.stack(base_contact)  # (S, N)
    wheel_err = np.stack(wheel_err)  # (S, N, 2)
    wheel_abs = np.stack(wheel_abs)  # (S, N, 2)
    wheel_signed = np.stack(wheel_signed)  # (S, N, 2)
    leg_pos = np.stack(leg_pos)  # (S, N, 6)
    leg_vel_abs = np.stack(leg_vel_abs)  # (S, N, 6)
    base_z = np.stack(base_z)  # (S, N)
    vx = np.stack(vx)  # (S, N)
    vx_err = np.stack(vx_err)  # (S, N)

    wheel_rate = wheel_contact.mean(axis=(0, 1)) * 100.0  # 每个轮子的接地比例
    both_rate = (wheel_contact[..., 0] & wheel_contact[..., 1]).mean() * 100.0
    base_rate = base_contact.mean() * 100.0

    wheel_err_mean = wheel_err.mean(axis=(0, 1))
    wheel_abs_mean = wheel_abs.mean()
    wheel_signed_mean = wheel_signed.mean(axis=(0, 1))
    target_w = args_cli.fixed_vx / WHEEL_RADIUS

    leg_std = leg_pos.std(axis=0).mean(axis=0)
    leg_ptp = np.ptp(leg_pos, axis=0).mean(axis=0)
    leg_vel_mean = leg_vel_abs.mean(axis=(0, 1))
    abad_mean = leg_pos.mean(axis=(0, 1))

    print("=" * 78)
    print(f"[diag] task={args_cli.task}  envs={args_cli.num_envs}  steps={args_cli.steps}")
    print(f"[diag] ckpt={args_cli.checkpoint}")
    print(f"[diag] fixed_vx={args_cli.fixed_vx:.2f} m/s  wheel_r={WHEEL_RADIUS:.3f} m  接触阈值={CONTACT_THRESHOLD} N")
    print("-" * 78)
    print(
        f"[diag] 1 wheel_contact        {WHEEL_BODY_NAMES[0]}={wheel_rate[0]:6.2f} %   "
        f"{WHEEL_BODY_NAMES[1]}={wheel_rate[1]:6.2f} %"
    )
    print(f"[diag] 2 both_wheel_contact   {both_rate:6.2f} %")
    print(
        f"[diag] 3 wheel_speed_match    mean|w - vx/r|: L={wheel_err_mean[0]:6.2f} R={wheel_err_mean[1]:6.2f} rad/s"
        f" ; mean|wheel_vel|={wheel_abs_mean:6.2f} rad/s (目标 vx/r={target_w:5.2f})"
        f" ; signed mean L={wheel_signed_mean[0]:+6.2f} R={wheel_signed_mean[1]:+6.2f} rad/s"
    )
    print(
        f"[diag] 4 leg_activity         mean std={leg_std.mean():.4f} rad  mean ptp={leg_ptp.mean():.4f} rad"
        f"  mean|joint_vel|={leg_vel_mean.mean():.3f} rad/s"
    )
    for col, name in enumerate(leg_names):
        print(f"[diag]      {name:<12} std={leg_std[col]:.4f} rad  ptp={leg_ptp[col]:.4f} rad  |vel|={leg_vel_mean[col]:.3f} rad/s")
    print(
        f"[diag] 5 abad_mean            abad_L={abad_mean[abad_l_col]:+.4f} rad  "
        f"abad_R={abad_mean[abad_r_col]:+.4f} rad  (正值大 = 张腿)"
    )
    print(
        f"[diag] 6 base_contact        {base_rate:6.2f} %  ; base_z mean={base_z.mean():.4f} m  "
        f"min={base_z.min():.4f} m"
    )
    print(f"[diag] 7 vx                   mean={vx.mean():+.3f} m/s  mean|vx-cmd|={vx_err.mean():.3f} m/s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        import sys

        sys.stdout.flush()
        simulation_app.close()
    raise SystemExit(code)

"""量化「靠腿走还是靠轮走」：轮接地率 / 轮转速 / 腿关节摆动 / 底盘拖地。

用法（headless）：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_locomotion.py \
      --task Isaac-Velocity-Flat-Nezha-S06-Full --num_envs 16 --steps 400 \
      --checkpoint <ckpt>
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
parser.add_argument("--fixed_vx", type=float, default=1.0)
parser.add_argument("--fixed_h", type=float, default=None, help="覆盖高度指令（None=不覆盖）")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

WHEEL_R = 0.127
LEG_NAMES = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"]


class MlpPolicy:
    """rsl_rl actor：obs_normalizer -> mlp(ELU) -> 8 维确定性动作。"""

    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mean = ck["obs_normalizer._mean"].numpy().astype(np.float64).reshape(-1)
        self.var = ck["obs_normalizer._var"].numpy().astype(np.float64).reshape(-1)
        self.w = [ck[f"mlp.{i}.weight"].numpy().T for i in (0, 2, 4, 6)]
        self.b = [ck[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(3):
            x = x @ self.w[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x @ self.w[3] + self.b[3]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.base_velocity.heading_command = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    reset = env.reset()
    obs = reset[0] if not isinstance(reset, dict) else reset[0]["policy"]
    if isinstance(obs, dict):
        obs = obs["policy"]

    policy = MlpPolicy(args_cli.checkpoint)
    robot = env.unwrapped.scene["robot"]
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    hcmd = env.unwrapped.command_manager.get_command("height")

    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    if sensor is None:
        print("[diag] 无 contact_forces 传感器，轮接地统计跳过")
    else:
        print(f"[diag] contact sensor bodies: {sensor.body_names}")
    body_idx = {n: i for i, n in enumerate(sensor.body_names)} if sensor is not None else {}
    wl = body_idx.get("wheel_L_Link")
    wr = body_idx.get("wheel_R_Link")
    bi = body_idx.get("base_link")

    joint_idx = {n: i for i, n in enumerate(robot.joint_names)}
    wheel_idx = [joint_idx["wheel_L_Joint"], joint_idx["wheel_R_Joint"]]
    leg_idx = [joint_idx[n] for n in LEG_NAMES]

    n = args_cli.steps
    contact_wl = np.zeros(n, dtype=bool)
    contact_wr = np.zeros(n, dtype=bool)
    contact_base = np.zeros(n, dtype=bool)
    wz = np.zeros(n)          # 轮速与 vx/r 的误差
    wspeed = np.zeros(n)      # 轮子平均转速绝对值
    joint_pos = np.zeros((n, len(LEG_NAMES)))
    joint_vel_abs = np.zeros((n, len(LEG_NAMES)))
    vx = np.zeros(n)
    base_z = np.zeros(n)
    vx_err = np.zeros(n)
    env_num = args_cli.num_envs
    vx_per_env = np.zeros(env_num)
    base_z_per_env = np.zeros(env_num)
    wl_rate = np.zeros(env_num)
    wr_rate = np.zeros(env_num)
    abadL = np.zeros(env_num)
    abadR = np.zeros(env_num)
    legvel = np.zeros(env_num)

    def forces(idx):
        f = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
        return torch.linalg.norm(f[:, idx], dim=-1)

    for i in range(n):
        if args_cli.fixed_vx is not None:
            cmd[:, 0] = args_cli.fixed_vx
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0
        if args_cli.fixed_h is not None:
            hcmd[:, 0] = args_cli.fixed_h
        if isinstance(obs, dict):
            obs = obs["policy"]
        act = torch.as_tensor(policy(obs.detach().cpu().numpy()), dtype=torch.float32, device=env.unwrapped.device)
        out = env.step(act)
        obs = out[0]
        if sensor is not None:
            if wl is not None and wr is not None:
                contact_wl[i] = bool((forces(wl) > 1.0).float().mean() > 0.5)
                contact_wr[i] = bool((forces(wr) > 1.0).float().mean() > 0.5)
            if bi is not None:
                contact_base[i] = bool((forces(bi) > 1.0).float().mean() > 0.5)
        v = robot.data.root_lin_vel_b[:, 0].cpu().numpy()
        vx[i] = v.mean()
        vx_err[i] = np.abs(v - cmd[:, 0].cpu().numpy()).mean()
        base_z[i] = robot.data.root_pos_w[:, 2].mean().item()
        jv = robot.data.joint_vel[:, wheel_idx].cpu().numpy()
        wspeed[i] = np.abs(jv).mean()
        wz[i] = np.abs(jv[:, 0] - vx[i] / WHEEL_R).mean()
        joint_pos[i] = robot.data.joint_pos[:, leg_idx].cpu().numpy().mean(axis=0)
        vx_per_env += robot.data.root_lin_vel_b[:, 0].cpu().numpy() / n
        base_z_per_env += robot.data.root_pos_w[:, 2].cpu().numpy() / n
        abadL += robot.data.joint_pos[:, joint_idx["abad_L_Joint"]].cpu().numpy() / n
        abadR += robot.data.joint_pos[:, joint_idx["abad_R_Joint"]].cpu().numpy() / n
        legvel += np.abs(robot.data.joint_vel[:, [joint_idx[x] for x in LEG_NAMES]].cpu().numpy()).mean(axis=1) / n
        if sensor is not None:
            wl_rate += (forces(wl) > 1.0).float().cpu().numpy() / n
            wr_rate += (forces(wr) > 1.0).float().cpu().numpy() / n
        joint_vel_abs[i] = np.abs(robot.data.joint_vel[:, leg_idx].cpu().numpy()).mean(axis=0)

    both = contact_wl & contact_wr
    # 每个环境一行的逐环境明细（判断是「全部走不好」还是「少数倒地环境拖低均值」）
    print("[diag] per-env: idx  cmd_vx   vx    base_z  wheelL  wheelR  abad_L  abad_R  legvel")
    cmd_vx = cmd[:, 0].detach().cpu().numpy()
    lz = base_z_per_env
    for e in range(env_num):
        print(
            f"[diag]   {e:3d}  {cmd_vx[e]:+.2f}  {vx_per_env[e]:+.2f}  {lz[e]:.3f}   "
            f"{wl_rate[e]*100:5.1f}%  {wr_rate[e]*100:5.1f}%  {abadL[e]:+.2f}  {abadR[e]:+.2f}  {legvel[e]:.2f}"
        )
    print("=" * 64)
    print(f"[diag] task={args_cli.task} envs={args_cli.num_envs} steps={n}")
    print(f"[diag] vx mean={vx.mean():+.3f}  |vx-cmd|={vx_err.mean():.3f}")
    print(f"[diag] wheel_contact  L={contact_wl.mean()*100:.1f}%  R={contact_wr.mean()*100:.1f}%  both={both.mean()*100:.1f}%")
    print(f"[diag] base_contact={contact_base.mean()*100:.1f}%  base_z mean={base_z.mean():.3f} min={base_z.min():.3f}")
    print(f"[diag] wheel_speed mean|w|={wspeed.mean():.2f} rad/s  (vx/r={vx.mean()/WHEEL_R:.2f})  err={wz.mean():.2f}")
    print("[diag] leg joint  pos_mean / pos_std / |vel|_mean:")
    for k, name in enumerate(LEG_NAMES):
        print(
            f"[diag]   {name:14s} pos={joint_pos[:, k].mean():+.3f}  std={joint_pos[:, k].std():.3f}  "
            f"|vel|={joint_vel_abs[:, k].mean():.3f}"
        )
    print(f"[diag] abad_mean L={joint_pos[:, 0].mean():+.3f} R={joint_pos[:, 3].mean():+.3f}")
    print("=" * 64)
    env.close()
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

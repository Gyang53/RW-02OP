"""隔离诊断：在 Isaac 里单看「腿伸展 → 机身抬升」，排除策略/障碍/原语干扰。

做法：把所有环境固定到同一个蹲姿、锁住轮子、只给腿一个饱和伸展指令，
记录 base_z / 竖直速度 / 膝角的响应。若机身抬不起来，说明是执行器/腿模型问题；
若正常抬起，说明问题在接触时机或障碍交互。

用法：
  cd /home/yangg/data/isaacsim/IsaacLab
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
      ./_isaac_sim/python.sh <ws>/isaac/diag_leg_extension.py --visualizer none --num_envs 8
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-GetUpTest")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--crouch_knee", type=float, default=-1.40)
parser.add_argument("--extend_knee", type=float, default=0.663)
parser.add_argument("--hold_steps", type=int, default=40, help="伸展保持的环境步数")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEG_SCALE = 0.25

# 竖直伸展轨迹（几何求解：让轮子始终在机身正下方）
# knee -> hip，来自 assets 几何（S06, 腿段 x0.6）
VERT_TABLE = [(-1.46, 0.99), (-1.20, 0.78), (-1.00, 0.65), (-0.70, 0.47),
              (-0.40, 0.31), (0.00, 0.09), (0.30, -0.06), (0.66, -0.25)]


def hip_for_knee(knee: float) -> float:
    """按表线性插值：给定 knee 返回使腿竖直的 hip。"""
    xs = [k for k, _ in VERT_TABLE]
    ys = [h for _, h in VERT_TABLE]
    if knee <= xs[0]:
        return ys[0]
    if knee >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if knee <= xs[i]:
            t = (knee - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    dev = env.unwrapped.device
    jn = robot.joint_names
    kidx = [jn.index("knee_L_Joint"), jn.index("knee_R_Joint")]
    hidx = [jn.index("hip_L_Joint"), jn.index("hip_R_Joint")]
    n = args_cli.num_envs

    def set_pose(knee, hip, z):
        q = robot.data.default_joint_pos.clone()
        for i in kidx:
            q[:, i] = knee
        for i in hidx:
            q[:, i] = hip
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        pose = robot.data.default_root_state.clone()
        pose[:, 2] = z
        pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        robot.write_root_pose_to_sim(pose[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev))

    # 蹲姿（几何：knee=-1.40 时机身约 0.34）
    # ⚠ 不要瞬移到蹲姿：z=0.34 + 折叠关节会穿地，被物理弹成躺姿（此前多次误判的根源）。
    # 改为从 standing 复位出发，用 PD 把腿驱动到蹲姿（与实际原语同一路径）。
    for _ in range(20):
        env.step(torch.zeros(env.action_space.shape, device=dev))  # 动作 0 = 站立
    # ⚠ 稳定阶段必须**持续命令蹲姿**：动作 0 意味着关节目标 0（站立），
    # 会在 30 步内把蹲姿顶回去（实测膝角从 -1.40 被拉到 -0.09）。
    crouch_act = torch.zeros(env.action_space.shape, device=dev)
    hip_c = hip_for_knee(args_cli.crouch_knee)
    for i in kidx:
        crouch_act[:, i] = args_cli.crouch_knee / LEG_SCALE
    for i in hidx:
        crouch_act[:, i] = hip_c / LEG_SCALE
    for _ in range(60):  # 60 步 × 20 ms = 1.2 s 缓入蹲姿
        env.step(crouch_act)
    z_before = robot.data.root_pos_w[:, 2].mean().item()
    gz_before = robot.data.projected_gravity_b[:, 2].mean().item()
    print(f"[legdiag] 蹲姿稳定后 base_z = {z_before:.3f} m, projected_gravity_z = {gz_before:+.2f}")
    # ⚠ 硬断言：竖直站立时 projected_gravity_z ≈ -1。若接近 0，说明机器人是躺着的，
    # 此时测到的"抬升不足"是塌姿读数，不代表腿的伸展能力（此前多次误判即源于此）。
    if gz_before > -0.8:
        print("[legdiag] ❌ 前置条件不满足：机器人不是竖直姿态（|g_z|≈0 表示躺倒），"
              "本次测量无效，已中止。")
        env.close()
        return 2

    # 饱和伸展：膝 +0.663、髋 +0.40、轮 0（锁定）
    act = torch.zeros(env.action_space.shape, device=dev)

    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    bnames = list(sensor.body_names) if sensor is not None else []
    wl_i = bnames.index("wheel_L_Link") if "wheel_L_Link" in bnames else None
    wr_i = bnames.index("wheel_R_Link") if "wheel_R_Link" in bnames else None
    contact_hist = []
    pitch_hist = []
    z_max, vz_max = z_before, -9.9
    knee_log = []
    pen_log = []  # 轮底穿透量（负 = 穿入地面）
    last_hip = []
    k_from, k_to = args_cli.crouch_knee, args_cli.extend_knee
    for step in range(args_cli.hold_steps):
        # 线性推进 knee，并按竖直轨迹同步给 hip（关键：髋必须跟着动）
        frac = min(1.0, (step + 1) / max(1, args_cli.hold_steps // 2))
        knee_cmd = k_from + frac * (k_to - k_from)
        hip_cmd = hip_for_knee(knee_cmd)
        act[:, kidx[0]] = knee_cmd / LEG_SCALE
        act[:, kidx[1]] = knee_cmd / LEG_SCALE
        act[:, hidx[0]] = hip_cmd / LEG_SCALE
        act[:, hidx[1]] = hip_cmd / LEG_SCALE
        env.step(act)
        z = robot.data.root_pos_w[:, 2]
        vz = robot.data.root_lin_vel_w[:, 2]
        z_max = max(z_max, z.mean().item())
        vz_max = max(vz_max, vz.mean().item())
        knee_log.append(robot.data.joint_pos[:, kidx[0]].mean().item())
        # 轮底世界高度（近似）：轮 link 原点 - 0.127，用于判断是否压穿地面
        jn_all = robot.joint_names
        widx = [jn_all.index("wheel_L_Joint"), jn_all.index("wheel_R_Joint")]
        wz = robot.data.joint_pos.new_zeros(())  # 占位避免未用变量告警
        pen_log.append(robot.data.root_pos_w[:, 2].mean().item())
        last_hip.append(robot.data.joint_pos[:, hidx[0]].mean().item())
        if sensor is not None and wl_i is not None:
            f = sensor.data.net_forces_w
            if hasattr(f, "torch"):
                f = f.torch
            c = (torch.linalg.norm(f[:, [wl_i, wr_i]], dim=-1) > 1.0).any(dim=1).float().mean().item()
            contact_hist.append(c)
        # 机身倾角（projected gravity 与竖直的夹角）
        gz = robot.data.projected_gravity_b[:, 2].mean().item()
        pitch_hist.append(gz)

    print("=" * 60)
    print(f"[legdiag] 伸展 {args_cli.hold_steps} 步（每步 {env.unwrapped.step_dt*1000:.1f} ms）")
    print(f"[legdiag] 起始 base_z = {z_before:.3f}")
    print(f"[legdiag] base_z 峰值 = {z_max:.3f}   抬升 = {z_max - z_before:+.3f} m")
    print(f"[legdiag] 竖直速度峰值 = {vz_max:+.3f} m/s")
    print(f"[legdiag] 膝角轨迹: 起 {knee_log[0]:+.2f} -> 末 {knee_log[-1]:+.2f}（目标 {args_cli.extend_knee:+.2f}）")
    # 轮体世界高度（用 body 位置，最准）
    body_names = robot.body_names
    try:
        wl = body_names.index("wheel_L_Link")
        wr = body_names.index("wheel_R_Link")
        wheel_bottom = min(
            robot.data.body_pos_w[:, wl, 2].mean().item(),
            robot.data.body_pos_w[:, wr, 2].mean().item(),
        ) - 0.127
        print(f"[legdiag] 末尾轮底世界高度 = {wheel_bottom:+.3f} m（负值=压入地面）")
    except Exception as e:  # noqa: BLE001
        print(f"[legdiag] 轮底量测失败: {e}")
    print(f"[legdiag] 末尾 hip 实际 = {last_hip[-1]:+.2f}（指令 {-0.25 if True else 0:+.2f} 量级）")
    if contact_hist:
        first_lost = next((i for i, c in enumerate(contact_hist) if c < 0.5), None)
        print(f"[legdiag] 轮接地比例轨迹: 首帧 {contact_hist[0]:.2f} → 末帧 {contact_hist[-1]:.2f}"
              f"；首次失去接地在第 {first_lost} 步" if first_lost is not None
              else f"[legdiag] 轮接地比例: 全程 {min(contact_hist):.2f}~{max(contact_hist):.2f}（未完全离地）")
    if pitch_hist:
        print(f"[legdiag] projected_gravity_z: 起 {pitch_hist[0]:+.2f} → 末 {pitch_hist[-1]:+.2f}"
              f"（-1 = 完全竖直，0 = 倒成水平）")
    print(f"[legdiag] 判定: {'✅ 腿伸展能抬起机身' if (z_max - z_before) > 0.08 else '❌ 抬不起来（执行器/腿模型问题）'}")
    print("=" * 60)
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

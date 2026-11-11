"""统一技能评测：对一个 checkpoint 量化 5 项技能，输出可比较的数字（headless）。

每项技能单独 rollout（各跑 --steps 步），打印 `[skills] <name>: <指标>=<数值>` 与
PASS/FAIL/SKIP，最后一行是 compact 汇总，便于训练回归时 grep：

    cd /home/yangg/data/isaacsim/IsaacLab
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python \
        ./_isaac_sim/python.sh <ws>/isaac/eval_skills.py --visualizer none \
        --num_envs 32 --steps 400 \
        --checkpoint logs/rsl_rl/nezha_flat/<run>/model_<n>.pt

技能与 PASS 条件（阈值见文件顶部常量，按"好策略"实测校准）：
  walk_fast      vx=1.0, h=0.60 : |vx-cmd|<0.20 且 双轮接地>0.80 且 摔倒率=0
  walk_stand     vx=0.6, h=0.60 : 同上
  crouch_walk    vx=0.4, h=0.35 : 同上 + |base_z-0.35|<0.06
  jump_obstacle  vx=1.0         : 越障成功率>0.50；越线瞬间要求腾空
                                  （z>站立+0.02 或 双轮离地 或 vz>0）
  getup                         : 倒地起点中站起比例>0.50，且起身后再走 100 步
                                  双轮接地>0.80

getup 的评测窗口固定 12.5 s（约 625 步；--steps 更大时按其值）。每个技能各做一次
独立 rollout，最后一行 `[skills-summary] ...` 是 5 项 PASS/FAIL/SKIP + 关键数字。

参考实测（16 envs × 400 steps，cmd vx=1.0）：
  正确轮式行走（S06 纯行走）: 双轮接地 98.5%，abad≈-0.12，腿 |vel| 0.2~0.5
  错误腿式行走（S06-Full）  : 双轮接地 0.2%，abad≈+0.20，腿 |vel| 2.4~2.7
因此 abad_mean / leg_vel_mean 是"靠轮走还是靠腿走"的核心判据，每项技能都会打印。

行走三项会把场景里的障碍物挪到 60 m 外：walk_* 测的是平地行走/蹲行，越障由
jump_obstacle 单独评测，否则 0.12-0.20 m 的障碍物会污染速度与接地指标。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha-S06-Full")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400, help="每项技能的 rollout 步数")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# ---- 机器人几何 / 判定阈值 ----
CONTACT_THRESHOLD = 1.0  # N，轮接地阈值（与 mdp 里的接触阈值一致）
BASE_CONTACT_THRESHOLD = 5.0  # N，底盘触地（摔倒）判定
FALL_Z = 0.25  # m，机身低于此高度视为倒地
TILT_GZ = -0.5  # projected_gravity_b.z 高于此值 = 倾角 >60°
STAND_Z = 0.602  # m，S06 站立机身高度
JUMP_Z = STAND_Z + 0.02  # m，越障瞬间的腾空判定
GETUP_FALL_Z = 0.45  # m，零动作落地后低于此值视为倒地起点
GETUP_STAND_Z = 0.55  # m，起身成功判定
GETUP_WINDOW_S = 12.5  # s，起身评测时间窗
POST_GETUP_STEPS = 100  # 起身成功后额外行走步数（测是否回到轮式行走）
POST_GETUP_VX = 0.6  # m/s，起身后行走指令

# ---- 技能指令 ----
WALK_FAST_VX = 1.0
WALK_STAND_VX = 0.6
CROUCH_VX = 0.4
STAND_HEIGHT = 0.60
CROUCH_HEIGHT = 0.35
OBSTACLE_PARK_X = 60.0  # m，行走技能把障碍物挪到这里

# ---- PASS 阈值 ----
PASS_VX_ERR = 0.20  # m/s
PASS_WHEEL_CONTACT = 0.80  # 双轮接地比例
PASS_HEIGHT_ERR = 0.06  # m
PASS_JUMP_SUCCESS = 0.50
PASS_GETUP_RATE = 0.50

LEG_JOINT_NAMES = ("abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint")
ABAD_JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint")
WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
BASE_BODY_NAME = "base_link"


class MlpPolicy:
    """rsl_rl 的 actor：obs_normalizer -> mlp(ELU) -> 8 维动作（确定性）。"""

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
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))  # ELU
        return x @ self.w[3] + self.b[3]


def _np(t) -> np.ndarray:
    """张量转 numpy（拷贝，避免之后被仿真写穿）。"""
    if isinstance(t, np.ndarray):
        return t.copy()
    return t.detach().cpu().numpy().copy()


class SkillProbe:
    """接触 / 姿态 / 腿活动的只读探针（轮接地、倒地、abad、腿速度）。"""

    def __init__(self, env):
        self.robot = env.unwrapped.scene["robot"]
        self.sensor = env.unwrapped.scene.sensors.get("contact_forces")
        self.body_idx = {n: i for i, n in enumerate(self.sensor.body_names)} if self.sensor is not None else {}
        self.wheel_ids = [self.body_idx[n] for n in WHEEL_BODY_NAMES if n in self.body_idx]
        self.base_id = self.body_idx.get(BASE_BODY_NAME)
        leg_ids, leg_names = self.robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
        self.leg_ids = list(leg_ids)
        self.abad_ids = [i for i, n in zip(self.leg_ids, leg_names) if n in ABAD_JOINT_NAMES]

    def wheel_contact(self):
        """(both, left, right) 三个 (num_envs,) bool；没有接触传感器时返回 None。"""
        if self.sensor is None or len(self.wheel_ids) != 2:
            return None
        forces = self.sensor.data.net_forces_w
        if forces is None:
            return None
        f = torch.linalg.norm(forces.torch[:, self.wheel_ids], dim=-1)  # (num_envs, 2)
        c = f > CONTACT_THRESHOLD
        return c[:, 0] & c[:, 1], c[:, 0], c[:, 1]

    def fallen(self) -> np.ndarray:
        """(num_envs,) bool：倾角 >60° / 机身低于 0.25 m / 底盘触地。"""
        data = self.robot.data
        tilt = data.projected_gravity_b.torch[:, 2] > TILT_GZ
        low = data.root_pos_w.torch[:, 2] < FALL_Z
        if self.sensor is not None and self.base_id is not None:
            forces = self.sensor.data.net_forces_w
            if forces is not None:
                f = torch.linalg.norm(forces.torch[:, self.base_id], dim=-1)
                low = low | (f > BASE_CONTACT_THRESHOLD)
        return _np(tilt | low)

    def leg_state(self) -> tuple[float, float]:
        """(abad_mean, leg_vel_mean)：张腿时 abad_mean 变正，腿摆得凶时 leg_vel_mean 变大。"""
        pos = self.robot.data.joint_pos.torch
        vel = self.robot.data.joint_vel.torch
        abad = pos[:, self.abad_ids].mean().item() if self.abad_ids else float("nan")
        leg_vel = vel[:, self.leg_ids].abs().mean().item() if self.leg_ids else float("nan")
        return abad, leg_vel


def get_command(env, name: str):
    """取指令张量；任务没有该指令时返回 None。"""
    try:
        return env.unwrapped.command_manager.get_command(name)
    except (KeyError, ValueError):
        return None


def set_commands(env, vx: float, height: float | None):
    """把速度/高度指令覆盖成固定值（指令是原地可写的张量缓冲区）。"""
    cmd = get_command(env, "base_velocity")
    if cmd is not None:
        cmd[:, 0] = vx
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
    hcmd = get_command(env, "height")
    if hcmd is not None and height is not None:
        hcmd[:, 0] = height


def policy_act(env, policy: MlpPolicy, obs) -> torch.Tensor:
    if isinstance(obs, dict):
        obs = obs["policy"]
    return torch.as_tensor(
        policy(np.nan_to_num(_np(obs).astype(np.float64))), dtype=torch.float32, device=env.unwrapped.device
    )


def get_obstacle(env):
    """场景里的障碍物；任务没有 obstacle 时返回 None（越障评测要 SKIP 而不是崩）。"""
    try:
        return env.unwrapped.scene["obstacle"]
    except (KeyError, AttributeError):
        return None


def park_obstacle_far(env):
    """把障碍物挪出行走路径（walk_* 只测平地行走/蹲行，越障由 jump_obstacle 单独测）。"""
    obstacle = get_obstacle(env)
    if obstacle is None:
        return
    pose = obstacle.data.default_root_pose.torch.clone()  # (num_envs, 7)
    pose[:, 0] = OBSTACLE_PARK_X
    pose[:, 1] = 0.0
    pose[:, 2] = 0.075
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=pose.dtype, device=pose.device)
    obstacle.write_root_pose_to_sim_index(root_pose=pose)


def reset_fallen_cfg(env):
    """reset_fallen 事件的配置；任务没有倒地复位时返回 None。"""
    try:
        return env.unwrapped.event_manager.get_term_cfg("reset_fallen")
    except (KeyError, ValueError):
        return None


def set_fall_probability(env, probability: float) -> float | None:
    """改写 reset_fallen 的倒地概率，返回写入值；任务无该事件时返回 None。"""
    term_cfg = reset_fallen_cfg(env)
    if term_cfg is None:
        return None
    term_cfg.params["probability"] = float(probability)
    env.unwrapped.event_manager.set_term_cfg("reset_fallen", term_cfg)
    return float(probability)


def task_fall_probability(env) -> float | None:
    """任务自带的倒地复位概率（训练用 ramp 的终值）。"""
    term_cfg = reset_fallen_cfg(env)
    if term_cfg is None:
        return None
    prob = term_cfg.params.get("probability")
    if prob is None:
        prob = term_cfg.params.get("fall_prob_max", 0.5)
    return float(prob)


def configure_for_eval(env_cfg):
    """固定指令评测：关掉 heading 控制 / 站立环境 / 指令重采样（否则观测里的指令会跳变）。"""
    if not hasattr(env_cfg, "commands"):
        return
    base_velocity = getattr(env_cfg.commands, "base_velocity", None)
    if base_velocity is not None:
        base_velocity.heading_command = False
        base_velocity.ranges.heading = None
        base_velocity.rel_standing_envs = 0.0
        base_velocity.resampling_time_range = (1e9, 1e9)
    height = getattr(env_cfg.commands, "height", None)
    if height is not None:
        height.resampling_time_range = (1e9, 1e9)


def run_walk(env, probe: SkillProbe, policy: MlpPolicy, vx: float, height: float, steps: int, height_mode: bool):
    """匀速行走 / 蹲行：速度跟踪 + 高度跟踪 + 双轮接地 + 摔倒率 + 腿活动。"""
    obs, _ = env.reset()
    has_height = get_command(env, "height") is not None  # 任务没有高度指令时 height_err 记 nan
    set_commands(env, vx, height)
    park_obstacle_far(env)
    # 复位会把指令重采样成随机样本；先走一步空动作刷新观测，之后才计指标
    obs, *_ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    n = env.unwrapped.num_envs
    vx_err, h_err, both_rate, left_rate, right_rate = [], [], [], [], []
    abad, leg_vel = [], []
    fell = np.zeros(n, dtype=bool)
    for _ in range(steps):
        set_commands(env, vx, height)
        obs, _, terminated, _, _ = env.step(policy_act(env, policy, obs))
        data = probe.robot.data
        v = data.root_lin_vel_b.torch[:, 0]
        vx_err.append((v - vx).abs().mean().item())
        z = data.root_pos_w.torch[:, 2]
        if has_height:
            h_err.append((z - height).abs().mean().item())
        contact = probe.wheel_contact()
        if contact is not None:
            both_rate.append(contact[0].float().mean().item())
            left_rate.append(contact[1].float().mean().item())
            right_rate.append(contact[2].float().mean().item())
        a, lv = probe.leg_state()
        abad.append(a)
        leg_vel.append(lv)
        fell |= _np(terminated) | probe.fallen()
        if bool(terminated.any()):
            park_obstacle_far(env)  # 终止重置会重新摆放障碍物，挪回 60 m 外

    metrics = {
        "vx_err": f"{np.mean(vx_err):.3f}",
        "wheel_contact": f"{np.mean(both_rate):.3f}" if both_rate else "nan",
        "wheel_contact_L": f"{np.mean(left_rate):.3f}" if left_rate else "nan",
        "wheel_contact_R": f"{np.mean(right_rate):.3f}" if right_rate else "nan",
        "fall_rate": f"{fell.mean():.3f}",
        "abad_mean": f"{np.mean(abad):+.3f}",
        "leg_vel_mean": f"{np.mean(leg_vel):.3f}",
        "height_err": f"{np.mean(h_err):.3f}" if h_err else "nan",
    }

    fails = []
    if np.mean(vx_err) >= PASS_VX_ERR:
        fails.append(f"vx_err>={PASS_VX_ERR}")
    if not both_rate:
        fails.append("no_contact_sensor")
    elif np.mean(both_rate) <= PASS_WHEEL_CONTACT:
        fails.append(f"wheel_contact<={PASS_WHEEL_CONTACT}")
    if fell.mean() > 0.0:
        fails.append("fall_rate>0")
    if height_mode and h_err and np.mean(h_err) >= PASS_HEIGHT_ERR:
        fails.append(f"height_err>={PASS_HEIGHT_ERR}")
    return (not fails), metrics, "; ".join(fails)


def run_jump(env, probe: SkillProbe, policy: MlpPolicy, vx: float, steps: int):
    """越障：reset 已把障碍物摆到前方 2.5 m；按"越线瞬间腾空"判定成功。"""
    obstacle = get_obstacle(env)
    if obstacle is None:
        return None, {}, "scenario 无 obstacle"
    obs, _ = env.reset()
    set_commands(env, vx, None)
    # 复位会把指令重采样成随机样本；先走一步空动作刷新观测，之后才计指标
    obs, *_ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    n = env.unwrapped.num_envs
    success = np.zeros(n, dtype=bool)
    prev_cross = np.zeros(n, dtype=bool)
    z_peak = np.zeros(n)
    vz_peak = np.zeros(n)
    both_rate, abad, leg_vel = [], [], []
    for _ in range(steps):
        set_commands(env, vx, None)
        obs, _, _, _, _ = env.step(policy_act(env, policy, obs))
        data = probe.robot.data
        x = _np(data.root_pos_w.torch[:, 0])
        z = _np(data.root_pos_w.torch[:, 2])
        vz = _np(data.root_lin_vel_w.torch[:, 2])
        obs_x = _np(obstacle.data.root_pos_w.torch[:, 0])
        z_peak = np.maximum(z_peak, z)
        vz_peak = np.maximum(vz_peak, vz)
        contact = probe.wheel_contact()
        if contact is not None:
            both = _np(contact[0])
            left = _np(contact[1])
            right = _np(contact[2])
            both_rate.append(both.mean())
            # 越线瞬间腾空：机身高于站立+0.02 / 双轮离地 / 机身向上运动
            airborne = (z > JUMP_Z) | (~left & ~right) | (vz > 0.0)
        else:
            airborne = z > JUMP_Z
        crossed_now = (x > obs_x) & ~prev_cross
        success |= crossed_now & airborne
        prev_cross = x > obs_x
        a, lv = probe.leg_state()
        abad.append(a)
        leg_vel.append(lv)

    success_rate = float(success.mean())
    metrics = {
        "success_rate": f"{success_rate:.3f}",
        "wheel_contact": f"{np.mean(both_rate):.3f}" if both_rate else "nan",
        "max_z": f"{z_peak.mean():.3f}",
        "vz_max": f"{vz_peak.mean():+.3f}",
        "abad_mean": f"{np.mean(abad):+.3f}",
        "leg_vel_mean": f"{np.mean(leg_vel):.3f}",
    }
    note = "" if success_rate > PASS_JUMP_SUCCESS else f"success_rate<={PASS_JUMP_SUCCESS}"
    return (success_rate > PASS_JUMP_SUCCESS), metrics, note


def run_getup(env, probe: SkillProbe, policy: MlpPolicy, fall_prob: float, steps: int):
    """起身：倒地起点中站起来的比例 + 起身后再走 100 步是否回到轮式行走。"""
    if fall_prob is None:
        return None, {}, "task 无 reset_fallen 事件"
    set_fall_probability(env, fall_prob)
    env.reset()
    set_commands(env, 0.0, STAND_HEIGHT)  # 起身阶段用中立指令：站起来别乱走
    # 先零动作 10 步让机体落到地上，按高度判定哪些环境是倒地起步
    zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs = None
    for _ in range(10):
        obs, *_ = env.step(zero)
    z0 = _np(probe.robot.data.root_pos_w.torch[:, 2])
    fallen = z0 < GETUP_FALL_Z
    n_fallen = int(fallen.sum())
    if n_fallen == 0:
        return None, {"fallen_ratio": "0.000", "fall_prob": f"{fall_prob:.2f}"}, "无倒地起点"

    both_rate, abad, leg_vel = [], [], []
    reached = np.zeros_like(fallen)
    z = z0
    for _ in range(steps):
        set_commands(env, 0.0, STAND_HEIGHT)
        obs, _, _, _, _ = env.step(policy_act(env, policy, obs))
        z = _np(probe.robot.data.root_pos_w.torch[:, 2])
        reached |= fallen & (z > GETUP_STAND_Z)
        contact = probe.wheel_contact()
        if contact is not None:
            both_rate.append(_np(contact[0]).mean())
        a, lv = probe.leg_state()
        abad.append(a)
        leg_vel.append(lv)

    stand_rate = float(reached[fallen].mean())
    standing = fallen & (z > GETUP_STAND_Z)  # 窗口结束时站着的倒地起点环境
    post_contact = []
    if standing.any():
        for _ in range(POST_GETUP_STEPS):
            set_commands(env, POST_GETUP_VX, STAND_HEIGHT)
            obs, _, _, _, _ = env.step(policy_act(env, policy, obs))
            contact = probe.wheel_contact()
            if contact is not None:
                post_contact.append(_np(contact[0])[standing].mean())

    metrics = {
        "stand_rate": f"{stand_rate:.3f}",
        "fallen_ratio": f"{fallen.mean():.3f}",
        "fall_prob": f"{fall_prob:.2f}",
        "wheel_contact": f"{np.mean(both_rate):.3f}" if both_rate else "nan",
        "post_wheel_contact": f"{np.mean(post_contact):.3f}" if post_contact else "nan",
        "abad_mean": f"{np.mean(abad):+.3f}",
        "leg_vel_mean": f"{np.mean(leg_vel):.3f}",
    }
    fails = []
    if stand_rate <= PASS_GETUP_RATE:
        fails.append(f"stand_rate<={PASS_GETUP_RATE}")
    if post_contact and np.mean(post_contact) <= PASS_WHEEL_CONTACT:
        fails.append(f"post_wheel_contact<={PASS_WHEEL_CONTACT}")
    elif not post_contact and standing.any():
        fails.append("no_contact_sensor")
    return (not fails), metrics, "; ".join(fails)


def report(name: str, passed, metrics: dict, note: str):
    for key, value in metrics.items():
        print(f"[skills] {name}: {key}={value}")
    tag = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    print(f"[skills] {name}: {tag}" + (f" ({note})" if note else ""))
    return name, tag, metrics, note


def print_summary(results):
    keys = (
        "vx_err",
        "height_err",
        "wheel_contact",
        "post_wheel_contact",
        "fall_rate",
        "success_rate",
        "max_z",
        "stand_rate",
        "abad_mean",
        "leg_vel_mean",
    )
    n_pass = n_fail = n_skip = 0
    parts = []
    for name, tag, metrics, note in results:
        if tag == "PASS":
            n_pass += 1
        elif tag == "FAIL":
            n_fail += 1
        else:
            n_skip += 1
        body = ",".join(f"{k}={metrics[k]}" for k in keys if k in metrics)
        if tag == "SKIP":
            detail = f"{body}; {note}" if body and note else (body or note or "skipped")
            parts.append(f"{name}=SKIP({detail})")
        else:
            parts.append(f"{name}={tag}({body})")
    print(f"[skills-summary] {'  '.join(parts)}  | {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    configure_for_eval(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    policy = MlpPolicy(args_cli.checkpoint)
    probe = SkillProbe(env)
    fall_prob = task_fall_probability(env)
    set_fall_probability(env, 0.0)  # 行走技能不要倒地起点

    print("=" * 78)
    print(f"[skills] task={args_cli.task} envs={args_cli.num_envs} steps={args_cli.steps}")
    print(f"[skills] ckpt={args_cli.checkpoint}")
    if probe.sensor is None or len(probe.wheel_ids) != 2:
        print(f"[skills] 警告: 没有 contact_forces 传感器（body 需含 {WHEEL_BODY_NAMES}），轮接地指标为 nan")
    print("-" * 78)

    results = []
    results.append(report("walk_fast", *run_walk(env, probe, policy, WALK_FAST_VX, STAND_HEIGHT, args_cli.steps, False)))
    results.append(report("walk_stand", *run_walk(env, probe, policy, WALK_STAND_VX, STAND_HEIGHT, args_cli.steps, False)))
    results.append(
        report("crouch_walk", *run_walk(env, probe, policy, CROUCH_VX, CROUCH_HEIGHT, args_cli.steps, True))
    )
    results.append(report("jump_obstacle", *run_jump(env, probe, policy, WALK_FAST_VX, args_cli.steps)))
    getup_steps = max(args_cli.steps, int(round(GETUP_WINDOW_S / env.unwrapped.step_dt)))
    results.append(report("getup", *run_getup(env, probe, policy, fall_prob, getup_steps)))

    print("-" * 78)
    print_summary(results)

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

"""跳跃原语：蹲下蓄力 → 饱和力矩弹射 → 腾空收腿 → 落地恢复。

为什么需要原语（而不是纯 RL）：
  MuJoCo 实测（isaac/jump_primitive_check.py --sweep）表明，用 PD 跟踪一条
  「蹲 → 伸」轨迹只能把机身送回站立高度（base_z 0.602→0.603，轮子离地 3 cm），
  因为关节目标一旦到达，扭矩就撤销了。真正的弹射必须**力矩饱和**：
  膝力矩 +120 N·m 持续 0.1~0.35 s，实测 vz 2.92 m/s、轮底离地 0.68~0.73 m；
  再加髋 +60 N·m 时 vz 4.85 m/s、离地 1.05 m。RL 探索在 4000 iters 内没找到这条
  饱和力矩路径（jump_takeoff 奖励始终 1e-2 量级），所以用脚本原语补上。

动作映射（与 Isaac 动作项一致）：
  腿 joint_pos：target = 0 + 0.25 * action  → action = target_rad / 0.25
  轮 wheel_vel： target = 10 * action       → action = target_rad_s / 10
  目标超出关节限位时 PD 自然饱和到力矩上限（= 我们要的弹射效果）。

相位时间（默认值来自 MuJoCo 实测）：
  crouch 0.45 s（膝 → -1.40）
  extend 0.20 s（膝动作 +8 饱和，髋 +2 助推）
  tuck   0.25 s（膝 → -1.20，轮速保持）
  recover 0.30 s（回到策略控制）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

# 策略实测导出的竖直参考轨迹（knee 0.35→0.60 高度指令的动力学有效动作，120 帧）
_REF_PATH = Path(__file__).with_name("vertical_ref.json")
_REF = None
if _REF_PATH.exists():
    _REF = json.loads(_REF_PATH.read_text())

LEG_SCALE = 0.25
WHEEL_SCALE = 10.0
# 骑行姿态（实测好策略：base_z 0.467、hip≈+0.40、knee≈-0.92）
HIP_RIDE = 0.40
KNEE_RIDE = -0.92
KNEE_CROUCH = -1.40
KNEE_TUCK = -1.20
# 伸展目标：正好到关节限位 +0.663（action = 0.663/0.25 ≈ 2.65）。
# 原来用 +8（target 2.0 rad）会让腿在离地前撞限位，Isaac 的限位约束吸收冲量，
# 弹射失效（实测 base_z 只到 0.604）。配合 stiffness 150 时，这个目标仍能让力矩
# 在行程中维持在 90 N·m 上限附近，并在接近限位时自然衰减。
EXTEND_TARGET_RAD = 0.663
EXTEND_ACTION = EXTEND_TARGET_RAD / LEG_SCALE
HIP_PUSH_ACTION = 2.0
# ⚠ 轮速方向：诊断显示 vx 为正时 wheel 关节速度为负（正转 = 世界 -x）。
# 因此「向前」要用负的轮速指令，否则原语会让机器人倒退。
WHEEL_ACTION = -1.0  # -10 rad/s ≈ 前进 1.27 m/s（腾空相位）
WHEEL_CROUCH_ACTION = -0.4  # 蓄力期保留 0.5 m/s 前进，避免起跳变成纯竖直
WHEEL_EXTEND_ACTION = -0.8  # 弹射期 1.0 m/s 前进，与竖直弹射合成抛物线


@dataclass
class JumpTiming:
    # 时序按「以 1 m/s 前进、0.6 m 触发」标定：
    # 蓄力 0.30 s 走 0.30 m → 起跳点距障碍 0.30 m；弹射+收腿 0.40 s 覆盖 0.40 m
    # → 正好在障碍上方腾空。crouch 0.45 s 会让起跳点离障碍太远，落点在障碍上。
    crouch: float = 0.30
    extend: float = 0.16
    tuck: float = 0.22
    recover: float = 0.25

    @property
    def total(self) -> float:
        return self.crouch + self.extend + self.tuck + self.recover


class JumpPrimitive:
    """按时间推进的跳跃动作发生器。

    用法（在环境步循环里）：
        prim = JumpPrimitive(num_envs, device)
        prim.trigger(active_mask)          # 障碍物进入触发距离时调用
        act = prim.actions()               # (num_envs, 8)；未触发的环境返回 None 表示用策略
    """

    def __init__(self, num_envs: int, device: str | torch.device, timing: JumpTiming | None = None,
                 knee_crouch: float = KNEE_CROUCH, knee_tuck: float = KNEE_TUCK,
                 cooldown_s: float = 1.5):
        self.n = num_envs
        self.device = device
        self.t = timing or JumpTiming()
        self.knee_crouch = knee_crouch
        self.knee_tuck = knee_tuck
        self._clock = torch.full((num_envs,), -1.0, device=device)  # <0 表示空闲
        self._active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 冷却：一次跳跃结束后 cooldown_s 内不再触发，避免"反复起跳打断行驶"
        # （实测频繁触发时 8 s 只前进 0.80 m，正常应 ~7 m）
        self.cooldown_s = cooldown_s
        self._cooldown = torch.zeros(num_envs, device=device)

    # ---- 状态
    @property
    def active(self) -> torch.Tensor:
        return self._active

    def trigger(self, mask: torch.Tensor) -> None:
        """把 mask 里为 True、当前空闲且不在冷却期的环境启动跳跃。"""
        ready = (self._clock < 0) & (self._cooldown <= 0)
        start = mask & ready
        self._clock = torch.where(start, torch.zeros_like(self._clock), self._clock)
        self._active = self._clock >= 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._clock.fill_(-1.0)
        else:
            self._clock[env_ids] = -1.0
        self._active = self._clock >= 0

    def advance(self, dt: float) -> None:
        """每个环境步推进一次；结束的环境进入冷却期。"""
        self._clock = torch.where(self._clock >= 0, self._clock + dt, self._clock)
        done = self._clock > self.t.total
        self._clock = torch.where(done, torch.full_like(self._clock, -1.0), self._clock)
        # 刚结束 → 开始冷却
        self._cooldown = torch.where(done, torch.full_like(self._cooldown, self.cooldown_s), self._cooldown)
        self._cooldown = (self._cooldown - dt).clamp(min=0.0)
        self._active = self._clock >= 0

    # ---- 动作
    def actions(self) -> torch.Tensor:
        """(num_envs, 8) 归一化动作；空闲环境全 0（调用方应改用策略输出）。

        动作顺序（Isaac 侧）：abad_L, hip_L, knee_L, abad_R, hip_R, knee_R, wheel_L, wheel_R

        若 `vertical_ref.json` 存在（策略实测的 h 0.35→0.60 动力学有效轨迹帧），
        则按参考轨迹回放；实测可产生 vz ≈ +1.1 m/s（弹道 ≈0.07 m，越障 ≈0.10–0.12 m）。
        否则退回手写 keyframe。
        """
        if _REF is not None:
            return self._actions_from_ref()
        a = torch.zeros((self.n, 8), device=self.device)
        c = self._clock
        t = self.t
        # 相位掩码
        pre = (c >= 0) & (c < t.crouch)
        ext = (c >= t.crouch) & (c < t.crouch + t.extend)
        tuck = (c >= t.crouch + t.extend) & (c < t.crouch + t.extend + t.tuck)

        knee_crouch_act = self.knee_crouch / LEG_SCALE
        knee_tuck_act = self.knee_tuck / LEG_SCALE

        # 蓄力：膝下蹲、髋保持、轮子保持滚动
        a[pre, 2] = knee_crouch_act
        a[pre, 5] = knee_crouch_act
        # 弹射：膝饱和伸展 + 髋助推 + 轮子保持
        a[ext, 2] = EXTEND_ACTION
        a[ext, 5] = EXTEND_ACTION
        a[ext, 1] = HIP_PUSH_ACTION
        a[ext, 4] = HIP_PUSH_ACTION
        # 腾空收腿
        a[tuck, 2] = knee_tuck_act
        a[tuck, 5] = knee_tuck_act
        # ⚠ 轮子在「蓄力 + 弹射」相位必须**锁死（目标 0 速度 = 刹车）**：
        # 否则腿伸展的反作用力被轮子的滚动吃掉，变成向前加速而不是把机身抬起来
        # （实测：轮子保持 -10 rad/s 时 knee 正常从 -1.51 打到 +1.00，
        #  但 base_z 峰值只有 0.604，完全没离地）。轮子当发射基座，腾空后再恢复滚动。
        # 轮速采用「部分保留前进」：完全锁死会只剩竖直速度，落地撞回障碍
        # （0.12m 障碍实测：base_z +0.15m 但 dx 回到 1.2m = 被推回起点）。
        # 蓄力期保留 0.5 m/s、弹射期 1.0 m/s、腾空 1.27 m/s，合成抛物线。
        a[pre, 6] = WHEEL_CROUCH_ACTION
        a[pre, 7] = WHEEL_CROUCH_ACTION
        a[ext, 6] = WHEEL_EXTEND_ACTION
        a[ext, 7] = WHEEL_EXTEND_ACTION
        a[tuck, 6] = WHEEL_ACTION
        a[tuck, 7] = WHEEL_ACTION

        # 极简姿态：蓄力/弹射/收腿三相位都显式给髋目标（骑行姿态 ~ 助推），
        # 避免策略在该窗口输出与弹射冲突的髋角。
        a[pre, 1] = HIP_RIDE / LEG_SCALE
        a[pre, 4] = HIP_RIDE / LEG_SCALE
        a[tuck, 1] = HIP_RIDE / LEG_SCALE
        a[tuck, 4] = HIP_RIDE / LEG_SCALE
        return a

    def trigger_distance(self, vx, vz_takeoff: float = 1.0) -> torch.Tensor:
        """闭环触发距离：vx * (蹲下时长 + 飞行时间)。

        飞行时间 T ≈ 2*vz/g（vz=1.0 → 0.20 s）。
        必须在蹲下**之前**触发，所以加上 crouch 相位时长：
        以 vx=1.0 为例 → d = 1.0 * (0.30 + 0.20) = 0.50 m。
        固定 0.9 m 会起跳过早、落在障碍前（实测 8 s 只前进 0.42 m 就被顶住）。
        """
        g = 9.81
        t_flight = 2.0 * vz_takeoff / g
        d = torch.as_tensor(vx, device=self.device, dtype=torch.float32) * (self.t.crouch + t_flight)
        return d.clamp(min=0.15, max=1.2)

    def _actions_from_ref(self) -> torch.Tensor:
        """按时间在参考轨迹上取帧，叠加轮速前馈。"""
        a = torch.zeros((self.n, 8), device=self.device)
        frames = _REF["frames"]
        dt_ref = float(_REF["dt"])
        n_f = len(frames)
        c = self._clock
        # 相位映射：crouch 保持第 0 帧；extend 按 speed 倍速走完轨迹；之后保持末帧
        t = self.t
        idx = torch.full_like(c, -1, dtype=torch.long)
        # 蓄力段：用第 0 帧（策略蹲姿）
        pre = (c >= 0) & (c < t.crouch)
        ext = (c >= t.crouch) & (c < t.crouch + t.extend)
        post = c >= t.crouch + t.extend
        idx[pre] = 0
        if ext.any():
            frac = ((c[ext] - t.crouch) / max(t.extend, 1e-6)).clamp(0.0, 1.0)
            idx[ext] = (frac * (n_f - 1)).long()
        idx[post] = n_f - 1
        # 写入关节目标（顺序与 leg_joint_order 一致：abad_L, hip_L, knee_L, abad_R, hip_R, knee_R）
        order = _REF["leg_joint_order"]
        slot = {n: k for k, n in enumerate(order)}
        device_order = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
                        "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"]
        frames_t = torch.tensor(frames, device=self.device)  # (n_f, 6)
        active = idx >= 0
        if active.any():
            picked = frames_t[idx.clamp(min=0)]  # (n, 6)
            for pos, name in enumerate(device_order):
                if name in slot:
                    a[:, pos] = torch.where(active, picked[:, slot[name]] / LEG_SCALE,
                                            torch.zeros_like(a[:, pos]))
        # 轮速：蓄力/伸展保持低前进速度，末段恢复
        a[pre | ext, 6] = WHEEL_CROUCH_ACTION
        a[pre | ext, 7] = WHEEL_CROUCH_ACTION
        a[post, 6] = WHEEL_ACTION
        a[post, 7] = WHEEL_ACTION
        return a

    def phase_name(self) -> list[str]:
        """调试用：每个环境当前相位名。"""
        names = []
        c = self._clock.cpu()
        t = self.t
        for v in c:
            if v < 0:
                names.append("idle")
            elif v < t.crouch:
                names.append("crouch")
            elif v < t.crouch + t.extend:
                names.append("extend")
            elif v < t.crouch + t.extend + t.tuck:
                names.append("tuck")
            else:
                names.append("recover")
        return names

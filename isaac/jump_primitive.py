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

from dataclasses import dataclass

import torch

LEG_SCALE = 0.25
WHEEL_SCALE = 10.0
# 骑行姿态（实测好策略：base_z 0.467、hip≈+0.40、knee≈-0.92）
HIP_RIDE = 0.40
KNEE_RIDE = -0.92
KNEE_CROUCH = -1.40
KNEE_TUCK = -1.20
EXTEND_ACTION = 8.0  # 饱和（target = 2.0 rad，远超 knee 上限 0.663）
HIP_PUSH_ACTION = 2.0
# ⚠ 轮速方向：诊断显示 vx 为正时 wheel 关节速度为负（正转 = 世界 -x）。
# 因此「向前」要用负的轮速指令，否则原语会让机器人倒退。
WHEEL_ACTION = -1.0  # -10 rad/s ≈ 前进 1.27 m/s


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
                 knee_crouch: float = KNEE_CROUCH, knee_tuck: float = KNEE_TUCK):
        self.n = num_envs
        self.device = device
        self.t = timing or JumpTiming()
        self.knee_crouch = knee_crouch
        self.knee_tuck = knee_tuck
        self._clock = torch.full((num_envs,), -1.0, device=device)  # <0 表示空闲
        self._active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # ---- 状态
    @property
    def active(self) -> torch.Tensor:
        return self._active

    def trigger(self, mask: torch.Tensor) -> None:
        """把 mask 里为 True 且当前空闲的环境启动跳跃（计时从 0 开始）。"""
        idle = self._clock < 0
        start = mask & idle
        self._clock = torch.where(start, torch.zeros_like(self._clock), self._clock)
        self._active = self._clock >= 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._clock.fill_(-1.0)
        else:
            self._clock[env_ids] = -1.0
        self._active = self._clock >= 0

    def advance(self, dt: float) -> None:
        """每个环境步推进一次；超时自动退出。"""
        self._clock = torch.where(
            self._clock >= 0, self._clock + dt, self._clock
        )
        done = self._clock > self.t.total
        self._clock = torch.where(done, torch.full_like(self._clock, -1.0), self._clock)
        self._active = self._clock >= 0

    # ---- 动作
    def actions(self) -> torch.Tensor:
        """(num_envs, 8) 归一化动作；空闲环境全 0（调用方应改用策略输出）。

        动作顺序（Isaac 侧）：abad_L, hip_L, knee_L, abad_R, hip_R, knee_R, wheel_L, wheel_R
        """
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
        # 轮子全相位保持滚动（落地后继续前进）
        for phase in (pre, ext, tuck):
            a[phase, 6] = WHEEL_ACTION
            a[phase, 7] = WHEEL_ACTION

        # 极简姿态：蓄力/弹射/收腿三相位都显式给髋目标（骑行姿态 ~ 助推），
        # 避免策略在该窗口输出与弹射冲突的髋角。
        a[pre, 1] = HIP_RIDE / LEG_SCALE
        a[pre, 4] = HIP_RIDE / LEG_SCALE
        a[tuck, 1] = HIP_RIDE / LEG_SCALE
        a[tuck, 4] = HIP_RIDE / LEG_SCALE
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

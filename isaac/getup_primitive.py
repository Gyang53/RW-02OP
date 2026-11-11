"""起身原语（仰卧可解）：折髋收轮 → 轮扭矩翻转 → 起身站立。

来源（MuJoCo 探针，S06 几何、tau≤90 N·m）：
  仰卧（pitch −90°）落地后 z≈0.202；
  「折髋 hip→−1.0 把轮收到身体下方 0.5 s」+「轮扭矩 45 N·m 翻转 2 s」
  → z 峰值 0.686、末段 0.679，**成功站起**（轮扭矩 ≤25 N·m 则失败）。
  俯卧（pitch +90°）用同一套时序会翻过头（峰值 z 1.169）再落回 0.205，
  需要闭环（见文档），本模块只承诺仰卧。

在 Isaac 动作空间的映射：
  腿 joint_pos：target = 0 + 0.25·action → action = rad / 0.25
  轮 wheel_vel ：target = 10·action       → 用「大目标速度」制造饱和轮扭矩
                  （速度环 damping=2，目标超前 20 rad/s ≈ 40 N·m，接近 MuJoCo 的 45）

阶段（feedback）：fold 0.5 s → flip（直到 base_z>0.55 或 2.5 s）→ stand 0.5 s
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LEG_SCALE = 0.25
ABAD_PLANT = -0.32  # 腿张开，轮子更能吃到地面（v1 文档配方）
KNEE_PLANT = 0.663
HIP_FOLD = -1.0
HIP_FLIP = -0.3
KNEE_STAND = -0.92
HIP_STAND = 0.40
WHEEL_FLIP_ACTION = 2.0  # target +20 rad/s → 饱和轮扭矩（方向在 Isaac 侧需按实测确认）


@dataclass
class GetUpTiming:
    fold: float = 0.5
    flip_max: float = 2.5
    stand: float = 0.5

    @property
    def total(self) -> float:
        return self.fold + self.flip_max + self.stand


class GetUpPrimitive:
    """仰卧起身动作发生器；`stand_z` 达到后提前结束翻转相位。"""

    def __init__(self, num_envs: int, device, timing: GetUpTiming | None = None,
                 stand_z: float = 0.55, wheel_flip_action: float = WHEEL_FLIP_ACTION):
        self.n = num_envs
        self.device = device
        self.t = timing or GetUpTiming()
        self.stand_z = stand_z
        self.wheel_flip_action = wheel_flip_action
        self._clock = torch.full((num_envs,), -1.0, device=device)
        self._active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def active(self) -> torch.Tensor:
        return self._active

    def trigger(self, mask: torch.Tensor) -> None:
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

    def advance(self, dt: float, base_z: torch.Tensor | None = None) -> None:
        """推进；`base_z` 用于翻转相位的提前收敛（站起来就进入 stand）。"""
        self._clock = torch.where(self._clock >= 0, self._clock + dt, self._clock)
        if base_z is not None:
            in_flip = (self._clock >= self.t.fold) & (self._clock < self.t.fold + self.t.flip_max)
            upright = base_z > self.stand_z
            push = in_flip & upright
            self._clock = torch.where(push, torch.full_like(self._clock, self.t.fold + self.t.flip_max), self._clock)
        done = self._clock > self.t.total
        self._clock = torch.where(done, torch.full_like(self._clock, -1.0), self._clock)
        self._active = self._clock >= 0

    def actions(self) -> torch.Tensor:
        """(num_envs, 8)：abad_L, hip_L, knee_L, abad_R, hip_R, knee_R, wheel_L, wheel_R。"""
        a = torch.zeros((self.n, 8), device=self.device)
        c = self._clock
        fold = (c >= 0) & (c < self.t.fold)
        flip = (c >= self.t.fold) & (c < self.t.fold + self.t.flip_max)
        stand = c >= self.t.fold + self.t.flip_max

        def set_legs(mask, hip, knee):
            a[mask, 1] = hip / LEG_SCALE
            a[mask, 4] = hip / LEG_SCALE
            a[mask, 2] = knee / LEG_SCALE
            a[mask, 5] = knee / LEG_SCALE

        set_legs(fold, HIP_FOLD, KNEE_PLANT)
        a[fold, 0] = ABAD_PLANT / LEG_SCALE
        a[fold, 3] = ABAD_PLANT / LEG_SCALE

        set_legs(flip, HIP_FLIP, KNEE_PLANT)
        a[flip, 6] = self.wheel_flip_action
        a[flip, 7] = self.wheel_flip_action

        set_legs(stand, HIP_STAND, KNEE_STAND)
        return a

    def phase_name(self) -> list[str]:
        names = []
        c = self._clock.cpu()
        for v in c:
            if v < 0:
                names.append("idle")
            elif v < self.t.fold:
                names.append("fold")
            elif v < self.t.fold + self.t.flip_max:
                names.append("flip")
            else:
                names.append("stand")
        return names

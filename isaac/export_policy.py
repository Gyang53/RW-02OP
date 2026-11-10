"""把 Isaac Lab(rsl_rl) 训练出的 actor 导出为 TorchScript。

    python3 isaac/export_policy.py \
        --checkpoint /home/yangg/data/isaacsim/IsaacLab/logs/rsl_rl/nezha_flat/<run>/model_1499.pt \
        --out isaac/policies/nezha_flat_isaac.pt

导出模型输入：原始观测（36 维旧 checkpoint / 37 维含 height_cmd_norm 的新 checkpoint，
维度取自 checkpoint 的 mlp.0.weight），输出 8 维确定性动作。
观测顺序（与 Isaac Lab 的 ObservationsCfg 一致）：
    base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | velocity_commands(3)
    | joint_pos(8) | joint_vel(8) | last_action(8) [| height_cmd_norm(1)]
动作顺序（Isaac 的 action term 顺序）：
    [abad_L, hip_L, knee_L, abad_R, hip_R, knee_R] 位置 × 0.25
    [wheel_L, wheel_R] 速度 × 10.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

ACT_DIM = 8


class IsaacActor(nn.Module):
    """obs_normalizer -> MLP(ELU) -> action。

    输入维度取自 checkpoint（`mlp.0.weight` 的列数）：旧 checkpoint 是 36，
    带机身高度指令的新 checkpoint 是 37。
    """

    def __init__(self, state_dict: dict):
        super().__init__()
        self.obs_dim = int(state_dict["mlp.0.weight"].shape[1])
        self.register_buffer("mean", state_dict["obs_normalizer._mean"].float())
        self.register_buffer("var", state_dict["obs_normalizer._var"].float())
        self.mlp = nn.Sequential(
            nn.Linear(self.obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, ACT_DIM),
        )
        self.mlp[0].weight.data = state_dict["mlp.0.weight"].float()
        self.mlp[0].bias.data = state_dict["mlp.0.bias"].float()
        self.mlp[2].weight.data = state_dict["mlp.2.weight"].float()
        self.mlp[2].bias.data = state_dict["mlp.2.bias"].float()
        self.mlp[4].weight.data = state_dict["mlp.4.weight"].float()
        self.mlp[4].bias.data = state_dict["mlp.4.bias"].float()
        self.mlp[6].weight.data = state_dict["mlp.6.weight"].float()
        self.mlp[6].bias.data = state_dict["mlp.6.bias"].float()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / torch.sqrt(self.var + 1e-8)
        x = torch.clamp(x, -10.0, 10.0)
        return self.mlp(x)


class ExportPolicy(nn.Module):
    def __init__(self, actor: IsaacActor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt_path.with_name("policy_isaac.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = IsaacActor(ck["actor_state_dict"]).eval()
    scripted = torch.jit.script(ExportPolicy(actor))
    with torch.no_grad():
        dummy = scripted(torch.zeros(1, actor.obs_dim))
    assert dummy.shape == (1, ACT_DIM), dummy.shape
    scripted.save(str(out))
    print(f"[export] {out}  input={actor.obs_dim} output={tuple(dummy.shape)} "
          f"range=[{dummy.min():.3f}, {dummy.max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

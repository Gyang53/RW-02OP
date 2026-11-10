"""把 rl/train.py 微调出的 checkpoint 导出为 TorchScript（37 维输入 → 8 维动作）。

    python3 rl/export_policy.py --ckpt rl/runs/nezha_mujoco/final.pt \
        --out isaac/policies/nezha_mujoco_ft.pt

导出格式与 isaac/export_policy.py 完全一致，可直接用 isaac/policy_node_isaac.py 跑。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 37
ACT_DIM = 8


class ExportPolicy(nn.Module):
    """obs 归一化 -> MLP(ELU) -> 动作（无 tanh，与 rsl_rl 一致）。"""

    def __init__(self, net, mean, var):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(var, dtype=torch.float32))
        self.actor_body = net.actor_body
        self.actor_head = net.actor_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / torch.sqrt(self.var + 1e-8)
        x = torch.clamp(x, -10.0, 10.0)
        return self.actor_head(self.actor_body(x))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt_path.with_name("policy.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    ppo_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from ppo import ActorCritic  # 延迟导入，避免路径问题

    net = ActorCritic(OBS_DIM, ACT_DIM, activation="elu", squash=False)
    net.load_state_dict(ppo_ckpt["model"])
    net.eval()

    mean = ppo_ckpt["obs_mean"].numpy() if ppo_ckpt.get("obs_mean") is not None else np.zeros(OBS_DIM)
    var = ppo_ckpt["obs_var"].numpy() if ppo_ckpt.get("obs_var") is not None else np.ones(OBS_DIM)

    scripted = torch.jit.script(ExportPolicy(net, mean, var).eval())
    with torch.no_grad():
        dummy = scripted(torch.zeros(1, OBS_DIM))
    assert dummy.shape == (1, ACT_DIM), dummy.shape
    scripted.save(str(out))
    print(f"[export] {out}  output={tuple(dummy.shape)}  range=[{dummy.min():.3f}, {dummy.max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

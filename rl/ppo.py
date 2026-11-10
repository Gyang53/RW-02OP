"""极简 PPO（适配 VecNezhaEnv，可热启动 Isaac Lab 训练出的 actor）。

- ELU + (256,128,64)，与 Isaac Lab 的 rsl_rl actor 结构一致，可直接加载权重
- 观测归一化（running mean/var），同样可从 Isaac checkpoint 加载
- squash=False 时不压缩动作（与 Isaac 的 Gaussian 策略一致）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def load(self, mean, var, count=1e6):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.var = np.asarray(var, dtype=np.float64).reshape(-1)
        self.count = float(count)


def _act_module(kind: str):
    return nn.ELU() if kind == "elu" else nn.Tanh()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 128, 64),
                 activation: str = "elu", init_log_std=None, squash: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.squash = squash

        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), _act_module(activation)]
            last = h
        self.actor_body = nn.Sequential(*layers)
        self.actor_head = nn.Linear(last, act_dim)

        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), _act_module(activation)]
            last = h
        self.critic_body = nn.Sequential(*layers)
        self.critic_head = nn.Linear(last, 1)

        init_std = torch.full((act_dim,), -1.0) if init_log_std is None else torch.as_tensor(init_log_std, dtype=torch.float32)
        self.log_std = nn.Parameter(init_std.clone())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs):
        return self.actor_head(self.actor_body(obs))

    def value(self, obs):
        return self.critic_head(self.critic_body(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mu = self.forward(obs)
        if deterministic:
            return (torch.tanh(mu) if self.squash else mu), None, None
        dist = torch.distributions.Normal(mu, self.log_std.exp())
        z = dist.sample()
        action = torch.tanh(z) if self.squash else z
        return action, dist.log_prob(z).sum(-1), self.value(obs)


def load_isaac_actor(net: ActorCritic, norm: RunningMeanStd, ckpt_path: str) -> dict:
    """把 rsl_rl checkpoint 的 actor 权重/观测归一化/输出标准差搬到本网络。

    Isaac checkpoint 的观测是 36 维，本网络的观测是 37 维（末尾多了 height_cmd_norm）：
    首层权重复制前 36 列，新增列置 0（高度指令热启动时不起作用，交给微调学）。
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    missing = []

    def copy_param(key, src, dst) -> bool:
        if key not in src:
            return False
        w = src[key].float()
        if w.shape == dst.shape:
            dst.copy_(w)
            return True
        if key == "mlp.0.weight" and w.dim() == 2 and dst.dim() == 2 \
                and w.shape[0] == dst.shape[0] and w.shape[1] < dst.shape[1]:
            dst.zero_()
            dst[:, :w.shape[1]].copy_(w)
            return True
        return False

    mapping = [
        ("mlp.0.weight", net.actor_body[0].weight), ("mlp.0.bias", net.actor_body[0].bias),
        ("mlp.2.weight", net.actor_body[2].weight), ("mlp.2.bias", net.actor_body[2].bias),
        ("mlp.4.weight", net.actor_body[4].weight), ("mlp.4.bias", net.actor_body[4].bias),
        ("mlp.6.weight", net.actor_head.weight), ("mlp.6.bias", net.actor_head.bias),
    ]
    with torch.no_grad():
        for key, tensor in mapping:
            if not copy_param(key, sd, tensor):
                missing.append(key)

    if "obs_normalizer._mean" in sd:
        # 归一化统计同样是 36 维：新增维补 (mean=0, var=1)
        mean = np.asarray(sd["obs_normalizer._mean"].cpu().numpy(), dtype=np.float64).reshape(-1)
        var = np.asarray(sd["obs_normalizer._var"].cpu().numpy(), dtype=np.float64).reshape(-1)
        pad = norm.mean.size - mean.size
        if pad > 0:
            mean = np.concatenate([mean, np.zeros(pad)])
            var = np.concatenate([var, np.ones(pad)])
        norm.load(mean, var, sd.get("obs_normalizer.count", 1e6))
    if "distribution.std_param" in sd:
        # rsl_rl 的 std_param 是标准差本身，本网络用 log_std
        std = sd["distribution.std_param"].float().reshape(-1).clamp(min=1e-3)
        with torch.no_grad():
            net.log_std.copy_(std.log())

    critic = ck.get("critic_state_dict", {})
    cmap = [
        ("mlp.0.weight", net.critic_body[0].weight), ("mlp.0.bias", net.critic_body[0].bias),
        ("mlp.2.weight", net.critic_body[2].weight), ("mlp.2.bias", net.critic_body[2].bias),
        ("mlp.4.weight", net.critic_body[4].weight), ("mlp.4.bias", net.critic_body[4].bias),
        ("mlp.6.weight", net.critic_head.weight), ("mlp.6.bias", net.critic_head.bias),
    ]
    with torch.no_grad():
        for key, tensor in cmap:
            copy_param(key, critic, tensor)

    return {"missing": missing, "iter": ck.get("iter")}


class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, epochs=5, num_minibatches=8, ent_coef=0.005, vf_coef=0.5,
                 max_grad_norm=1.0, device="cpu", init_log_std=None, squash=False,
                 activation="elu"):
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim, init_log_std=init_log_std,
                               squash=squash, activation=activation).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.num_minibatches = num_minibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.squash = squash

    def update(self, rollouts):
        obs = torch.as_tensor(rollouts["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(rollouts["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(rollouts["logp"], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(rollouts["values"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rollouts["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(rollouts["dones"], dtype=torch.float32, device=self.device)
        last_value = torch.as_tensor(rollouts["last_value"], dtype=torch.float32, device=self.device)

        T, N = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(N, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + values

        obs = obs.reshape(T * N, -1)
        act = act.reshape(T * N, -1)
        old_logp = old_logp.reshape(T * N)
        adv = adv.reshape(T * N)
        returns = returns.reshape(T * N)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = T * N
        bs = max(n // self.num_minibatches, 1)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n - bs + 1, bs):
                mb = idx[start:start + bs]
                mu = self.net.forward(obs[mb])
                dist = torch.distributions.Normal(mu, self.net.log_std.exp())
                z = torch.atanh(torch.clamp(act[mb], -0.999999, 0.999999)) if self.squash else act[mb]
                logp = dist.log_prob(z).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (logp - old_logp[mb]).exp()
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[mb]
                pg_loss = -torch.min(surr1, surr2).mean()
                v = self.net.value(obs[mb])
                v_loss = ((v - returns[mb]) ** 2).mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.item()
                updates += 1
        return {k: v / max(updates, 1) for k, v in stats.items()}

    def save(self, path, normalizer=None):
        ckpt = {"model": self.net.state_dict(),
                "obs_mean": None if normalizer is None else torch.as_tensor(normalizer.mean, dtype=torch.float64),
                "obs_var": None if normalizer is None else torch.as_tensor(normalizer.var, dtype=torch.float64),
                "squash": self.squash}
        torch.save(ckpt, path)

    def load(self, path, normalizer=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["model"])
        if normalizer is not None and ckpt.get("obs_mean") is not None:
            normalizer.mean = ckpt["obs_mean"].numpy().astype(np.float64)
            normalizer.var = ckpt["obs_var"].numpy().astype(np.float64)

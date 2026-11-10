"""在 MuJoCo 里并行 PPO 训练 / 微调 nezhav1 策略（与 Isaac Lab 任务同接口）。

热启动 Isaac 训练出的策略：
    python3 rl/train.py --init_from <IsaacLab logs>/rsl_rl/nezha_flat/<run>/model_1499.pt \
        --num-envs 64 --iters 400 --lr 1e-4 --save rl/runs/nezha_mujoco

从零训练（不需要 Isaac）：
    python3 rl/train.py --num-envs 64 --iters 1500 --save rl/runs/nezha_mujoco
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nezha_env import ACT_DIM, OBS_DIM, EnvCfg
from ppo import ActorCritic, PPO, RunningMeanStd, load_isaac_actor


def collect(env, net, norm, horizon, device, ep_state):
    N = env.num_envs
    obs = env.get_obs()
    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones")}
    ep_rew = ep_state["ret"]
    ep_len = ep_state["len"]
    finished = []
    info = None
    for _ in range(horizon):
        nobs = norm.normalize(obs)
        with torch.no_grad():
            a, logp, v = net.act(torch.as_tensor(nobs, dtype=torch.float32, device=device))
        a_np = a.cpu().numpy()
        buf["obs"].append(nobs)
        buf["actions"].append(a_np)
        buf["logp"].append(logp.cpu().numpy())
        buf["values"].append(v.cpu().numpy())
        obs, rew, done, info = env.step(a_np)
        norm.update(obs)
        buf["rewards"].append(rew)
        buf["dones"].append(done)
        ep_rew += rew
        ep_len += 1
        if done.any():
            for i in np.nonzero(done)[0]:
                finished.append((ep_rew[i], ep_len[i], info["fall"][i], info["base_z"][i]))
                ep_rew[i] = 0.0
                ep_len[i] = 0
    with torch.no_grad():
        last_v = net.value(torch.as_tensor(norm.normalize(obs), dtype=torch.float32, device=device))
    out = {k: np.stack(v) for k, v in buf.items()}
    out["last_value"] = last_v.cpu().numpy()

    if finished:
        arr = np.array(finished, dtype=np.float64)
        stats = {"ep_rew": arr[:, 0].mean(), "ep_len": arr[:, 1].mean(),
                 "fall_rate": arr[:, 2].mean(), "base_z": arr[:, 3].mean()}
    else:
        stats = {"ep_rew": float("nan"), "ep_len": float("nan"),
                 "fall_rate": float("nan"), "base_z": float("nan")}
    cmd = env.cmd.cpu().numpy() if torch.is_tensor(env.cmd) else env.cmd
    stats["vx_err"] = float(np.mean(np.abs(info["vx"] - cmd[:, 0])))
    stats["wz_err"] = float(np.mean(np.abs(info["wz"] - cmd[:, 2])))
    return out, stats


@torch.no_grad()
def evaluate(env, net, norm, seconds, device, fixed_vx=None, fixed_wz=0.0):
    """确定性策略跑一段，返回成功率/跟踪误差/高度。"""
    obs = env.reset()
    steps = int(seconds / (env.cfg.physics_dt * env.cfg.decimation))
    N = env.num_envs
    alive = np.ones(N)
    z_sum = np.zeros(N)
    vx_err = np.zeros(N)
    wz_err = np.zeros(N)
    for _ in range(steps):
        if fixed_vx is not None:
            if torch.is_tensor(env.cmd):
                env.cmd[:, 0] = fixed_vx
                env.cmd[:, 1] = 0.0
                env.cmd[:, 2] = fixed_wz
            else:
                env.cmd[:, 0] = fixed_vx
                env.cmd[:, 1] = 0.0
                env.cmd[:, 2] = fixed_wz
        a = net.act(torch.as_tensor(norm.normalize(obs), dtype=torch.float32, device=device),
                    deterministic=True)[0].cpu().numpy()
        obs, _, done, info = env.step(a)
        # 只统计摔倒，不把 timeout 当失败
        alive *= (1.0 - info["fall"])
        z_sum += info["base_z"]
        cmd = env.cmd.cpu().numpy() if torch.is_tensor(env.cmd) else env.cmd
        vx_err += np.abs(info["vx"] - cmd[:, 0])
        wz_err += np.abs(info["wz"] - cmd[:, 2])
    return {
        "success": float(alive.mean()),
        "base_z": float((z_sum / steps).mean()),
        "vx_err": float((vx_err / steps).mean()),
        "wz_err": float((wz_err / steps).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="None=自动选 cuda")
    p.add_argument("--init_from", default=None, help="Isaac Lab 的 rsl_rl checkpoint（热启动）")
    p.add_argument("--resume", default=None, help="本工具自己的 checkpoint")
    p.add_argument("--save", default="rl/runs/nezha_mujoco")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-seconds", type=float, default=20.0)
    p.add_argument("--eval-fixed-vx", type=float, default=None, help="评测时固定 vx 指令")
    p.add_argument("--eval-fixed-wz", type=float, default=0.0)
    p.add_argument("--slope", type=int, default=1, help="1=启用斜木板地形")
    p.add_argument("--slope-angle", type=float, default=None, help="固定斜板角度 (rad)")
    p.add_argument("--slope-x", type=float, default=2.0)
    p.add_argument("--no-obs-noise", action="store_true")
    p.add_argument("--cmd-forward-bias", type=float, default=0.0)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu",
                   help="gpu=mujoco_warp 物理+网络都在 GPU；cpu=muJoCo CPU 物理")
    p.add_argument("--physics-dt", type=float, default=None,
                   help="物理步长；gpu 默认 0.005(200Hz,与 Isaac 一致)，cpu 默认 0.001")
    p.add_argument("--vx-range", type=float, nargs=2, default=(-1.0, 1.0))
    p.add_argument("--wz-range", type=float, nargs=2, default=(-1.0, 1.0))
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[device] {device} | {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = EnvCfg(num_envs=args.num_envs, seed=args.seed,
                 slope_enabled=bool(args.slope), slope_x=args.slope_x,
                 obs_noise=not args.no_obs_noise,
                 cmd_forward_bias=args.cmd_forward_bias,
                 cmd_vx_range=tuple(args.vx_range), cmd_wz_range=tuple(args.wz_range))
    if args.backend == "gpu":
        cfg.physics_dt = args.physics_dt if args.physics_dt is not None else 0.005
    elif args.physics_dt is not None:
        cfg.physics_dt = args.physics_dt
    cfg.decimation = int(round(0.02 / cfg.physics_dt))
    if args.slope_angle is not None:
        cfg.slope_angle_range = (args.slope_angle, args.slope_angle)
        cfg.slope_randomize = False
    if args.backend == "gpu":
        from nezha_env_gpu import VecNezhaEnvGpu
        env = VecNezhaEnvGpu(cfg, device=args.device)
        print(f"[backend] mujoco_warp GPU  nworld={cfg.num_envs}")
    else:
        from nezha_env import VecNezhaEnv
        env = VecNezhaEnv(cfg)

    norm = RunningMeanStd((OBS_DIM,))
    norm.update(env.get_obs())

    ppo = PPO(OBS_DIM, ACT_DIM, lr=args.lr, device=args.device,
              init_log_std=np.full(ACT_DIM, -1.0), squash=False, activation="elu")

    if args.init_from:
        info = load_isaac_actor(ppo.net, norm, args.init_from)
        print(f"[init] 从 Isaac checkpoint 加载: iter={info['iter']} missing={info['missing']}")
    if args.resume:
        ppo.load(args.resume, norm)
        print(f"[resume] {args.resume}")

    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_cfg.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))

    base = evaluate(env, ppo.net, norm, args.eval_seconds, args.device,
                    args.eval_fixed_vx, args.eval_fixed_wz)
    print(f"[eval] init: success={base['success']*100:.1f}% z={base['base_z']:.3f} "
          f"vx_err={base['vx_err']:.3f} wz_err={base['wz_err']:.3f}")
    if args.eval_only:
        return

    t0 = time.time()
    history = [{"iter": 0, "eval": base}]
    ep_state = {"ret": np.zeros(args.num_envs), "len": np.zeros(args.num_envs, dtype=np.int64)}
    for it in range(1, args.iters + 1):
        roll, stats = collect(env, ppo.net, norm, args.horizon, args.device, ep_state)
        losses = ppo.update(roll)
        history.append({"iter": it, **stats, **losses})

        if it % args.log_every == 0 or it == 1:
            sps = it * args.horizon * args.num_envs / (time.time() - t0)
            print(f"it {it:4d} | ep_rew {stats['ep_rew']:7.2f} | ep_len {stats['ep_len']:5.0f} "
                  f"| fall {stats['fall_rate']*100:5.1f}% | z {stats['base_z']:.3f} "
                  f"| verr {stats['vx_err']:.2f} werr {stats['wz_err']:.2f} "
                  f"| pi {losses['pg_loss']:+.3f} v {losses['v_loss']:.3f} H {losses['entropy']:.2f} "
                  f"| {sps:5.0f} sps")
        if it % args.save_every == 0 or it == args.iters:
            ppo.save(out_dir / "checkpoint.pt", norm)
            (out_dir / "history.json").write_text(json.dumps(history))

    final = evaluate(env, ppo.net, norm, args.eval_seconds, args.device,
                     args.eval_fixed_vx, args.eval_fixed_wz)
    print(f"[eval] final: success={final['success']*100:.1f}% z={final['base_z']:.3f} "
          f"vx_err={final['vx_err']:.3f} wz_err={final['wz_err']:.3f}")
    history.append({"iter": args.iters, "eval": final})
    ppo.save(out_dir / "final.pt", norm)
    (out_dir / "history.json").write_text(json.dumps(history))
    print(f"[done] {out_dir / 'final.pt'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

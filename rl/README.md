# rl/ — MuJoCo 并行 PPO（GPU）+ Isaac 策略热启动微调

与 Isaac Lab 任务（`Isaac-Velocity-Flat-Nezha`）**同接口**：37 维观测、8 维动作，
所以 Isaac 训练出的 actor 可以直接加载到这里继续微调，微调结果也能直接给
`isaac/policy_node_isaac.py` 在 mjsim 里跑。

## 两种后端

| backend | 物理 | 网络 | 吞吐（本机） |
|---|---|---|---|
| `gpu`（默认） | `mujoco_warp`（NVIDIA Warp，GPU） | GPU | **≈71k env-step/s**（4096 envs，200 Hz 物理） |
| `cpu` | MuJoCo `mj_step` | GPU | ≈5k env-step/s（64 envs，1 kHz 物理） |

`gpu` 后端默认物理步长 0.005（200 Hz，与 Isaac 训练一致），`cpu` 后端用 0.001（1 kHz，与 mjsim 部署一致）。

> MJX 不支持「圆柱 × box」碰撞（轮子 × 斜板），所以用 `mujoco_warp`（支持完整 MuJoCo 碰撞对）。

## 场景

训练场景自带一块斜木板（`slope`，默认在机器人前方 2 m，角度 -0.25~-0.08 rad 随机），
用来练上下坡能力。用 `--slope 0` 关掉，`--slope-angle` 固定角度，`--slope-x` 改位置。

## 用法

```bash
# 热启动微调（GPU，4096 环境，约 25 分钟 / 400 iter）
python3 rl/train.py \
    --init_from /home/yangg/data/isaacsim/IsaacLab/logs/rsl_rl/nezha_flat/<run>/model_1499.pt \
    --backend gpu --num-envs 4096 --iters 400 --lr 1e-4 \
    --slope 1 --vx-range -0.6 0.6 --wz-range -0.6 0.6 \
    --save rl/runs/nezha_mujoco_gpu

# 只评测（不训练）
python3 rl/train.py --init_from <ckpt> --eval-only --num-envs 256 --backend gpu

# 导出 TorchScript（格式与 isaac/export_policy.py 一致）
python3 rl/export_policy.py --ckpt rl/runs/nezha_mujoco_gpu/final.pt \
    --out isaac/policies/nezha_mujoco_ft.pt

# 在 mjsim 里跑
python3 isaac/policy_node_isaac.py --ros-args -p policy:=isaac/policies/nezha_mujoco_ft.pt
```

## 接口（训练 = 部署）

观测 37 维：
```
base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | cmd(3)
| joint_pos(8) | joint_vel(8) | last_action(8) | height_cmd_norm(1)
```
第 37 维是机身高度指令 `height_cmd_norm = (h - 0.65) / 0.25`，目标高度 `h ∈ [0.45, 0.85]` 米，
每个 episode reset 时均匀采样；部署时用键盘 `z` 蹲下 / `x` 起身（按住即改变，松开保持当前高度）。
动作 8 维，执行器顺序 `abad_L, hip_L, knee_L, wheel_L, abad_R, hip_R, knee_R, wheel_R`：
- 腿：`pos = 0.25a`，`kp=40, kd=2`
- 轮：`vel = 10a`，`kp=0, kd=2`

基座速度统一用 **`R^T · qvel`**（机体系），`sim_manager.get_base_state()` 与训练环境用同一公式。

奖励与 Isaac 的 `RewardsCfg` 对齐：
`track_lin 1.5 / track_ang 0.75 / lin_vel_z -2 / ang_vel_xy -0.05 / flat_ori -2.5 /
torques -2e-4 / dof_acc -2.5e-7 / action_rate -0.01 / pos_limits -0.5 / track_base_height 0.75`。

`track_base_height = exp(-(base_z - height_cmd)^2 / 0.01)`。

## 热启动实现

`rl/ppo.py: load_isaac_actor()` 映射 rsl_rl checkpoint：
`mlp.{0,2,4,6}` → `actor_body/actor_head`、`obs_normalizer._mean/_var` → 归一化、`distribution.std_param` → `log_std`（注意 rsl_rl 的 `std_param` 是标准差本身，需要取 log）。

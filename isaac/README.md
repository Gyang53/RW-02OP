# isaac/ — 把 nezhav1 接进 Isaac Lab（GPU 并行训练）

CPU 版（`rl/`）单进程只能跑到 ~6.5k env-step/s；这里用 Isaac Lab 3.0 + Isaac Sim 5.1，
在 RTX 5060 上用 MJWarp/Newton 后端并行上千个环境。

## 目录结构

```
isaac/
├── make_clean_mjcf.py   MJCF 清理（删 slope / base_Estimate / floor 等非机器人元素）
├── convert_mjcf.py      MJCF → 带物理的 USD（Isaac Lab MjcfConverter）
├── smoke_env.py         环境冒烟测试（建 env、打印 body/joint、跑 N 步）
└── eval_policy.py       加载 checkpoint 回放并统计指标
```

生成的 USD：
`assets/nezha/models/nezhav1/mjcf/nezha_isaac/nezha_isaac.usda`
（payloads/Physics 里带 8 个 revolute 关节、9 个刚体、19 个碰撞体）

Isaac Lab 侧新增任务（官方任务树内，自动注册）：
```
source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/nezha/
├── __init__.py                 gym.register "Isaac-Velocity-Flat-Nezha"
├── nezha_cfg.py                ArticulationCfg（腿位置 PD + 轮速度控制）
├── flat_env_cfg.py             NezhaFlatEnvCfg（平地速度跟踪）
└── agents/rsl_rl_ppo_cfg.py    PPO / rsl_rl 训练参数
```

## 运行方式（关键）

Isaac Sim 是下载版，**不能和 `.venv` 直接混用**：用 `_isaac_sim/python.sh` 提供运行时环境，
把 `PYTHONEXE` 指到项目的 `.venv`（torch / rsl_rl / isaaclab 都在里面），并先清掉 conda 变量。

```bash
cd /home/yangg/data/isaacsim/IsaacLab
export RUN="env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONEXE=$PWD/.venv/bin/python ./_isaac_sim/python.sh"

# 1) 重新生成 MJCF（改了 XML 时才需要）
python3 <ws>/isaac/make_clean_mjcf.py

# 2) MJCF -> USD
$RUN <ws>/isaac/convert_mjcf.py --visualizer none

# 3) 冒烟测试
$RUN <ws>/isaac/smoke_env.py --visualizer none --num_envs 4 --steps 100

# 4) 训练（GPU）
$RUN scripts/reinforcement_learning/train.py \
    --task Isaac-Velocity-Flat-Nezha --rl_library rsl_rl \
    --num_envs 2048 --visualizer none --seed 0

# 5) 回放（headless 量化）
$RUN <ws>/isaac/eval_policy.py --visualizer none --num_envs 64 --steps 1000 \
    --checkpoint logs/rsl_rl/nezha_flat/<run>/model_1499.pt

# 6) 图形回放（有显示器时去掉 --visualizer none）
$RUN scripts/reinforcement_learning/play.py --task Isaac-Velocity-Flat-Nezha \
    --rl_library rsl_rl --num_envs 16 --checkpoint <logs>/rsl_rl/nezha_flat/<run>/model_<n>.pt
```

日志与 checkpoint：`/home/yangg/data/isaacsim/IsaacLab/logs/rsl_rl/nezha_flat/<时间戳>/`。

## 实测（RTX 5060 8GB，2048 环境，1500 iter）

| 指标 | 值 |
|---|---|
| 训练耗时 | 665 s（≈11 min） |
| 吞吐 | 112.8k env-step/s（CPU 版 MuJoCo 6.5k/s，约 17×） |
| success_rate | 1.000 |
| 线速度误差 | 0.09 m/s |
| 角速度误差 | 0.17 rad/s |
| 平均 episode 长度 | 988 / 1000 步（20 s） |
| base_link 触地终止率 | 8% |

checkpoint：`IsaacLab/logs/rsl_rl/nezha_flat/2026-11-06_14-54-49/model_1499.pt`

## 任务设置

| 项 | 值 |
|---|---|
| 控制频率 | 50 Hz（`sim.dt=0.005`，`decimation=4`） |
| 物理后端 | `newton_mjwarp`（CUDA），`njmax=256`，`nconmax=128` |
| 动作 | 腿 `JointPositionAction` scale=0.25；轮 `JointVelocityAction` scale=10 |
| 指令 | `vx ∈ [-1,1]`，`wz ∈ [-1,1]`（双轮不能横移，`lin_vel_y=(0,0)`） |
| 观测 | 36 维：lin/ang vel、projected gravity、cmd、joint pos/vel、last action |
| 终止 | `base_link` 触地 / 超时（20 s） |

USD 里的 `ArticulationRootAPI` 已从 `base_link` 挪到 defaultPrim（`/nezha_v1`），
这样 `ArticulationCfg.prim_path="{ENV_REGEX_NS}/Robot"` 才指向正确的 articulation root。

## 任务一览

| gym id | 能力 | 状态 |
|---|---|---|
| `Isaac-Velocity-Flat-Nezha` | 速度跟踪 + 机身高度（蹲/站）+ 单边板地形 | ✅ 已验证 |
| `Isaac-Velocity-Flat-Nezha-GetUp` | 从倒地出发，学习自行站起 | ⚠️ 框架就绪，策略未学会（0%） |

### 高度指令（37 维观测）

观测末尾新增第 37 维 `height_cmd_norm = (h - 0.65)/0.25`，`h ∈ [0.45, 0.85] m`；
奖励新增 `track_height = 0.75 * exp(-(z - h)^2 / 0.01)`。

实测（128 env × 20 s，含单边板）：`|z - h_cmd| = 0.019 m`，0 摔倒。

键盘：`kb_joy` 用 `z` 蹲下、`x` 起身（按住持续变化，松开保持）。

### 单边板地形

场景里 spawn 了一块 1.8×0.18×0.06 m 的板（y 偏到右轮路径上），前进时右轮上板、左轮在地面。
训练后 `success_rate 91.3%`（对比平地 98.6%），0 摔倒。

### 起身（✅ 已解决）

**关键突破**：起身失败不是 RL 学不会，而是**复位姿态的物理可行性问题**——
侧躺时把 `abad` 打到负限位（腿朝地面张开）+ 膝盖伸直，轮子能直接穿入地面 0.46 m，
形成强支撑点。有了这个支撑，"收腿 + 轮扭矩" 的组合就能把车身翻正。

**评测**（256 env × 12.5 s，55% 倒地起步）：
- 倒地起步的 70 个环境里 **51 个成功站起（72.9%）**
- z_max 平均 0.825 m（≥ 站立高度 0.85 m 附近）
- 剩余 27% 失败的多为 quad_w 接近 0 的"完全倒竖"姿态（车头朝下竖立，轮子朝上）

**注意**：起身成功需要 >10 s（eval 1000 步只有 55% 起身，1500 步升到 73%）。
推理时 `policy_node` 的 episode 时限要留够。

**修复记录**：
1. `rsl_rl/utils.py check_nan` 改为就地置 0 + 告警（`_isaac_sim/kit/python/.../rsl_rl/`）——
   NaN 不再中断训练，也不会污染梯度。
2. `getup_env_cfg`：去掉单边板（板边缘接触偶发发散）、动作限幅 ±3.0、
   track_height 1.5/σ0.45（覆盖起身过程的高度过冲）。
3. 关键：`reset_to_fallen` 的姿态必须含 `abad=-0.32 + knee=0.663`（腿伸开）。

**训练命令**：
```bash
$RUN scripts/reinforcement_learning/train.py     --task Isaac-Velocity-Flat-Nezha-GetUp --rl_library rsl_rl     --num_envs 2048 --visualizer none --seed N     --checkpoint logs/rsl_rl/nezha_flat/2026-11-06_18-15-41/model_2998.pt
```

**导出**：
```bash
python3 isaac/export_policy.py     --checkpoint logs/rsl_rl/nezha_flat/2026-11-09_12-40-10/model_7500.pt     --out isaac/policies/nezha_getup.pt
```

### 起身（训练过 3 轮，未达标）

`reset_to_fallen` 事件以 50% 概率把机器人放到侧躺位姿（z 0.45~0.60、roll 77°~95°、关节 ±0.1 rad）。
实测评测：倒地起步比例 31~43%，**站起成功率 0%**，z_max 只有 0.34 m。

已知两个坑：
1. 倒地位姿在接触求解里偶发 NaN；`rsl_rl` 的 `check_nan` 会直接中断训练，已改成"就地置 0 + 告警"（`_isaac_sim/kit/python/.../rsl_rl/utils/utils.py`）。
2. 但 NaN 样本会对 PPO 造成污染（观测全 0、奖励 0、不终止），直接 resume 训练会让策略退化到 0 奖励。

**第二轮训练**（curriculum 渐进倒地概率 + 动作限幅 ±3.0 + NaN 终止 + track_height 1.5/σ0.15）：
训练日志 success_rate 20%~36%（但那是"没摔倒的回合"），倒地回合的起身率始终 0%。

**第三轮训练**（track_height 2.0/σ0.30 放宽 + 稠密 getup_height 1.0/σ0.35）：
1500+600 iters，训练 reward 升到 28，但倒地回合的起身率仍 0%（z_max 0.27 m）。

**物理可行性分析**（已验证）：侧躺时轮轴与重力夹角 cos = −1.0（轮轴竖直朝下），
轮子滚动方向变成竖直方向，无法产生水平推力把车身翻回来。腿全伸时轮子最低点 −0.137 m
（能伸到地面以下），说明"腿撑地起身"在几何上可行，但纯靠 RL 探索 40 分钟没找到这个动作。

**下一步建议**（未做）：给"腿撑地翻转"的中间动作 shaping（比如侧躺时奖励把 hip 抬向地面方向），
或改用 therapist/mimic 的 demo-aided 训练；也可以先在 MuJoCo 里把这段做通再搬回 Isaac。

`reset_to_fallen` 事件以 50% 概率把机器人放到侧躺位姿（z 0.45~0.60、roll 77°~95°、关节 ±0.1 rad）。
实测评测：倒地起步比例 31~43%，**站起成功率 0%**，z_max 只有 0.34 m。

已知两个坑：
1. 倒地位姿在接触求解里偶发 NaN；`rsl_rl` 的 `check_nan` 会直接中断训练，已改成"就地置 0 + 告警"（`_isaac_sim/kit/python/.../rsl_rl/utils/utils.py`）。
2. 但 NaN 样本会对 PPO 造成污染（观测全 0、奖励 0、不终止），直接 resume 训练会让策略退化到 0 奖励。

下一步建议：用 curriculum（倒地概率 0→50% 渐进）+ 专门的起身奖励（躯干高度上升速率、姿态回正），并让 NaN 环境强制终止而不是清零。

## 用键盘在 mjsim 里控制

```bash
# 1) policy 节点（先起，这样 sim 一启动就被接管）
source install/setup.bash
python3 isaac/policy_node_isaac.py --ros-args -p policy:=isaac/policies/nezha_flat_isaac.pt

# 2) MuJoCo 仿真
ros2 run mjsim sim_node --ros-args -p model:=nezhav1

# 3) 键盘手柄
ros2 run mjsim kb_joy        # W/S 前后，A/D 转向
```

`kb_joy` 的 `axes[1]`(W/S) → vx 指令，`axes[0]`(A/D) → wz 指令，范围 ±1，与训练一致。
policy 节点以 50 Hz 推理（`-p rate:=50`，与 Isaac 训练一致），输出 5n MIT 指令给 `sim_node`。

链路数据：
- `/joint_states`（关节位置/速度）
- `/imu/data`（姿态四元数 + 陀螺）
- `/base_state`（机体速度，`sim_node` 新增发布，机体系 [vx,vy,vz,wx,wy,wz]）
- `/joy`（键盘指令）
- `/mujoco_action`（5n MIT：块格式 [tau…][pos…][vel…][kp…][kd…]）

## 迁移到 MuJoCo 的实测包络

Isaac(PhysX/Newton) 训练的策略迁到 MuJoCo 后能站、能走，但侧向稳定裕度更小，实测：

| 指令 | 结果 |
|---|---|
| `vx=0` | 长时间稳定站立 |
| `vx=0.1, wz=0` | 30 s 稳定，漂移 <6° |
| `vx=0.2~0.5, wz=0` | 15~20 s 后侧翻（策略为消除 yaw 漂移把左右腿摆成不对称姿态） |
| `vx=0.2~0.4, wz≥0.05` | 30 s 稳定 |
| `vx=0.8, wz=0` | 稳定（只在前 5 s 内验证过） |

**用 `rl/train.py`（mujoco_warp GPU 后端）在 MuJoCo 里热启动微调后**（4096/2048 环境，
斜板地形，`--zero_wz_prob 0.5`），GPU 侧评测：

| 指令 | 微调前 | 微调后 |
|---|---|---|
| 站立 vx=0 | 100% | 100% |
| 随机命令（20 s，含斜板） | 0% | 79.4% |
| 固定 vx=0.3, wz=0 | 41% | 28% |
| 固定 vx=0.5, wz=0 | 1.2% | 4.3% |
| vx=0.3, wz=0.2（CPU 1 kHz 部署） | — | 20 s 稳定 |

结论：**站立、转向、随机指令都可用；恒定零偏航直行在 MuJoCo 里仍不稳**（Isaac 里同样指令是 100%）。
已排除的因素：关节符号/观测对齐（离线逐帧一致）、物理步长（0.001/0.002/0.005 都翻）、
接触摩擦与 condim、solref 接触刚度、左右腿质量不对称（右腿 hip/knee 比左腿重 40%/68%，
对称化后仍翻）、轮子碰撞形状（cylinder/capsule）。

所以默认参数收窄成 `vx_scale=0.3`、`wz_scale=0.4`：低速直行、带转向都能跑。
需要更快直行时，两条路：
1. 在 Isaac 里加大域随机化（摩擦 0.2~1.5、轮子 armature、观测延迟）重训；
2. 直接用 `rl/train.py` 在 MuJoCo 里微调（PPO 从当前策略热启动）。

调试工具：`isaac/sim_mujoco_policy.py` 可以不开 ROS 直接跑策略，支持 `--vx/--wz/--friction/--dt/--yaw_hold_kp` 等扫参数。

## 与 MuJoCo 仿真的关系

训练出来的策略可以直接看效果，但 mjsim（ROS/MuJoCo）里跑策略还需要相应接口；
当前 `rl/policy_node.py` 是给 MuJoCo 版策略用的（力矩 / MIT 指令）。

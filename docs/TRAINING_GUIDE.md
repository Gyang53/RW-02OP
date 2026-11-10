# nezhav1 轮足机器人 · 强化学习训练完整手册

> 面向新手，从零开始：装环境 → 训练 → 调优 → 部署到仿真和实机。
> 全程使用一台带 NVIDIA GPU（≥8 GB 显存）的 Ubuntu 22.04/24.04 电脑。

---

## 目录

1. [你的机器人在哪里](#1-你的机器人在哪里)
2. [环境安装（一次性）](#2-环境安装一次性)
3. [快速验证：能不能跑](#3-快速验证能不能跑)
4. [训练你的第一个策略](#4-训练你的第一个策略)
5. [看训练效果（Newton Viewer）](#5-看训练效果newton-viewer)
6. [Newton Viewer 键盘控制机器人](#6-newton-viewer-键盘控制机器人)
7. [在 MuJoCo 仿真里用键盘控制](#7-在-mujoco-仿真里用键盘控制)
8. [调优指南：策略不好怎么办](#8-调优指南策略不好怎么办)
9. [进阶：加新能力（蹲下/起身/单边板）](#9-进阶加新能力)
10. [常见问题排查](#10-常见问题排查)
11. [文件速查表](#11-文件速查表)

---

## 1. 你的机器人在哪里

这个仓库里有三套东西，搞清楚它们的角色就不会迷路：

```
┌─────────────────────────────────────────────────────────────┐
│  assets/nezha/          机器人模型（MJCF + USD + mesh）      │
│  ├── models/nezhav1/mjcf/mjmodel.xml     ← MuJoCo 模型      │
│  └── policies/                            ← 策略权重（空的）  │
│                                                             │
│  isaac/                 Isaac Lab 训练相关                   │
│  ├── make_clean_mjcf.py  清理 MJCF（去掉斜板等非机器人元素）│
│  ├── convert_mjcf.py     MJCF → USD（给 Isaac Sim 用）      │
│  ├── train.py            ⭐ 训练入口（在 IsaacLab 目录里跑）  │
│  ├── eval_policy.py      评测脚本                            │
│  ├── export_policy.py    导出 TorchScript                   │
│  ├── policy_node_isaac.py  ROS2 节点（mjsim 里跑策略）       │
│  ├── eval_getup.py       起身评测                            │
│  └── play_viewer.sh      Newton Viewer 看效果                │
│      play_viewer_kb.sh   Newton Viewer + 键盘控制（I/J/K/L）  │
│                                                             │
│  rl/                    MuJoCo 原生 PPO（备选，CPU/GPU）     │
│  ├── nezha_env.py        CPU 环境                            │
│  ├── nezha_env_gpu.py    GPU 环境（mujoco_warp）             │
│  ├── train.py            训练入口                            │
│  └── ppo.py              PPO 实现                            │
│                                                             │
│  src/mjsim/             ROS2 仿真节点（策略部署目标）        │
│  src/nezha_lqr_controller/  原厂 LQR 控制器（对比参考）      │
└─────────────────────────────────────────────────────────────┘
```

**训练流程一句话**：改 MJCF → 转成 USD → 用 Isaac Lab 训练 → 导出 TorchScript → 在 ROS2 仿真里跑。

---

## 2. 环境安装（一次性）

### 2.1 你需要的东西

| 组件 | 用来干嘛 | 在哪 |
|---|---|---|
| Isaac Sim 5.1 | 物理仿真引擎 | `/home/yangg/data/isaacsim/IsaacLab/_isaac_sim/` |
| Isaac Lab 3.0 | RL 训练框架 | `/home/yangg/data/isaacsim/IsaacLab/` |
| Isaac Lab .venv | Python 虚拟环境（torch/rsl_rl） | `/home/yangg/data/isaacsim/IsaacLab/.venv/` |
| ROS2 Jazzy | 仿真部署 | `/opt/ros/jazzy/` |
| mjsim 包 | MuJoCo 仿真节点 | 本仓库 `src/mjsim/`（已编译） |

> **已经有环境了？** 跳到第 3 节。以下只需要做一次。

### 2.2 编译 ROS2 工作空间

```bash
cd /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController
source /opt/ros/jazzy/setup.bash
colcon build --packages-select mjsim
source install/setup.bash
```

### 2.3 生成机器人 USD（MJCF → USD）

```bash
# 第一步：清理 MJCF（删掉斜板/参考体等非机器人元素）
python3 isaac/make_clean_mjcf.py
# 输出: assets/nezha/models/nezhav1/mjcf/nezha_isaac.xml

# 第二步：转 USD（需要 Isaac Sim 运行时，约 2 分钟）
cd /home/yangg/data/isaacsim/IsaacLab
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONEXE=$PWD/.venv/bin/python \
  ./_isaac_sim/python.sh \
  /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController/isaac/convert_mjcf.py \
  --visualizer none
```

**为什么需要两步？** 原始 MJCF 里有三样东西不能带进 Isaac：
- `slope`（斜板）：会被当成机器人的一部分，导致物理爆炸
- `base_Estimate`：可视化参考体（自由关节），会干扰物理
- `floor`：Isaac Lab 自己管地面

### 2.4 Isaac Lab 训练环境的关键设置

训练命令必须通过 `_isaac_sim/python.sh` 启动（不能用裸 python 或 .venv）：

```bash
# 模板（记住这一行，后面所有 Isaac 命令都用它）
cd /home/yangg/data/isaacsim/IsaacLab
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONEXE=$PWD/.venv/bin/python \
  ./_isaac_sim/python.sh <你的脚本或命令>
```

> **为什么这么麻烦？** Isaac Sim 是"下载版"运行时（`_isaac_sim/`），它自带一套
> Python 环境。`.venv` 里装了 torch/rsl_rl 等训练库。用 `PYTHONEXE` 把两者桥接：
> 运行时环境 + 训练库。`env -u CONDA_PREFIX` 是为了防止 conda 的库冲突。

---

## 3. 快速验证：能不能跑

### 3.1 环境冒烟测试

```bash
cd /home/yangg/data/isaacsim/IsaacLab
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONEXE=$PWD/.venv/bin/python \
  ./_isaac_sim/python.sh \
  /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController/isaac/smoke_env.py \
  --visualizer none --num_envs 4 --steps 50
```

**期望输出**：
```
[smoke] bodies (9): [base_link, abad_L_Link, ...]
[smoke] joints (8): [abad_L_Joint, hip_L_Joint, ...]
[smoke] action space: Box(-inf, inf, (4, 8), float32)
[smoke] obs space: Dict('policy': Box(-inf, inf, (4, 37), float32))
[smoke] step   0 base_z=0.880 rew=+0.015
[smoke] OK
```

看到 `OK` 就说明环境没问题。关注三个数字：
- **bodies (9)**：base_link + 4 条腿的 link
- **joints (8)**：abad/hip/knee/wheel × 左右
- **obs (4, 37)**：观测是 37 维（36 基础 + 1 高度指令）

### 3.2 冒烟不过？

| 报错 | 原因 | 解法 |
|---|---|---|
| `ImportError: No module named 'isaaclab'` | .venv 没激活 | 确认用了 `PYTHONEXE=.../.venv/bin/python` |
| `Failed to allocate on device 'cuda:0'` | 显存不足 | 关掉浏览器/IDE，或减 `--num_envs` |
| `cylinder-box collisions not implemented` | MJX 限制 | 用 mujoco_warp（本项目默认），不用 MJX |
| `nefc overflow` | 接触太多 | 在 `flat_env_cfg.py` 加大 `njmax`（当前 256） |

---

## 4. 训练你的第一个策略

### 4.1 基础训练（速度跟踪 + 高度 + 单边板）

```bash
cd /home/yangg/data/isaacsim/IsaacLab

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONEXE=$PWD/.venv/bin/python \
  ./_isaac_sim/python.sh \
  scripts/reinforcement_learning/train.py \
  --task Isaac-Velocity-Flat-Nezha \
  --rl_library rsl_rl \
  --num_envs 2048 \
  --visualizer none \
  --seed 1
```

**参数说明**：

| 参数 | 含义 | 建议值 |
|---|---|---|
| `--num_envs` | 并行环境数 | 2048（8GB 显存） / 4096（12GB+） |
| `--seed` | 随机种子 | 换种子=换一轮训练 |
| `--visualizer none` | 无界面训练 | 训练时不开窗口（省 GPU） |
| `--max_iterations` | 训练轮数 | 默认 1500，约 15 分钟 |

**训练输出**（每 50 iter 打印一次）：
```
Mean reward: 44.61
Episode_Reward/track_height: 0.6892
Metrics/base_velocity/error_vel_xy: 0.1036
Metrics/success_rate: 0.9792
Iteration time: 0.42s
Time elapsed: 0:02:43
ETA: 0:08:33
```

**怎么判断训练好了**：
| 指标 | 目标 | 含义 |
|---|---|---|
| `success_rate` | > 0.95 | 20 秒不摔的比例 |
| `error_vel_xy` | < 0.15 | 速度跟踪误差 (m/s) |
| `error_vel_yaw` | < 0.25 | 转向跟踪误差 (rad/s) |
| `track_height` | > 0.65 | 高度跟踪（1.0=完美） |
| `Mean reward` | > 40 | 总奖励（越高越好） |

**Checkpoint 保存在**：`IsaacLab/logs/rsl_rl/nezha_flat/<时间戳>/model_XXXX.pt`

### 4.2 训练耗时参考

| 显卡 | 环境数 | 吞吐 | 1500 iter 耗时 |
|---|---|---|---|
| RTX 5060 8GB | 2048 | ~48k steps/s | ~16 分钟 |
| RTX 4090 24GB | 4096 | ~100k+ steps/s | ~8 分钟 |
| CPU only | 64 | ~5k steps/s | ~2 小时（不推荐） |

---

## 5. 看训练效果（Newton Viewer）

### 5.1 快速启动（自动找最新 checkpoint）

```bash
cd /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController
./isaac/play_viewer.sh
```

### 5.2 指定 checkpoint 和环境数

```bash
./isaac/play_viewer.sh /home/yangg/data/isaacsim/IsaacLab/logs/rsl_rl/nezha_flat/<run>/model_1499.pt 16
```

第二个参数是并行环境数（16 个机器人同时走，看多样性）。

### 5.3 窗口内容

- 机器人 OpenGL 线框（每个环境一个实例）
- 左侧面板：`Pause Simulation` / `Reset Episode` 按钮
- 鼠标：左键旋转，中键平移，滚轮缩放

### 5.4 注意事项

| 事项 | 说明 |
|---|---|
| 渲染方式 | `--visualizer newton_gl`（轻量 OpenGL，8 GB 卡够用） |
| 完整 Isaac 界面 | `--visualizer kit`（RTX 渲染，8 GB 卡 OOM，不要用） |
| 窗口黑的 | 窗口还没完成初始化，等 2-3 分钟 |
| 窗口被遮挡 | `xdotool windowraise <window_id>` 提到最前 |
| checkpoint 路径 | `IsaacLab/logs/rsl_rl/nezha_flat/<run>/model_XXXX.pt` |
| 启动耗时 | 首次约 2-3 分钟（加载 USD + mesh） |

### 5.5 看什么

| 指标 | 判断 | 含义 |
|---|---|---|
| 走路姿态 | 线框平稳前进，轮子持续旋转 | 速度策略正常 |
| 转向 | A/D 按下后平滑转弯 | 转向策略正常 |
| 蹲下/站起 | Z 按住后机身下降（需要 policy_node 在线） | 高度策略正常 |
| 摔倒频率 | 线框侧翻不恢复 | 策略有问题，回到第 8 章调优 |

---

## 6. Newton Viewer 键盘控制机器人

在 Newton Viewer 里直接回放策略并用键盘驱动机器人走路——不用开 ROS2、不用三个终端，
一个命令就是一个带键盘控制的图形窗口。

### 6.1 启动

```bash
cd /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController

# 自动用最新的 checkpoint
./isaac/play_viewer_kb.sh

# 或指定 checkpoint 和环境数（推荐已验证的 2026-11-06_18-15-41/model_2998.pt）
./isaac/play_viewer_kb.sh \
    /home/yangg/data/isaacsim/IsaacLab/logs/rsl_rl/nezha_flat/2026-11-06_18-15-41/model_2998.pt 1
```

**第二个参数**是并行环境数。想看 16 个机器人一起走就填 16，键盘命令对所有环境生效。

> **checkpoint 选择很关键**：`ls -t` 找到的"最新"不一定是走得最好的。
> 训练中途切换任务（如加入起身训练）可能让速度策略退化。
> 挑 checkpoint 先用 `isaac/eval_policy.py --visualizer none` 量化验证，再拿来 Viewer 回放。
> 已验证可用的：`2026-11-06_18-15-41/model_2998.pt`（vx 误差 0.084 m/s，20 s 不摔）。

### 6.2 键盘操作

**查看器窗口必须在前台**（点一下窗口标题栏）。首次启动约 2-3 分钟（加载 USD + mesh），
终端出现 `[kb] viewer 已绑定: NewtonViewerGL` 说明键盘通道接好了。

| 按键 | 功能 | 说明 |
|---|---|---|
| `I`（按住） | 前进（1.0 m/s） | 释放即停 |
| `J`（按住） | 左转（−0.5 rad/s） | |
| `L`（按住） | 右转（+0.5 rad/s） | |
| `I` + `L` | 边走边右转 | 命令可叠加 |
| `K` | 停止（命令清零） | 不按任何键也是停止 |
| 关闭窗口 / Ctrl+C | 退出 | |

| 按键 | 实测表现（model_2998.pt） |
|---|---|
| 站立（不按键） | z≈0.72 m 稳定，漂移 ±0.01 m/s |
| `I` | vx 跟踪 0.97–1.04 m/s |
| `I`+`L` | vx≈1.0 m/s 同时转向 |

### 6.3 终端遥测

每 2 秒打印一行状态，不用盯窗口也能确认在动：

```
[kb] viewer 已绑定: NewtonViewerGL
[t=  2.0s] step=100 cmd=(0.00,+0.00) vx=+0.03 z=0.720 h_cmd=0.70
[t=  2.0s] step=16300 cmd=(1.00,+0.00) vx=+0.98 z=0.714 h_cmd=0.70
```

| 字段 | 含义 |
|---|---|
| `cmd=(vx, wz)` | 当前键盘指令 |
| `vx` | 机器人实际前进速度 (m/s)——与 cmd 前半接近 = 跟踪正常 |
| `z` | 机身高度 (m)——稳定在 0.70–0.73 = 没摔 |
| `h_cmd` | 高度指令（回放时固定 0.70，防止每 10 s 随机蹲起） |

### 6.4 修改速度/转向大小

编辑 `isaac/play_keyboard.py` 里 `KeyboardCommand.poll()`：

```python
if self._viewer.is_key_down(self._key_forward):
    fwd = 1.0        # 改这里：最大前进速度 (m/s)
if self._viewer.is_key_down(self._key_turn_left):
    turn -= 0.5      # 改这里：左转角速度 (rad/s)
```

### 6.5 常见问题

| 现象 | 原因 | 解法 |
|---|---|---|
| 按键没反应 | 窗口不在前台 | 点一下 Newton Viewer 窗口再按 |
| 按键没反应、也没绑定日志 | viewer 还没创建完 | 等出现 `[kb] viewer 已绑定`（最多 3 分钟） |
| 机器人抖动、不走 | checkpoint 不会走路（训练退化） | 换 `2026-11-06_18-15-41/model_2998.pt`，用 eval_policy 先验证 |
| 机器人每 10 s 蹲一下 | 高度指令被随机重采样 | 正常情况不会发生（脚本已固定 0.70）；自行改代码时注意 `height_cmd[:, 0] = 0.70` 这行 |
| 窗口黑的 | 首次初始化慢 | 等 2-3 分钟 |
| OOM | 显存不足 | 减小第二个参数（环境数），或关浏览器 |

### 6.6 与第 7 章（MuJoCo 键盘）的区别

| | 本章（play_viewer_kb.sh） | 第 7 章（policy_node + sim_node） |
|---|---|---|
| 物理引擎 | Newton（GPU） | MuJoCo（CPU） |
| 终端数 | 1 个 | 3 个 |
| 用途 | 训练效果快速回放 | 部署链路验证（与实机同构） |
| 键盘 | I/J/K/L | W/A/S/D + Z/X 蹲站 |
| 高度控制 | 固定 0.70 m | Z/X 可调 0.45–0.85 m |

---

## 7. 在 MuJoCo 仿真里用键盘控制

这是策略的实际部署环节：把训练好的策略加载到 ROS2 的 MuJoCo 仿真节点里。

### 7.1 导出策略

```bash
cd /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController

# 从 Isaac checkpoint 导出 TorchScript
python3 isaac/export_policy.py \
  --checkpoint /home/yangg/data/isaacsim/IsaacLab/logs/rsl_rl/nezha_flat/<run>/model_1499.pt \
  --out isaac/policies/nezha_height.pt
```

### 7.2 启动仿真 + 键盘

开三个终端（**启动顺序很重要：先起策略节点，再起仿真，最后起键盘**）：

```bash
# 终端 1：策略节点（先启动！）
cd /home/yangg/data/model_sim/RW-02OP/RW-02OP_OpenWBipedController
source install/setup.bash
python3 isaac/policy_node_isaac.py --ros-args \
  -p policy:=isaac/policies/nezha_height.pt

# 终端 2：MuJoCo 仿真
ros2 run mjsim sim_node --ros-args -p model:=nezhav1

# 终端 3：键盘
ros2 run mjsim kb_joy
```

**键盘操作**：

| 按键 | 功能 | 说明 |
|---|---|---|
| `W` / `S` | 前进 / 后退 | 按住持续生效 |
| `A` / `D` | 左转 / 右转 | |
| `Z`（按住） | 蹲下 | 机身降到 0.45 m |
| `X`（按住） | 站起 | 机身升到 0.85 m |
| `Space` | 急停 | |
| `Q` | 退出 kb_joy | 退出时的 RCLError 报错无害 |

> **启动顺序很重要**：先起策略节点（终端 1），再起仿真（终端 2）。
> 反过来的话机器人会先塌在地上几秒。
>
> **Z/X 不生效？** 高度指令的消费方是策略节点（终端 1），不是 sim_node。
> 如果策略节点没启动，Z/X 完全没效果。详见第 10 章排查。

### 7.3 调整速度和高度

```bash
# 策略节点启动参数
python3 isaac/policy_node_isaac.py --ros-args \
  -p policy:=isaac/policies/nezha_height.pt \
  -p vx_scale:=0.5 \
  -p wz_scale:=0.6 \
  -p height_min:=0.50 \
  -p height_max:=0.85 \
  -p height_rate:=0.2
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `vx_scale` | 0.3 | W/S 按下时的目标速度 (m/s) |
| `wz_scale` | 0.4 | A/D 按下时的转向速度 (rad/s) |
| `height_min` | 0.45 | Z 键蹲到的最低高度 (m) |
| `height_max` | 0.85 | X 键站到的最高高度 (m) |
| `height_rate` | 0.15 | 高度变化速率 (m/s) |

---

## 8. 调优指南：策略不好怎么办

### 8.1 训练不收敛（success_rate 上不去）

**先看是哪种不好**：

| 症状 | 可能原因 | 解法 |
|---|---|---|
| 一开始就不动 | 探索噪声太大 | 降低 `init_std`（在 `agents/rsl_rl_ppo_cfg.py`） |
| 前期能走后面倒 | 奖励权重不平衡 | 加大 `track_lin_vel` 或减小 `dof_torques` |
| 一直小幅抖动 | PD 增益太高 | 在 `nezha_cfg.py` 降 `stiffness` |
| 摔倒率高 | 终止太严格 | 放宽 `min_base_z` 或 `max_tilt_cos` |
| reward 剧烈震荡 | 学习率太大 | `--lr 5e-5` |

### 8.2 奖励权重速查

在 `nezha/flat_env_cfg.py` 里调（改完重训）：

```python
self.rewards.track_lin_vel_xy_exp.weight = 1.5    # 前进速度跟踪（主要目标）
self.rewards.track_ang_vel_z_exp.weight = 0.75     # 转向跟踪
self.rewards.track_height.weight = 1.5             # 高度跟踪
self.rewards.track_height.params["std"] = 0.45     # 高度容差（σ，越大越宽松）
self.rewards.flat_orientation_l2.weight = -2.5     # 机身水平惩罚（防侧翻）
self.rewards.dof_torques_l2.weight = -2e-4         # 力矩惩罚（越小动作越柔和）
self.rewards.action_rate_l2.weight = -0.01         # 动作变化率惩罚（防抖动）
self.rewards.lin_vel_z_l2.weight = -2.0            # 垂直速度惩罚（防弹跳）
```

**调参口诀**：
- 不前进 → 加大 `track_lin_vel`，减小 `torques`
- 抖动大 → 加大 `action_rate` 惩罚，减小 `dof_acc`
- 侧翻 → 加大 `flat_orientation` 惩罚
- 高度不跟 → 加大 `track_height`，减小 `std`

### 8.3 物理参数调优

在 `nezha_cfg.py`（机器人配置）里调：

```python
"legs": ImplicitActuatorCfg(
    joint_names_expr=[LEG_JOINTS],
    effort_limit_sim=120.0,     # 最大力矩 (N·m)
    velocity_limit_sim=30.0,    # 最大角速度 (rad/s)
    stiffness=40.0,             # PD 位置增益 kp
    damping=2.0,                # PD 速度增益 kd
),
"wheels": ImplicitActuatorCfg(
    joint_names_expr=[WHEEL_JOINTS],
    effort_limit_sim=40.0,
    velocity_limit_sim=50.0,
    stiffness=0.0,              # 轮子不做位置控制
    damping=2.0,                # 轮子速度增益
),
```

| 参数 | 效果 | 注意 |
|---|---|---|
| `stiffness` ↑ | 腿更硬，响应快 | 太高会震荡 |
| `damping` ↑ | 更稳定，但响应慢 | 轮子的 damping = 速度控制增益 |
| `effort_limit` ↑ | 力气大 | 太大会物理发散 |

### 8.4 训练技巧（从实践中总结的）

1. **热启动**：用已有 checkpoint 继续训练新能力，比从零快 3~5 倍
   ```bash
   --checkpoint logs/rsl_rl/nezha_flat/<旧run>/model_XXXX.pt
   ```

2. **降低学习率微调**：热启动时用 `--lr 1e-4`（默认 3e-4），避免破坏已有能力

3. **换种子重跑**：RL 有随机性，同一配置不同种子结果差很多。跑 3 个种子取最好的

4. **保存频率**：默认每 50 iter 存一次。想多留几个中间结果改 `save_interval`

5. **早停**：看到 `success_rate` 连续 200 iter 不涨就可以停了（Ctrl+C，checkpoint 已保存）

### 8.5 sim-to-deploy 的高度观测差异（已修复，记录坑）

**Isaac 训练时**：`height_commands` 观测 = `UniformHeightCommand.command`
= **绝对高度**（0.45~0.85 m，`mdp.generated_commands` 直接返回）

**部署节点旧版本**：`height_cmd_norm = (h - 0.65) / 0.25`（错误地做了归一化）
→ 策略收到的值在 [-0.8, +0.8]，而训练时是 [0.45, 0.85]，维度对但数值完全不同

**修复**：`isaac/policy_node_isaac.py` 已改为 `np.array([self.height])`（绝对高度）。
如果策略跑起来姿态不对、蹲不下去/站不起来，先检查这里的观测格式。

### 8.6 起身训练的特殊配置

起身是特殊任务，用 `Isaac-Velocity-Flat-Nezha-GetUp`（不是普通任务）：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONEXE=$PWD/.venv/bin/python \
  ./_isaac_sim/python.sh \
  scripts/reinforcement_learning/train.py \
  --task Isaac-Velocity-Flat-Nezha-GetUp \
  --rl_library rsl_rl \
  --num_envs 2048 \
  --visualizer none \
  --checkpoint logs/rsl_rl/nezha_flat/<速度策略run>/model_XXXX.pt   # 热启动
```

**起身任务的关键差异**：
- 50% 回合从倒地姿态开始（`reset_to_fallen` 事件）
- 倒地姿态：`abad=-0.32`（腿朝地面张开）+ `knee=0.663`（伸直）
  → 这个姿态让轮子穿入地面，给策略一个可用的支撑点
- 不检查 `base_contact`（倒地时 base_link 本来就贴地）
- 动作限幅 ±3.0（防止倒地时输出极端值导致数值发散）
- `track_height` 用 σ=0.45（起身过程中高度误差很大，需要宽梯度）

---

## 9. 进阶：加新能力

### 9.1 当前已有的能力

| 任务 | gym id | 能力 |
|---|---|---|
| 基础 | `Isaac-Velocity-Flat-Nezha` | 速度跟踪 + 高度 + 单边板 |
| 起身 | `Isaac-Velocity-Flat-Nezha-GetUp` | 从倒地站起 |

### 9.2 加新能力的步骤

以"走斜坡"为例：

1. **在 `flat_env_cfg.py` 里改场景**
   ```python
   # 加一个斜坡 box
   self.scene.ramp = AssetBaseCfg(
       prim_path="{ENV_REGEX_NS}/ramp",
       spawn=sim_utils.CuboidCfg(size=(1.0, 0.5, 0.05), ...),
       init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.03)),
   )
   ```

2. **在奖励里加对应项**（比如上坡时给额外奖励）
   ```python
   self.rewards.slope_bonus = RewTerm(
       func=my_custom_reward, weight=0.5, params={...}
   )
   ```

3. **注册新任务**（如果需要不同参数）
   ```python
   # 在 nezha/__init__.py 里
   gym.register(id="Isaac-Velocity-Flat-Nezha-Slope", ...)
   ```

4. **训练并评测**
   ```bash
   # 训练
   $RUN scripts/reinforcement_learning/train.py --task Isaac-Velocity-Flat-Nezha-Slope ...
   # 评测
   $RUN <ws>/isaac/eval_policy.py --task Isaac-Velocity-Flat-Nezha-Slope ...
   ```

### 9.3 加新的观测维度

观测顺序（37 维）在 `flat_env_cfg.py` 的 `ObservationsCfg` 里定义：

```python
class PolicyCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, ...)       # 0:3
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, ...)       # 3:6
    projected_gravity = ObsTerm(...)                          # 6:9
    velocity_commands = ObsTerm(...)                          # 9:12
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, ...)          # 12:20
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, ...)          # 20:28
    actions = ObsTerm(func=mdp.last_action)                   # 28:36
    height_command = ObsTerm(func=mdp.generated_commands, ...) # 36:37
```

**加新维度时注意**：要同步修改 `rl/nezha_env.py` 的 `OBS_DIM` 和
`isaac/policy_node_isaac.py` 的 `OBS_DIM`，否则部署时维度不匹配。

### 9.4 部署到实机

策略导出后的 TorchScript 可以直接在 ROS2 节点里跑。部署到实机需要：
1. 把 `isaac/policy_node_isaac.py` 的话题从仿真改为实机驱动
2. 输入：实机 IMU + 关节编码器
3. 输出：实机电机指令
4. 注意 sim-to-real gap：训练时的 PD 增益/摩擦系数要和实机标定一致

---

## 10. 常见问题排查

### Q: 训练时 GPU OOM
```
RuntimeError: Failed to allocate XXX bytes on device 'cuda:0'
```
**解法**：减 `--num_envs`（2048 → 1024 → 512），或关掉浏览器/IDE。

### Q: 训练时出现 NaN，直接崩了
```
ValueError: The observation group 'policy' contains NaN values
```
**解法**：这个项目已经 patch 了 rsl_rl（NaN 就地清零 + 告警）。
如果又出现，说明物理发散太严重，检查：
- `reset_to_fallen` 的 z_range 是否太低（穿透地面）
- `effort_limit_sim` 是否太大
- `njmax`/`nconmax` 是否够

### Q: `--visualizer kit` 启动后进程被杀
**原因**：Isaac Sim RTX 渲染需要 >4 GB 显存，8 GB 卡不够。
**解法**：用 `--visualizer newton_gl`（轻量 OpenGL）。

### Q: kb_joy 报 `RCLError: failed to shutdown`，Z/X 没反应
这是 kb_joy 退出时的已知无害报错（Q 退出时 rclpy 被关了两次），**不影响 Z/X 功能**。

**Z/X 真正没反应的原因**：kb_joy 发布 `joy.axes[2]`，但消费它的是
`isaac/policy_node_isaac.py`（不是 sim_node）。如果策略节点没启动，
或者节点版本旧（height 观测格式不匹配），Z/X 就完全没效果。

**排查步骤**：
```bash
# 1. 确认 kb_joy 正常发 joy
ros2 topic echo /joy    # 按 Z/X，期望 axes[2] = -1.0 / +1.0
# 2. 确认策略节点收到 joy
ros2 topic info /joy    # 期望 Publisher count >= 2（kb_joy + policy_node）
# 3. 确认 /mujoco_action 有数据
ros2 topic hz /mujoco_action    # 期望 >= 50 Hz
```
如果第 1 步 axes[2] 没变化：kb_joy 版本旧，`colcon build --packages-select mjsim` 重编译。
如果第 3 步没数据：策略节点崩了，看它的终端报错。

### Q: `play_viewer.sh` 窗口打开但是黑的
**原因**：窗口还没完成初始化，或被其他窗口遮挡。
**解法**：等 2-3 分钟，用 `xdotool windowraise` 把窗口提到最前。

### Q: `play_viewer_kb.sh` 按键没反应
**排查顺序**：
1. Newton Viewer 窗口是否在前台？点一下窗口再按 `I`。
2. 终端有没有 `[kb] viewer 已绑定: NewtonViewerGL`？没有就还在初始化，等最多 3 分钟。
3. 遥测里 `cmd=(1.00,+0.00)` 变了吗？
   - 变了但 `vx` 不动 → **checkpoint 问题**（策略不会走），换
     `2026-11-06_18-15-41/model_2998.pt`，或先用 `eval_policy.py` 量化验证。
   - 没变 → 键盘轮询没接上，重跑并确认前台。

### Q: `ros2 run mjsim` 找不到包
**解法**：`source install/setup.bash`（每次开新终端都要）。

### Q: policy_node 报 `OBS_DIM mismatch`
**原因**：导出的策略是 36 维（旧），但节点期望 37 维。
**解法**：用最新的 `isaac/export_policy.py` 重新导出。

### Q: MuJoCo 里机器人侧翻后站不起来
**原因**：侧翻时轮轴垂直于地面（与重力夹角 cos=-1.0），轮子无法产生水平力。
**解法**：这是物理结构限制。起身训练用 `Isaac-Velocity-Flat-Nezha-GetUp`
（从倒地姿态开始），或者加"翻转"专用动作。

---

## 11. 文件速查表

| 路径 | 说明 |
|---|---|
| **训练** | |
| `isaac/make_clean_mjcf.py` | 清理 MJCF |
| `isaac/convert_mjcf.py` | MJCF→USD |
| `isaac/smoke_env.py` | 环境冒烟测试 |
| `isaac/eval_policy.py` | 速度/高度评测 |
| `isaac/eval_getup.py` | 起身评测 |
| `isaac/export_policy.py` | 导出 TorchScript |
| `isaac/play_viewer.sh` | Newton Viewer 看效果 |
| `isaac/play_viewer_kb.sh` | Newton Viewer + 键盘控制（I/J/K/L） |
| `isaac/policy_node_isaac.py` | ROS2 部署节点 |
| `isaac/sim_mujoco_policy.py` | MuJoCo 离线策略测试（无 ROS） |
| **Isaac Lab 训练配置**（完整前缀 `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/`） | |
| `.../nezha/nezha_cfg.py` | 机器人（PD 增益/限位） |
| `.../nezha/flat_env_cfg.py` | 环境（奖励/观测/终止/地形） |
| `.../nezha/getup_env_cfg.py` | 起身任务配置 |
| `.../nezha/mdp/commands.py` | 高度指令 |
| `.../nezha/mdp/events.py` | 倒地复位/斜板隐藏 |
| `.../nezha/mdp/rewards.py` | 自定义奖励函数 |
| `.../nezha/mdp/terminations.py` | NaN 终止 |
| `.../nezha/mdp/curriculums.py` | 课程学习 |
| `.../nezha/agents/rsl_rl_ppo_cfg.py` | PPO 超参 |
| **MuJoCo 备选** | |
| `rl/nezha_env.py` | CPU 环境 |
| `rl/nezha_env_gpu.py` | GPU 环境（mujoco_warp） |
| `rl/train.py` | MuJoCo PPO 训练 |
| `rl/ppo.py` | PPO 实现 |
| **机器人模型** | |
| `assets/.../mjmodel.xml` | MuJoCo MJCF |
| `assets/.../nezha_isaac.usda` | Isaac USD |

---

## 附录 A：完整命令速查

```bash
# ===== 一次性环境准备 =====
python3 isaac/make_clean_mjcf.py                          # 清理 MJCF
cd IsaacLab && $RUN <ws>/isaac/convert_mjcf.py --visualizer none  # 转 USD

# ===== 训练 =====
cd IsaacLab
$RUN scripts/reinforcement_learning/train.py \
    --task Isaac-Velocity-Flat-Nezha --rl_library rsl_rl \
    --num_envs 2048 --visualizer none --seed 1

# 起身任务
$RUN scripts/reinforcement_learning/train.py \
    --task Isaac-Velocity-Flat-Nezha-GetUp --rl_library rsl_rl \
    --num_envs 2048 --visualizer none \
    --checkpoint logs/rsl_rl/nezha_flat/<run>/model_XXXX.pt

# ===== 评测 =====
$RUN <ws>/isaac/eval_policy.py --visualizer none \
    --num_envs 128 --steps 1000 --checkpoint <ckpt>
$RUN <ws>/isaac/eval_getup.py --visualizer none \
    --num_envs 128 --steps 600 --fall_prob 0.6 --checkpoint <ckpt>

# ===== 可视化 =====
./isaac/play_viewer.sh                                    # Newton Viewer
./isaac/play_viewer_kb.sh <ckpt> 1                        # Newton Viewer + 键盘（I/J/K/L）

# ===== 导出部署 =====
python3 isaac/export_policy.py --checkpoint <ckpt> --out isaac/policies/xxx.pt

# ===== ROS2 部署 =====
source install/setup.bash
python3 isaac/policy_node_isaac.py --ros-args -p policy:=isaac/policies/xxx.pt
ros2 run mjsim sim_node --ros-args -p model:=nezhav1
ros2 run mjsim kb_joy
```

## 附录 B：已实测的性能基线

| 能力 | 指标 | 数值 |
|---|---|---|
| 速度跟踪 | vx 误差 | 0.05 m/s |
| 转向跟踪 | wz 误差 | 0.08 rad/s |
| 高度跟踪 | z 误差 | 0.019 m |
| 摔倒率（20s） | base_contact | 0% |
| 起身成功率 | 12.5 s 内 | 72.9% |
| 单边板 | success_rate | 91.3% |
| 训练速度 | GPU 吞吐 | 48k env-step/s |

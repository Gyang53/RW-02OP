# 在 Newton Viewer 里回放 nezhav1 训练好的策略，键盘全功能控制机器人。
#
# cd /home/yangg/data/isaacsim/IsaacLab
# env -u CONDA_PREFIX PYTHONEXE=./.venv/bin/python ./_isaac_sim/python.sh \
#     <ws>/isaac/play_keyboard.py --checkpoint <ckpt> --num_envs 1 --visualizer newton_gl
#
# 键盘（查看器窗口必须在前台，可叠加）:
#   W / S   — 前进 +1.0 m/s / 后退 −1.0 m/s
#   A / D   — 左转 / 右转 (±0.5 rad/s)
#   Z / X   — 蹲下 0.45 m / 站高 0.85 m（按住到目标，松开保持）
#   C       — 回到中间高度 0.70 m
#   R       — 手动复位全部环境（配合起身策略演示"倒下→站起"）
#   释放行走键自动停止

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", required=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


# 键盘命令 → 观测写入
class KeyboardCommand:
    """全功能键盘 → base_velocity + height 命令覆盖（轮询 viewer.is_key_down）.

    register_key_press 回调会被 Newton viewer 的 GUI/事件链吞掉，不可靠；
    RendererGL 内部 push 了 KeyStateHandler，is_key_down 轮询是稳定路径。

    按键（可叠加）：
      W 前进 +1.0 m/s   S 后退 −1.0 m/s   A 左转 −0.5 rad/s   D 右转 +0.5 rad/s
      Z 蹲下 h=0.45     X 站高 h=0.85     C 中间高度 h=0.70（不按 Z/X/C 时高度保持不变）
      R 强制复位所有环境（模拟"倒下后重新开始"，配合起身策略演示 getup）
    """

    def __init__(self, num_envs: int, device: str, viewer=None):
        import pyglet

        self.commands = torch.zeros(num_envs, 3, device=device)
        self.height = torch.full((num_envs, 1), 0.70, device=device)
        self.reset_requested = False
        self.device = device
        self._viewer = viewer
        self._key_forward = pyglet.window.key.W
        self._key_backward = pyglet.window.key.S
        self._key_turn_left = pyglet.window.key.A
        self._key_turn_right = pyglet.window.key.D
        self._key_crouch = pyglet.window.key.Z
        self._key_stand = pyglet.window.key.X
        self._key_mid = pyglet.window.key.C
        self._key_reset = pyglet.window.key.R

    def set_viewer(self, viewer):
        """viewer 是 NewtonViewerGL，有 is_key_down(symbol)->bool。"""
        self._viewer = viewer

    def poll(self):
        """每帧调用：轮询按键状态写入命令。前进+转向、行走+蹲站均可叠加。"""
        self.commands[:] = 0.0
        self.reset_requested = False
        if self._viewer is None:
            return
        fwd, turn = 0.0, 0.0
        if self._viewer.is_key_down(self._key_forward):
            fwd += 1.0
        if self._viewer.is_key_down(self._key_backward):
            fwd -= 1.0
        if self._viewer.is_key_down(self._key_turn_left):
            turn -= 0.5
        if self._viewer.is_key_down(self._key_turn_right):
            turn += 0.5
        self.commands[:, 0] = fwd
        self.commands[:, 2] = turn
        # 高度：按住持续生效，松开保持当前值（rate 限制在主循环里做）
        if self._viewer.is_key_down(self._key_crouch):
            self.height[:, 0] = 0.45
        elif self._viewer.is_key_down(self._key_stand):
            self.height[:, 0] = 0.85
        elif self._viewer.is_key_down(self._key_mid):
            self.height[:, 0] = 0.70
        self.reset_requested = self._viewer.is_key_down(self._key_reset)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # 关闭 heading command（heading 模式用相对偏航角速度，不是绝对速度）
    env_cfg.commands.base_velocity.heading_command = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    _reset = env.reset()
    obs = _reset[0]
    if isinstance(obs, dict):
        obs = obs["policy"]

    # load checkpoint (MlpPolicy)
    ck = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    obs_dim = int(ck["mlp.0.weight"].shape[1])
    mean = ck["obs_normalizer._mean"].numpy().astype(np.float64).reshape(-1)
    var = ck["obs_normalizer._var"].numpy().astype(np.float64)
    w = [ck[f"mlp.{i}.weight"].numpy().T for i in (0, 2, 4, 6)]
    b = [ck[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]

    def policy(obs):
        x = (obs - mean) / np.sqrt(var + 1e-8)
        for i in range(3):
            x = x @ w[i] + b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x @ w[3] + b[3]

    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")
    height_cmd = env.unwrapped.command_manager.get_command("height")

    # 获取 renderer：sim_ctx._visualizers 里每个是 NewtonGLVisualizer
    # 它的 _viewer.renderer 有 register_key_press
    # visualizer 是懒创建的（soft_reset 时），放到 while 里第一帧查找
    sim_ctx = env.unwrapped.sim.simulation_context if hasattr(env.unwrapped.sim, "simulation_context") else None
    if sim_ctx is None:
        from isaaclab.sim import SimulationContext

        sim_ctx = SimulationContext.instance()

    kb = KeyboardCommand(args_cli.num_envs, env.unwrapped.device)
    dt = env.unwrapped.step_dt
    viewer_found = False
    while simulation_app.is_running():
        with torch.inference_mode():
            # 每帧查找 viewer（懒创建），找到后用 is_key_down 轮询
            if not viewer_found and sim_ctx is not None:
                if not sim_ctx._visualizers:
                    sim_ctx.initialize_visualizers()
                for viz in getattr(sim_ctx, "_visualizers", []):
                    v = getattr(viz, "_viewer", None)
                    if v is not None and hasattr(v, "is_key_down"):
                        kb.set_viewer(v)
                        viewer_found = True
                        print(f"[kb] viewer 已绑定: {type(v).__name__}")
                        break
            kb.poll()
            # 键盘命令写入 base_velocity
            command[:, 0] = kb.commands[:, 0]
            command[:, 2] = kb.commands[:, 2]
            # 高度：键盘 Z/X/C 直接设目标；无按键时以 height_rate m/s 向目标缓动，
            # 避免 0.4 m 阶跃把策略打出分布外。同时阻止 10 s 重采样的随机蹲起。
            _rate = 0.25  # m/s
            _cur = height_cmd[:, 0]
            _tgt = kb.height[:, 0]
            height_cmd[:, 0] = _cur + torch.clamp(_tgt - _cur, -_rate * dt, _rate * dt)
            # R：手动复位全部环境（演示"倒下→起身"循环）
            if kb.reset_requested and not main._reset_latch:
                env.unwrapped.reset()
                obs = env.unwrapped.observation_manager.compute()
                if isinstance(obs, dict):
                    obs = obs["policy"]
                main._reset_latch = True
                print("[kb] 手动复位全部环境", flush=True)
            elif not kb.reset_requested:
                main._reset_latch = False
            if not hasattr(main, "_t"):
                main._t = 0.0
                main._n = 0
                main._reset_latch = False
            main._t += dt
            main._n += 1
            if main._t >= 2.0:
                v = robot.data.root_lin_vel_b[0]
                fallen = int((robot.data.root_pos_w[:, 2] < 0.35).sum().item())
                print(
                    f"[t={main._t:5.1f}s] step={main._n} cmd=({command[0, 0]:+.2f},{command[0, 2]:+.2f}) "
                    f"vx={v[0]:+.2f} z={robot.data.root_pos_w[0, 2]:.3f} "
                    f"h_cmd={height_cmd[0, 0]:.2f} fallen={fallen}/{kb.commands.shape[0]}",
                    flush=True,
                )
                main._t = 0.0
            if isinstance(obs, dict):
                obs = obs["policy"]
            obs_np = obs.detach().cpu().numpy()
            actions = torch.as_tensor(policy(obs_np), dtype=torch.float32, device=env.unwrapped.device)
            result = env.step(actions)
            obs = result[0]
            if isinstance(obs, dict):
                obs = obs["policy"]


if __name__ == "__main__":
    main()
    simulation_app.close()

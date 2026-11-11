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
#   --jump  — 障碍临近自动蹲跳越障（需要含障碍物的场景）
#   --launch — G 键模拟发射机构（软件给竖直初速，用于演示 0.6 m 跳跃与落地）
#   释放行走键自动停止

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Nezha")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--h_crouch", type=float, default=0.45, help="Z 键蹲下目标高度 (m)")
parser.add_argument("--h_stand", type=float, default=0.85, help="X 键站高目标高度 (m)")
parser.add_argument("--h_mid", type=float, default=0.70, help="C 键中间高度 (m)")
parser.add_argument("--jump", action="store_true", help="障碍临近时自动执行蹲跳原语")
parser.add_argument("--jump_dist", type=float, default=0.6, help="触发跳跃的障碍距离 (m)")
parser.add_argument("--launch", action="store_true",
                    help="启用 G 键发射（软件模拟发射机构：直接给竖直初速，用于机械未就位时演示 0.6 m 跳跃）")
parser.add_argument("--launch_vz", type=float, default=3.43,
                    help="发射初速 m/s（3.43 ≈ 0.6 m 跳跃高度）")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from jump_primitive import JumpPrimitive  # noqa: E402


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

    def __init__(self, num_envs: int, device: str, viewer=None,
                 h_crouch: float = 0.45, h_stand: float = 0.85, h_mid: float = 0.70):
        import pyglet

        self.commands = torch.zeros(num_envs, 3, device=device)
        self.height = torch.full((num_envs, 1), h_mid, device=device)
        self._h_crouch = h_crouch
        self._h_stand = h_stand
        self._h_mid = h_mid
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
        self._key_launch = pyglet.window.key.G

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
            self.height[:, 0] = self._h_crouch
        elif self._viewer.is_key_down(self._key_stand):
            self.height[:, 0] = self._h_stand
        elif self._viewer.is_key_down(self._key_mid):
            self.height[:, 0] = self._h_mid
        self.reset_requested = self._viewer.is_key_down(self._key_reset)
        self.launch_requested = self._viewer.is_key_down(getattr(self, "_key_launch", None)) \
            if getattr(self, "_key_launch", None) else False


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

    # 跳跃原语（可选）：需要场景里有 obstacle
    jump_prim = None
    obs_pos = None
    if args_cli.jump:
        if "obstacle" in env.unwrapped.scene.keys():
            jump_prim = JumpPrimitive(args_cli.num_envs, env.unwrapped.device)
            obs_pos = env.unwrapped.scene["obstacle"].data.root_pos_w
            print("[jump] 跳跃原语已启用，触发距离 "
                  f"{args_cli.jump_dist:.2f} m（蹲 0.30s → 饱和弹射 0.16s → 收腿 0.22s）")
        else:
            print("[jump] 场景没有 obstacle，跳跃原语不可用（需要 TASK=...-S06-V4）")

    kb = KeyboardCommand(
        args_cli.num_envs,
        env.unwrapped.device,
        h_crouch=args_cli.h_crouch,
        h_stand=args_cli.h_stand,
        h_mid=args_cli.h_mid,
    )
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
                jump_txt = ""
                if jump_prim is not None:
                    jump_txt = f" jump={jump_prim.phase_name()[0]}"
                    if obs_pos is not None:
                        jump_txt += f" obs_dx={(obs_pos[0, 0] - robot.data.root_pos_w[0, 0]).item():+.2f}"
                print(
                    f"[t={main._t:5.1f}s] step={main._n} cmd=({command[0, 0]:+.2f},{command[0, 2]:+.2f}) "
                    f"vx={v[0]:+.2f} z={robot.data.root_pos_w[0, 2]:.3f} "
                    f"h_cmd={height_cmd[0, 0]:.2f} fallen={fallen}/{kb.commands.shape[0]}{jump_txt}",
                    flush=True,
                )
                main._t = 0.0
            if isinstance(obs, dict):
                obs = obs["policy"]
            obs_np = obs.detach().cpu().numpy()
            actions = torch.as_tensor(policy(obs_np), dtype=torch.float32, device=env.unwrapped.device)
            # 障碍临近 → 用跳跃原语覆盖动作（蹲下 → 饱和力矩弹射 → 收腿）
            if jump_prim is not None:
                dx = (obs_pos[:, 0] - robot.data.root_pos_w[:, 0]).abs()
                jump_prim.trigger(dx < args_cli.jump_dist)
                jump_prim.advance(dt)
                if jump_prim.active.any():
                    actions = torch.where(jump_prim.active.unsqueeze(-1), jump_prim.actions(), actions)
            # G 键：模拟发射机构（直接给竖直初速），验证 0.6 m 跳跃 + 落地
            if args_cli.launch and kb.launch_requested and not getattr(main, "_launch_latch", False):
                vel = robot.data.root_lin_vel_w.clone()
                vel6 = torch.zeros((args_cli.num_envs, 6), device=env.unwrapped.device)
                vel6[:, 0] = vel[:, 0]
                vel6[:, 2] = args_cli.launch_vz
                robot.write_root_velocity_to_sim(vel6)
                main._launch_latch = True
                print(f"[launch] 发射！vz={args_cli.launch_vz:.2f} m/s "
                      f"(≈{args_cli.launch_vz**2/(2*9.81):.2f} m 跳跃高度)", flush=True)
            elif not (args_cli.launch and kb.launch_requested):
                main._launch_latch = False
            result = env.step(actions)
            obs = result[0]
            if isinstance(obs, dict):
                obs = obs["policy"]


if __name__ == "__main__":
    main()
    simulation_app.close()

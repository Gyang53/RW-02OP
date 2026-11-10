

## 说明
需要最新的CMotorCommand模块



## 注意

**MJCF模型文件**
- USD文件所有的motor关节使用正向进行训练
- MJCF模型文件电机方向设置为与实际机器人相同
- 


控制器
- 

### 仿真运行
```shell
# 编译
colcon build --packages-select mjsim
# 软链接编译
colcon build --symlink-install --packages-select mjsim

# 加载工作空间（必须先 source，否则 ros2 找不到 mjsim）
source install/setup.bash

# 运行sim_node（当前仓库注册的模型名只有 nezhav1，没有 nezhav1v1）
ros2 run mjsim sim_node --ros-args -p model:=nezhav1
# 0.2X倍速运行仿真
ros2 run mjsim sim_node --ros-args -p model:=nezhav1 -p sim_rate:=0.2

# 运行控制器（C++ LQR 控制器，mjsim 里没有 nezhav1_controller2 这个入口）
ros2 run nezha_lqr_controller wblqr_controller
# 手柄（LB+A 进入站立）
ros2 run joy joy_node
# 无手柄时可用键盘 Joy 发布器替代
ros2 run mjsim kb_joy
```

**已知现象（实测）**
- 只跑 `sim_node` 时没有任何保持力矩（`cfgs.py` 里 nezhav1 的 `default_kp`/`default_kd` 全为 0），
  模型会从 0.88 m 蹲到地面，看起来像“没仿真效果”，其实仿真在正常跑（窗口标题 `MuJoCo : nezha_v1`，约 0.97 倍实时）。
- `wblqr_controller` 启动后处于 `REST`（tau=0，轮子也没有力矩），按 `LB+A` 切 `STANDING`，
  3 s 后自动切 `RUNNING` 才用轮子做平衡；机器人一旦倒下不会自己站起来，需要重启 `sim_node`。

**可能想修改**
- 相机跟踪距离：sim_node, 搜索`enable_camera_tracking`，修改跟踪主体与相机距离
- 

### policy测试

```shell
# 数据测试(data.json)
python3 src/mjsim/test/test_policy.py --frames 60 --frame 5
# 默认obs，切换action计算(前10帧参考意义不大)
python3 src/mjsim/test/test_policy.py --frames 60 --test action
```

### 工具
```shell
# 模型基础信息
python3 src/mjsim/test/test_model.py nezhav1

# 可视化
ros2 run mjsim model_viewer --ros-args -p model:=nezhav1
# Web可视化（viser，WebGL，可以在手机上查看）
ros2 run mjsim viser_model_viewer --ros-args -p model:=nezhav1 -p port:=8080

# 计算器
tools.ipynb
# 键盘Joy指令发布器
ros2 run mjsim kb_joy

# Web状态监测器
ros2 run mjsim web_monitor
```

**数据记录与分析**

```shell
# 记录数据
ros2 run mjsim log_node --ros-args -p model_type:=nezhav1 -p log_rate:=100
# 数据分析
cd /home/shiro/wBipedController/logs/nezhav1
python3 plot_logs.py 

```







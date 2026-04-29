# lidar_body_calib_ceres_ros2

ROS2 Humble + Ceres 的多 Mid-360 雷达到车体坐标系外参标定节点。

## 功能

- 读取多个 Mid-360 的 YAML 配置
- 每个雷达可绑定独立点云 topic
- 累计约 2 秒点云
- 依据**车体系理想平面先验**和**初始外参**，先把理想平面变换到雷达系，做 ROI 点筛选
- 用 PCL RANSAC 拟合平面
- 用 Ceres 做“平面小偏差 + 鲁棒损失”的非线性联合优化
- 输出优化后的 `rpy / t / R`
- 可选发布 TF 和可视化 Marker

## 安装依赖

```bash
sudo apt update
sudo apt install -y \
  ros-humble-pcl-conversions \
  ros-humble-tf2-ros \
  ros-humble-geometry-msgs \
  ros-humble-visualization-msgs \
  libpcl-dev libyaml-cpp-dev libceres-dev
```

## 编译

```bash
conda deactivate
cd ~/ros2_ws/src
cp -r /path/to/lidar_body_calib_ceres_ros2 .
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select lidar_body_calib_ceres_ros2
source install/setup.bash
```
## 如果删除
```bash
rm -rf build/lidar_body_calib_ceres_ros2 install/lidar_body_calib_ceres_ros2 log
```
## 运行

```bash
ros2 launch lidar_body_calib_ceres_ros2 calibrate.launch.py
```

或直接运行：
无点云文件保存
```bash
ros2 run lidar_body_calib_ceres_ros2 lidar_body_calib_ceres_ros2 --ros-args \
  -p config_path:=/home/wangjunfeng/ros2_ws/src/lidar_body_calib_ceres_ros2/config/lidar_config.yaml \
  -p target_lidar_ip:=192.168.1.135 \
  -p accumulation_time_sec:=2.0 \
  -p roi_distance_threshold:=0.08 \
  -p ransac_distance_threshold:=0.03 \
  -p min_plane_points:=300 \
  -p voxel_leaf_size:=0.01 \
  -p publish_tf:=true \
  -p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result.yaml
```

保存点云文件

```bash
conda deactivate
rm -rf build/lidar_body_calib_ceres_ros2 install/lidar_body_calib_ceres_ros2 log
colcon build --packages-select lidar_body_calib_ceres_ros2
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/humble/lib
source install/setup.bash 
```

```bash
ros2 run lidar_body_calib_ceres_ros2 lidar_body_calib_ceres_ros2 \
  --ros-args \
  -p config_path:=/home/wangjunfeng/ros2_ws/src/lidar_body_calib_ceres_ros2/config/lidar_config.yaml \
  -p target_lidar_ip:=192.168.1.135 \
  -p accumulation_time_sec:=3.0 \
  -p roi_distance_threshold:=0.02 \
  -p ransac_distance_threshold:=0.01 \
  -p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result.yaml \
  -p output_cloud_dir:=/home/wangjunfeng/ros2_ws/output/mid360_calib_outputs
```
## 配置文件说明

配置中 `planes` 是**车体系**下的理想平面，采用标准式：

`Ax + By + Cz + D = 0`

建议统一使用**米**。例如 `z = 0.80` 应写成：

```yaml
- [0.0, 0.0, -1.0, 0.80]
```

如果你仍然使用 `- [0, 0, -800, 1]` 这种形式，数值比例不一致，会导致平面含义失真。


# 26-04-22 版本

这版的能力：

- 订阅 sensor_msgs/msg/PointCloud2
- 自动从 YAML 的对应 lidar_ip 节点读取 topic，也支持用参数 input_topic 覆盖
- 从首帧开始累计 accumulation_time_sec
- 转成 pcl::PointCloud<pcl::PointXYZ>
- 调用现有流程：
  - loadYamlInput()
  - extractPlaneMeasurementsWithPadding()
  - solveCeres()
  - printResult()
- 导出累计原始点云到 output_cloud_dir
- 导出解算结果到 output_result_path
- 退出码与解算结果绑定：
  - 0 严格通过
  - 2 可用但未严格通过
  - 3 解算失败
  - 1 运行时错误

这个版本额外依赖：

- sensor_msgs
- pcl_conversions
- pcl_ros 或至少 PCL + conversions
- yaml-cpp
- ceres
- Eigen3

运行命令可以继续沿用，建议加上 topic 覆盖更稳一些，例如：

```bash
conda deactivate
rm -rf build/lidar_body_calib_ceres_ros2 install/lidar_body_calib_ceres_ros2 log
colcon build --packages-select lidar_body_calib_ceres_ros2
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/humble/lib
source install/setup.bash 
```

```bash
ros2 run lidar_body_calib_ceres_ros2 lidar_body_calib_ceres_ros2 \
--ros-args \
-p config_path:=/home/wangjunfeng/ros2_ws/src/lidar_body_calib_ceres_ros2/config/lidar_config.yaml \
-p target_lidar_ip:=192.168.1.135 \
-p input_topic:=/livox/lidar_192_168_1_135 \
-p accumulation_time_sec:=3.0 \
-p roi_distance_threshold:=0.02 \
-p ransac_distance_threshold:=0.01 \
-p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result.yaml \
-p output_cloud_dir:=/home/wangjunfeng/ros2_ws/output/mid360_calib_outputs \
```


# 平地 8 字轨迹动态法
**这版主要增加了这些能力：**

- 保留原来的静态三平面批量求解作为主标定。
- 在 finalizeBatch() 后继续执行可选的 8 字轨迹动态验证。
- 支持从 figure8_data_path 指向的目录读取：
  - body_trajectory.csv
  - lidar_trajectory.csv
- 对两条轨迹做时间偏移扫描，自动找更优 time_offset_sec。
- 基于相对运动构造 hand-eye 一致性约束，评估当前静态外参是否满足动态轨迹关系。
- 可选做小范围微调，只允许很小的 rpy/t 修正，符合 PDF 里“动态复核/小幅微调”的定位。
- 在输出 YAML 中增加更完整的 figure8_verification 结果，包括：
  - matched_pose_pairs
  - motion_pairs
  - estimated_time_offset_sec
  - rotation_consistency_deg
  - translation_consistency_m
  - refined_result

**这版新增的 ROS2 参数包括：**

- enable_figure8_verification
- figure8_data_path
- figure8_enable_micro_refine
- figure8_replace_final_result_on_success
- figure8_max_rotation_error_deg
- figure8_max_translation_error_m
- figure8_max_time_offset_sec
- figure8_time_offset_step_sec
- figure8_association_tolerance_sec
- figure8_min_relative_translation_m
- figure8_min_relative_rotation_deg
- figure8_min_motion_pairs
- figure8_max_num_iterations

这版动态法的输入 CSV 约定为：

```csv
timestamp_sec,tx,ty,tz,qx,qy,qz,qw
0.000, ...
0.050, ...
...
```


其中：

- body_trajectory.csv 表示车体/底盘轨迹
- lidar_trajectory.csv 表示雷达里程计或 LiDAR-IMU 轨迹

我也说明一下当前边界：

- 这版已经不是“占位接口”，而是可运行的离线动态复核后端。
- 但它目前假设你已经有两条轨迹 CSV；还没有直接做 rosbag 回放解析。
- 我没有在这里完成一次完整 ROS2 工程编译验证，所以不能声称已经在你本机环境 100% 编译通过；这一步还需要你在现有工作空间里实际编译确认。
- 动态微调采用的是“小范围 hand-eye 一致性微调”，是工程上比较稳的第二阶段实现，和 PDF 的定位一致；它不是把静态法完全改写成一个大而全的联合优化器。

建议你这样用：

```bash
conda deactivate
rm -rf build/lidar_body_calib_ceres_ros2 install/lidar_body_calib_ceres_ros2 log
colcon build --packages-select lidar_body_calib_ceres_ros2
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/humble/lib
source install/setup.bash 
```

```bash
ros2 run lidar_body_calib_ceres_ros2 lidar_body_calib_ceres_ros2 \
--ros-args \
-p config_path:=/home/wangjunfeng/ros2_ws/src/lidar_body_calib_ceres_ros2/config/lidar_config.yaml \
-p target_lidar_ip:=192.168.1.135 \
-p input_topic:=/livox/lidar_192_168_1_135 \
-p accumulation_time_sec:=3.0 \
-p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result.yaml \
-p output_cloud_dir:=/home/wangjunfeng/ros2_ws/output/mid360_calib_outputs \
-p enable_figure8_verification:=false \
-p figure8_data_path:=/home/wangjunfeng/ros2_ws/output/figure8_dataset \
-p figure8_enable_micro_refine:=false
```
其中 /path/to/figure8_dataset 目录下放：

- body_trajectory.csv
- lidar_trajectory.csv


# 支持多雷达版本
核心变化是：

- 支持多雷达同时标定
- 支持从 YAML 按 雷达 IP → topic 自动匹配
- 新增 target_lidar_ips 参数，支持逗号分隔多 IP
- 如果 target_lidar_ips 和 target_lidar_ip 都不填，会自动读取 YAML 顶层的全部雷达 IP
- 输出结果改为一个 YAML 文件里包含多个雷达的结果
  

**这版代码的使用方式建议改成：**

```bash
conda deactivate
rm -rf build/lidar_body_calib_ceres_ros2 install/lidar_body_calib_ceres_ros2 log
colcon build --packages-select lidar_body_calib_ceres_ros2
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/humble/lib
source install/setup.bash 
```

```bash
ros2 run lidar_body_calib_ceres_ros2 lidar_body_calib_ceres_ros2 \
--ros-args \
-p config_path:=/home/wangjunfeng/ros2_ws/src/lidar_body_calib_ceres_ros2/config/lidar_config.yaml \
-p target_lidar_ips:="192.168.1.135,192.168.1.141" \
-p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result_multi.yaml \
-p output_cloud_dir:=/home/wangjunfeng/ros2_ws/output/mid360_calib_outputs
```

也支持只标定一个：
```bash
-p target_lidar_ip:=192.168.1.135
```

```bash
ros2 run lidar_body_calib_ceres_ros2 lidar_body_calib_ceres_ros2 \
--ros-args \
-p config_path:=/home/wangjunfeng/ros2_ws/src/lidar_body_calib_ceres_ros2/config/lidar_config.yaml \
-p target_lidar_ips:=192.168.1.135 \
-p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result_multi.yaml \
-p output_cloud_dir:=/home/wangjunfeng/ros2_ws/output/mid360_calib_outputs
```

还支持完全不传 IP，此时会自动从 YAML 顶层把所有雷达都订阅起来一起标定。

你还需要把 YAML 里的 topic 配成这种形式才行：
```yaml
192.168.1.135:
  topic: /livox/lidar_192_168_1_135
...

192.168.1.141:
  topic: /livox/lidar_192_168_1_141
...
```


# !todo
需要验证平地 8 字轨迹法
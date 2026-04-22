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
ros2 run lidar_body_calib_ceres_ros2 multi_mid360_calibrator --ros-args \
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
rm -rf build/multi_mid360_calibrator install/multi_mid360_calibrator log
colcon build --packages-select multi_mid360_calibrator
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/humble/lib
source install/setup.bash 
```

```bash
ros2 run multi_mid360_calibrator multi_mid360_calibrator \
  --ros-args \
  -p config_path:=/home/wangjunfeng/ros2_ws/src/multi_mid360_calibrator/config/lidar_config.yaml \
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
rm -rf build/multi_mid360_calibrator install/multi_mid360_calibrator log
colcon build --packages-select multi_mid360_calibrator
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/humble/lib
source install/setup.bash 
```

```bash
ros2 run multi_mid360_calibrator multi_mid360_calibrator \
--ros-args \
-p config_path:=/home/wangjunfeng/ros2_ws/src/multi_mid360_calibrator/config/lidar_config.yaml \
-p target_lidar_ip:=192.168.1.135 \
-p input_topic:=/livox/lidar \
-p accumulation_time_sec:=3.0 \
-p roi_distance_threshold:=0.02 \
-p ransac_distance_threshold:=0.01 \
-p output_result_path:=/home/wangjunfeng/ros2_ws/output/lidar_body_calib_result.yaml \
-p output_cloud_dir:=/home/wangjunfeng/ros2_ws/output/mid360_calib_outputs
```
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('lidar_body_calib_ceres_ros2')
    config = os.path.join(pkg_share, 'config', 'lidar_config.yaml')
    return LaunchDescription([
        Node(
            package='lidar_body_calib_ceres_ros2',
            executable='multi_mid360_calibrator',
            name='multi_mid360_calibrator',
            output='screen',
            parameters=[{
                'config_path': config,
                'target_lidar_ip': '192.168.1.135',
                'accumulation_time_sec': 2.0,
                'roi_distance_threshold': 0.08,
                'ransac_distance_threshold': 0.02,
                'min_plane_points': 300,
                'voxel_leaf_size': 0.01,
                'publish_tf': True,
                'output_result_path': '/tmp/lidar_body_calib_result.yaml'
            }]
        )
    ])

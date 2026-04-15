import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("gen3_robot")
    default_model_path = os.path.join(
        pkg_dir, "models", "kinova_gen3", "gen3.xml"
    )

    model_xml_path = LaunchConfiguration("model_xml_path")
    enable_viewer = LaunchConfiguration("enable_viewer")
    publish_rate = LaunchConfiguration("publish_rate")
    use_circle = LaunchConfiguration("use_circle")

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_xml_path",
            default_value=default_model_path,
            description="Path to MuJoCo XML model",
        ),
        DeclareLaunchArgument(
            "enable_viewer",
            default_value="true",
            description="Whether to enable MuJoCo viewer",
        ),
        DeclareLaunchArgument(
            "publish_rate",
            default_value="100.0",
            description="Simulation publish rate",
        ),
        DeclareLaunchArgument(
            "use_circle",
            default_value="true",
            description="Whether to launch circular trajectory publisher",
        ),

        Node(
            package="gen3_robot",
            executable="gen3_robot_node",
            name="gen3_robot_node",
            output="screen",
            parameters=[{
                "model_xml_path": model_xml_path,
                "publish_rate": publish_rate,
                "enable_viewer": enable_viewer,
            }],
        ),

        Node(
            package="gen3_robot",
            executable="trajectory_publisher",
            name="circle_trajectory_publisher",
            output="screen",
            parameters=[{
                "topic_name": "ee_pose_target",
                "publish_rate": 50.0,
                "center_x": 0.45,
                "center_y": 0.0,
                "center_z": 0.35,
                "radius": 0.08,
                "angular_speed": 0.8,
            }],
            condition=None,
        ),
    ])



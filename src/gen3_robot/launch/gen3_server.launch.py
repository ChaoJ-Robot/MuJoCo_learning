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

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_xml_path",
            default_value=default_model_path,
        ),
        DeclareLaunchArgument(
            "enable_viewer",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "publish_rate",
            default_value="100.0",
        ),
        Node(
            package="gen3_robot",
            executable="gen3_server",
            name="gen3_server",
            output="screen",
            parameters=[{
                "model_xml_path": LaunchConfiguration("model_xml_path"),
                "enable_viewer": LaunchConfiguration("enable_viewer"),
                "publish_rate": LaunchConfiguration("publish_rate"),
            }],
        ),
    ])
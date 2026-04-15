import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包目录
    pkg_dir = get_package_share_directory('gen3_robot')
    
    # 定义启动参数
    model_xml_path = LaunchConfiguration('model_xml_path')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # 默认模型文件路径
    default_model_path = os.path.join(
        pkg_dir, 'resource', 'models', 'kinova_gen3', 'gen3.xml'
    )
    
    return LaunchDescription([
        # 声明启动参数
        DeclareLaunchArgument(
            'model_xml_path',
            default_value=default_model_path,
            description='Path to the MuJoCo model XML file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # 启动MuJoCo ROS2节点
        Node(
            package='gen3_robot',
            executable='gen3_robot_node',
            name='gen3_robot_node',
            output='screen',
            parameters=[{
                'model_xml_path': model_xml_path,
                'use_sim_time': use_sim_time,
            }],
            remappings=[
                # 可以在这里添加重映射
            ]
        ),
    ])
cd /home/jack/Desktop/mujoco/mujoco_demo
source gen3_env/bin/activate
source /opt/ros/humble/setup.bash
source install/setup.bash

# 终端1：启动MuJoCo仿真
python3 /home/jack/Desktop/mujoco/mujoco_demo/src/gen3_robot/gen3_robot/gen3_robot.py

# 终端2：启动
ros2 run orbbec_camera orbbec_camera_node &

# 终端3：启动检测器
python3 src/gen3_robot/gen3_robot/color_detector.py

import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jack/Desktop/mujoco/mujoco_demo/install/gen3_robot'

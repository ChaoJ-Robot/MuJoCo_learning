#!/usr/bin/env python3

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .gen3_controller import Gen3MujocoController

 
def rotation_matrix_to_quaternion(R: np.ndarray):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = np.trace(R)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w], dtype=float)


class Gen3RobotNode(Node):
    def __init__(self):
        super().__init__("gen3_robot_node")

        pkg_share = get_package_share_directory("gen3_robot")
        default_model_xml = os.path.join(
            pkg_share, "models", "kinova_gen3", "gen3.xml"
        )

        self.declare_parameter("model_xml_path", default_model_xml)
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("enable_viewer", True)
        self.declare_parameter("site_name", "pinch_site")

        model_xml_path = self.get_parameter("model_xml_path").value
        publish_rate = float(self.get_parameter("publish_rate").value)
        enable_viewer = bool(self.get_parameter("enable_viewer").value)
        site_name = self.get_parameter("site_name").value

        self.controller = Gen3MujocoController(
            model_xml_path=model_xml_path,
            site_name=site_name,
        )
        self.enable_viewer = enable_viewer

        if self.enable_viewer:
            self.get_logger().info("正在启动 MuJoCo viewer...")
            self.controller.launch_viewer()
            self.get_logger().info("MuJoCo viewer 已启动")

        self.joint_state_pub = self.create_publisher(JointState, "joint_states", 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, "ee_pose", 10)

        self.joint_target_sub = self.create_subscription(
            Float64MultiArray,
            "joint_target",
            self.joint_target_callback,
            10,
        )

        self.ee_pose_target_sub = self.create_subscription(
            PoseStamped,
            "ee_pose_target",
            self.ee_pose_target_callback,
            10,
        )

        period = 1.0 / publish_rate if publish_rate > 0.0 else 0.01
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info("Gen3 Mujoco ROS2 节点启动成功")
        self.get_logger().info(f"模型路径: {model_xml_path}")

    def joint_target_callback(self, msg: Float64MultiArray):
        try:
            if len(msg.data) != 7:
                self.get_logger().warn("joint_target 必须是长度为7的数组")
                return
            self.controller.set_joint_target(list(msg.data))
        except Exception as e:
            self.get_logger().error(f"设置关节目标失败: {e}")

    def ee_pose_target_callback(self, msg: PoseStamped):
        try:
            pos_des = np.array(
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ],
                dtype=float,
            )
            self.controller.set_ee_pose(pos_des, rot_des=None)
        except Exception as e:
            self.get_logger().error(f"设置末端目标失败: {e}")

    def publish_joint_state(self):
        state = self.controller.get_state()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"
        ]
        msg.position = state["qpos"].tolist()
        msg.velocity = state["qvel"].tolist()
        msg.effort = state["ctrl"].tolist()
        self.joint_state_pub.publish(msg)

    def publish_ee_pose(self):
        ee_pos, ee_rot = self.controller.get_eef_state()
        quat = rotation_matrix_to_quaternion(ee_rot)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position = Point(
            x=float(ee_pos[0]),
            y=float(ee_pos[1]),
            z=float(ee_pos[2]),
        )
        msg.pose.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        )
        self.ee_pose_pub.publish(msg)

    def timer_callback(self):
        try:
            self.controller.step()
            self.publish_joint_state()
            self.publish_ee_pose()
            if self.enable_viewer:
                self.controller.sync_viewer()
        except Exception as e:
            self.get_logger().error(f"定时器执行失败: {e}")

    def destroy_node(self):
        try:
            self.controller.close_viewer()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Gen3RobotNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()



#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class CommandSender(Node):
    def __init__(self):
        super().__init__("command_sender")

        self.ee_pub = self.create_publisher(PoseStamped, "ee_pose_target", 10)
        self.cmd_pub = self.create_publisher(String, "robot_command", 10)

        self.declare_parameter("mode", "ee_target")   # ee_target / home
        self.declare_parameter("x", 0.45)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", 0.35)

        self.mode = self.get_parameter("mode").value
        self.x = float(self.get_parameter("x").value)
        self.y = float(self.get_parameter("y").value)
        self.z = float(self.get_parameter("z").value)

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.sent = False

    def timer_callback(self):
        if self.sent:
            return

        if self.mode == "home":
            msg = String()
            msg.data = "home"
            self.cmd_pub.publish(msg)
            self.get_logger().info("已发送 home 命令")
        elif self.mode == "ee_target": 
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.pose.position.x = self.x
            msg.pose.position.y = self.y
            msg.pose.position.z = self.z
            msg.pose.orientation.w = 1.0
            self.ee_pub.publish(msg)
            self.get_logger().info(f"已发送末端目标: ({self.x}, {self.y}, {self.z})")

        self.sent = True


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CommandSender()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class CircleTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("circle_trajectory_publisher")

        self.declare_parameter("topic_name", "ee_pose_target")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("center_x", 0.45)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("center_z", 0.35)
        self.declare_parameter("radius", 0.08)
        self.declare_parameter("angular_speed", 0.6)

        topic_name = self.get_parameter("topic_name").value
        publish_rate = float(self.get_parameter("publish_rate").value)

        self.cx = float(self.get_parameter("center_x").value)
        self.cy = float(self.get_parameter("center_y").value)
        self.cz = float(self.get_parameter("center_z").value)
        self.radius = float(self.get_parameter("radius").value)
        self.omega = float(self.get_parameter("angular_speed").value)

        self.pub = self.create_publisher(PoseStamped, topic_name, 10)
        self.t = 0.0

        period = 1.0 / publish_rate if publish_rate > 0.0 else 0.02
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info("圆轨迹发布节点已启动")

    def timer_callback(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        x = self.cx
        y = self.cy + self.radius * math.cos(self.omega * self.t)
        z = self.cz + self.radius * math.sin(self.omega * self.t)

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        self.pub.publish(msg)
        self.t += self.timer.timer_period_ns * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = CircleTrajectoryPublisher()
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
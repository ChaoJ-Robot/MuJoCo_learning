#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


class ButtonEvent(Node):
    def __init__(self):
        super().__init__("button_event")

        self.axes = [0.0] * 8
        self.buttons = [0] * 19
        self.prev_buttons = [0] * 19

        self.current_ee_pos = None
        self.target_pos = None

        self.deadzone = 0.20
        self.linear_speed = 0.20
        self.dt = 0.01

        self.ee_pub = self.create_publisher(PoseStamped, "ee_pose_target", 10)
        self.cmd_pub = self.create_publisher(String, "robot_command", 10)

        self.create_subscription(Joy, "joy", self.joy_callback, 10)
        self.create_subscription(PoseStamped, "ee_pose", self.ee_pose_callback, 10)

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info("ButtonEvent node has been started")

    def joy_callback(self, msg: Joy):
        self.axes = list(msg.axes)
        self.buttons = list(msg.buttons)

    def ee_pose_callback(self, msg: PoseStamped):
        self.current_ee_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=float)

        if self.target_pos is None:
            self.target_pos = self.current_ee_pos.copy()
            self.get_logger().info(
                f"初始化目标点: "
                f"x={self.target_pos[0]:.3f}, "
                f"y={self.target_pos[1]:.3f}, "
                f"z={self.target_pos[2]:.3f}"
            )

    def apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.deadzone else value

    def edge_pressed(self, idx: int) -> bool:
        if idx >= len(self.buttons) or idx >= len(self.prev_buttons):
            return False
        return self.buttons[idx] == 1 and self.prev_buttons[idx] == 0

    def send_command(self, text: str):
        msg = String()
        msg.data = text
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Published command: {text}")

    def send_ee_target(self):
        if self.target_pos is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = float(self.target_pos[0])
        msg.pose.position.y = float(self.target_pos[1])
        msg.pose.position.z = float(self.target_pos[2])

        msg.pose.orientation.w = 1.0
        self.ee_pub.publish(msg)

    def timer_callback(self):
        if self.target_pos is None:
            return

        left_x = self.apply_deadzone(self.axes[0]) if len(self.axes) > 0 else 0.0
        left_y = self.apply_deadzone(self.axes[1]) if len(self.axes) > 1 else 0.0
        right_y = self.apply_deadzone(self.axes[3]) if len(self.axes) > 3 else 0.0

        moved = False

        # 左摇杆左右：末端 y 增减
        if left_x != 0.0:
            self.target_pos[1] += self.linear_speed * self.dt * left_x
            moved = True

        # 左摇杆上下：末端 z 增减
        # 上推通常是负值，所以这里用减号更符合直觉
        if left_y != 0.0:
            self.target_pos[2] -= self.linear_speed * self.dt * left_y
            moved = True

        # 右摇杆上下：末端 x 增减
        if right_y != 0.0:
            self.target_pos[0] += self.linear_speed * self.dt * right_y
            moved = True

        # 只有真的有摇杆输入时才限幅和发送目标
        if moved:
            self.target_pos[0] = np.clip(self.target_pos[0], -0.50, 0.50)
            self.target_pos[1] = np.clip(self.target_pos[1], -0.50, 0.50)
            self.target_pos[2] = np.clip(self.target_pos[2], 0.10, 1.00)
            self.send_ee_target()

        # A：home
        if self.edge_pressed(0):
            self.send_command("home")

        # B：stop
        if self.edge_pressed(1):
            self.send_command("stop")

        # X：打印目标点和实际点
        if self.edge_pressed(2):
            target_str = (
                f"target=({self.target_pos[0]:.3f}, "
                f"{self.target_pos[1]:.3f}, "
                f"{self.target_pos[2]:.3f})"
            )
            if self.current_ee_pos is not None:
                actual_str = (
                    f"actual=({self.current_ee_pos[0]:.3f}, "
                    f"{self.current_ee_pos[1]:.3f}, "
                    f"{self.current_ee_pos[2]:.3f})"
                )
            else:
                actual_str = "actual=None"
            self.get_logger().info(f"{target_str}, {actual_str}")

        # Y：把目标点同步到当前实际位置
        if self.edge_pressed(3):
            if self.current_ee_pos is not None:
                self.target_pos = self.current_ee_pos.copy()
                self.get_logger().info(
                    f"目标点已同步到实际位置: "
                    f"x={self.target_pos[0]:.3f}, "
                    f"y={self.target_pos[1]:.3f}, "
                    f"z={self.target_pos[2]:.3f}"
                )

        # LB：速度减半
        if self.edge_pressed(4):
            self.linear_speed = max(0.02, self.linear_speed * 0.5)
            self.get_logger().info(f"Linear speed reduced to {self.linear_speed:.3f}")

        # RB：速度翻倍
        if self.edge_pressed(5):
            self.linear_speed = min(0.50, self.linear_speed * 2.0)
            self.get_logger().info(f"Linear speed increased to {self.linear_speed:.3f}")

        self.prev_buttons = self.buttons.copy()


def main(args=None):
    rclpy.init(args=args)
    node = ButtonEvent()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
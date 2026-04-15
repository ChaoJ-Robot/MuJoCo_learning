#!/usr/bin/env python3

import os
import math
import time
import threading
import importlib.util
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Float64MultiArray


def load_base_controller():
    """
    从安装后的 share 目录动态加载 base_controller.py
    """
    pkg_share = get_package_share_directory('gen3_robot')
    base_controller_path = os.path.join(
        pkg_share,
        'resource',
        'models',
        'kinova_gen3',
        'base_controller.py'
    )

    if not os.path.exists(base_controller_path):
        raise FileNotFoundError(f'未找到 base_controller.py: {base_controller_path}')

    spec = importlib.util.spec_from_file_location(
        'gen3_base_controller_module',
        base_controller_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseController


BaseController = load_base_controller()


def rotation_matrix_to_quaternion(R: np.ndarray):
    """
    3x3 旋转矩阵转四元数 (x, y, z, w)
    """
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


class Gen3MujocoController(BaseController):
    def __init__(self, model_xml_path: str):
        super().__init__(name='Gen3MujocoController')

        if not os.path.exists(model_xml_path):
            raise FileNotFoundError(f'模型文件不存在: {model_xml_path}')

        self.model_xml_path = model_xml_path
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)

        self.ndof = 7
        self.kp_pos = 3.0
        self.kp_rot = 2.0
        self.damping = 1e-4

        self.site_name = 'pinch_site'
        self.site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_name
        )
        if self.site_id == -1:
            raise ValueError(f'没有找到末端 site: {self.site_name}')

        self.key_name = 'home'
        self.key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            self.key_name
        )

        self.lock = threading.Lock()
        self.viewer = None

        self.set_up()

    def set_up(self, **kwargs):
        initial_qpos = kwargs.get('initial_qpos', None)

        with self.lock:
            if initial_qpos is None:
                if self.key_id != -1:
                    mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
                else:
                    self.data.qpos[:self.ndof] = 0.0
                    self.data.qvel[:self.ndof] = 0.0
                    if self.model.nu >= self.ndof:
                        self.data.ctrl[:self.ndof] = self.data.qpos[:self.ndof]
            else:
                q = np.asarray(initial_qpos, dtype=float)
                if q.shape != (self.ndof,):
                    raise ValueError('initial_qpos 必须是长度为7的数组')
                self.data.qpos[:self.ndof] = q
                self.data.qvel[:self.ndof] = 0.0
                if self.model.nu >= self.ndof:
                    self.data.ctrl[:self.ndof] = q

            mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        with self.lock:
            ee_pos = self.data.site_xpos[self.site_id].copy()
            ee_rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

            return {
                'time': float(self.data.time),
                'qpos': self.data.qpos[:self.ndof].copy(),
                'qvel': self.data.qvel[:self.ndof].copy(),
                'ctrl': self.data.ctrl[:self.ndof].copy() if self.model.nu >= self.ndof else np.zeros(self.ndof),
                'ee_pos': ee_pos,
                'ee_rot': ee_rot,
            }

    def get_eef_state(self):
        state = self.get_state()
        return state['ee_pos'], state['ee_rot']

    def set_action(self, action: list, **kwargs):
        self.set_joint_target(action)

    def set_joint_target(self, q_target):
        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape != (self.ndof,):
            raise ValueError('q_target 必须是长度为7的数组')

        with self.lock:
            if self.model.nu <= 0:
                raise RuntimeError('模型没有 actuator，无法写入 ctrl')

            nctrl = min(self.ndof, self.model.nu)
            ctrl = q_target.copy()

            for i in range(nctrl):
                if self.model.actuator_ctrllimited[i]:
                    low, high = self.model.actuator_ctrlrange[i]
                    ctrl[i] = np.clip(ctrl[i], low, high)

            self.data.ctrl[:nctrl] = ctrl[:nctrl]

    def compute_pose_error(self, pos_des, rot_des=None):
        pos_cur, rot_cur = self.get_eef_state()
        e_pos = np.asarray(pos_des, dtype=float).reshape(3,) - pos_cur

        if rot_des is None:
            e_rot = np.zeros(3)
        else:
            rot_des = np.asarray(rot_des, dtype=float).reshape(3, 3)
            e_rot = 0.5 * (
                np.cross(rot_cur[:, 0], rot_des[:, 0]) +
                np.cross(rot_cur[:, 1], rot_des[:, 1]) +
                np.cross(rot_cur[:, 2], rot_des[:, 2])
            )

        return e_pos, e_rot

    def compute_jacobian(self):
        with self.lock:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            return jacp[:, :self.ndof], jacr[:, :self.ndof]

    def set_eef_action(self, pos_des, rot_des=None):
        self.set_ee_pose(pos_des, rot_des)

    def set_ee_pose(self, pos_des, rot_des=None):
        e_pos, e_rot = self.compute_pose_error(pos_des, rot_des)
        jacp, jacr = self.compute_jacobian()

        if rot_des is None:
            J = jacp
            e = self.kp_pos * e_pos
            A = J @ J.T + self.damping * np.eye(3)
            qdot = J.T @ np.linalg.solve(A, e)
        else:
            J = np.vstack([jacp, jacr])
            e = np.concatenate([self.kp_pos * e_pos, self.kp_rot * e_rot])
            A = J @ J.T + self.damping * np.eye(6)
            qdot = J.T @ np.linalg.solve(A, e)

        with self.lock:
            q_des = self.data.qpos[:self.ndof].copy() + qdot * self.model.opt.timestep

        self.set_joint_target(q_des)

    def step(self, nstep=1):
        with self.lock:
            mujoco.mj_step(self.model, self.data, nstep=nstep)

    def launch_viewer(self):
        """
        启动 MuJoCo 可视化窗口
        """
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def sync_viewer(self):
        if self.viewer is not None and self.viewer.is_running():
            with self.lock:
                self.viewer.sync()

    def close_viewer(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


class Gen3RobotNode(Node):
    def __init__(self):
        super().__init__('gen3_robot_node')

        pkg_share = get_package_share_directory('gen3_robot')
        default_model_xml = os.path.join(
            pkg_share,
            'resource',
            'models',
            'kinova_gen3',
            'gen3.xml'
        )

        self.declare_parameter('model_xml_path', default_model_xml)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('enable_viewer', True)

        model_xml_path = self.get_parameter('model_xml_path').value
        publish_rate = float(self.get_parameter('publish_rate').value)
        enable_viewer = bool(self.get_parameter('enable_viewer').value)

        self.controller = Gen3MujocoController(model_xml_path)
        self.enable_viewer = enable_viewer

        if self.enable_viewer:
            self.get_logger().info('正在启动 MuJoCo viewer...')
            self.controller.launch_viewer()
            self.get_logger().info('MuJoCo viewer 已启动')

        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, 'ee_pose', 10)

        self.joint_target_sub = self.create_subscription(
            Float64MultiArray,
            'joint_target',
            self.joint_target_callback,
            10
        )

        self.ee_pose_target_sub = self.create_subscription(
            PoseStamped,
            'ee_pose_target',
            self.ee_pose_target_callback,
            10
        )

        period = 1.0 / publish_rate if publish_rate > 0.0 else 0.01
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info('Gen3 Mujoco ROS2 节点启动成功')
        self.get_logger().info(f'模型路径: {model_xml_path}')

    def joint_target_callback(self, msg: Float64MultiArray):
        try:
            if len(msg.data) != 7:
                self.get_logger().warn('joint_target 必须是长度为7的数组')
                return
            self.controller.set_joint_target(list(msg.data))
        except Exception as e:
            self.get_logger().error(f'设置关节目标失败: {e}')

    def ee_pose_target_callback(self, msg: PoseStamped):
        try:
            pos_des = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ], dtype=float)

            self.controller.set_ee_pose(pos_des, rot_des=None)
        except Exception as e:
            self.get_logger().error(f'设置末端目标失败: {e}')

    def publish_joint_state(self):
        state = self.controller.get_state()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]
        msg.position = state['qpos'].tolist()
        msg.velocity = state['qvel'].tolist()
        msg.effort = state['ctrl'].tolist()

        self.joint_state_pub.publish(msg)

    def publish_ee_pose(self):
        ee_pos, ee_rot = self.controller.get_eef_state()
        quat = rotation_matrix_to_quaternion(ee_rot)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position = Point(
            x=float(ee_pos[0]),
            y=float(ee_pos[1]),
            z=float(ee_pos[2])
        )
        msg.pose.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3])
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
            self.get_logger().error(f'定时器执行失败: {e}')

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
    except Exception as e:
        print(f'节点启动失败: {e}')
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


if __name__ == '__main__':
    main()
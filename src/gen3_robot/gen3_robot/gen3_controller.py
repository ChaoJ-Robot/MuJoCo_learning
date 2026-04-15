#!/usr/bin/env python3

import os
import threading
from typing import Any, Dict

import mujoco
import mujoco.viewer
import numpy as np
import math

from .base_controller import BaseController


class Gen3MujocoController(BaseController):
    def __init__(self, model_xml_path: str, site_name: str = "pinch_site"):
        super().__init__(name="Gen3MujocoController")

        if not os.path.exists(model_xml_path):
            raise FileNotFoundError(f"模型文件不存在: {model_xml_path}")

        self.model_xml_path = model_xml_path
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)

        self.ndof = 7
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name
        )

        if self.site_id == -1:
            raise ValueError(f"未找到末端 site: {self.site_name}")

        self.key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )

        self.kp_pos = 3.0
        self.kp_rot = 2.0
        self.damping = 1e-4

        self.lock = threading.Lock()
        self.viewer = None

        self.set_up()

    def set_up(self, **kwargs: Any) -> None:
        initial_qpos = [0.0, 0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2, 0.0]

        self.data.qpos[:self.ndof] = initial_qpos
        self.data.ctrl[:self.ndof] = initial_qpos

        mujoco.mj_forward(self.model, self.data)
        print("初始位置设置完成: qpos =", self.data.qpos[:self.ndof])

    def reset(self):
        self.set_up()
 
    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            ee_pos = self.data.site_xpos[self.site_id].copy()
            ee_rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            return {
                "time": float(self.data.time),
                "qpos": self.data.qpos[:self.ndof].copy(),
                "qvel": self.data.qvel[:self.ndof].copy(),
                "ctrl": self.data.ctrl[:self.ndof].copy() if self.model.nu >= self.ndof else np.zeros(self.ndof),
                "ee_pos": ee_pos,
                "ee_rot": ee_rot,
            }

    def get_eef_state(self):
        state = self.get_state()
        print(f"当前末端位置: {state['ee_pos']}, {state['ee_rot']}")
        return state["ee_pos"], state["ee_rot"]

    def set_action(self, action, **kwargs: Any) -> None:
        self.set_joint_target(action)

    def set_joint_target(self, q_target):
        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape != (self.ndof,):
            raise ValueError("q_target 必须是长度为7的数组")

        with self.lock:
            nctrl = min(self.ndof, self.model.nu)
            self.data.ctrl[:nctrl] = q_target[:nctrl]

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

    def step(self, nstep: int = 1):
        with self.lock:
            mujoco.mj_step(self.model, self.data, nstep=nstep)

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

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
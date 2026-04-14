import mujoco
import time 

import mujoco.viewer 
import numpy as np
from base_controller import BaseController


class Gen3Robot(BaseController):
    def __init__(self, model_xml_path):
        super().__init__(name="Gen3Robot")

        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,"pinch_site")
        self.key_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_KEY,"home")

        self.ndof = 7

        self.kp_pos = 3.0      # 位置控制的比例增益
        self.kp_rot = 2.0      # 旋转控制的比例增益  
        self.damping = 1e-4    # 阻尼系数

        self.reset_to_home()

    def reset_to_home(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)
        
    def set_up(self, initial_qpos=None):
        if initial_qpos is None:
            self.reset_to_home()
        else:
            initial_qpos = np.asarray(initial_qpos, dtype=float)
            if initial_qpos.shape != (7,):
                raise ValueError("initial_qpos 必须是长度为7的数组")

            self.data.qpos[:7] = initial_qpos
            self.data.ctrl[:7] = initial_qpos
            mujoco.mj_forward(self.model, self.data)

        print("qpos =", self.data.qpos[:7])
        print("初始位置设置完成")


    def get_state(self):
        # 末端位置
        ee_pos = self.data.site_xpos[self.site_id].copy()

        # 当前末端旋转矩阵（9个数 reshape 成 3x3）
        ee_rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos[:7].copy(),
            "qvel": self.data.qvel[:7].copy(),
            "ctrl": self.data.ctrl[:7].copy(),
            "ee_pos": ee_pos,
            "ee_rot": ee_rot,
        }

    def get_ee_pose(self):
        state = self.get_state()
        return state["ee_pos"], state["ee_rot"]
    
    def set_joint_target(self, q_target):
        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape != (7,):
            raise ValueError("q_target 必须是长度为7的数组")
        self.data.ctrl[:7] = self.kp_pos * (q_target - self.data.qpos[:7]) - self.damping * self.data.qvel[:7]

        ctrl = q_target.copy()
        
        # 若 actuator 有 ctrlrange，则裁剪
        for i in range(min(self.ndof, self.model.nu)):
            if self.model.actuator_ctrllimited[i]:
                low, high = self.model.actuator_ctrlrange[i]
                ctrl[i] = np.clip(ctrl[i], low, high)

        self.data.ctrl[:7] = ctrl

    def compute_pose_error(self, pos_des, rot_des=None):
        """
        pos_des: (3,)
        rot_des: (3,3) or None
        返回:
            e_pos: (3,)
            e_rot: (3,)
        """
        pos_cur, rot_cur = self.get_ee_pose()
        e_pos = np.asarray(pos_des, dtype=float) - pos_cur

        if rot_des is None:
            e_rot = np.zeros(3)
        else:
            rot_des = np.asarray(rot_des, dtype=float).reshape(3, 3)

            # 小角度姿态误差：0.5 * sum(cross(R_cur[:,i], R_des[:,i]))
            e_rot = 0.5 * (
                np.cross(rot_cur[:, 0], rot_des[:, 0]) +
                np.cross(rot_cur[:, 1], rot_des[:, 1]) +
                np.cross(rot_cur[:, 2], rot_des[:, 2])
            )

        return e_pos, e_rot

    def compute_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        # 你的机械臂是7自由度，取前7列
        return jacp[:, :self.ndof], jacr[:, :self.ndof]

    def set_ee_pose(self, pos_des, rot_des=None):
        """
        末端位姿控制:
        1) 计算末端误差
        2) 用 Jacobian 反解关节速度
        3) 积分成关节目标
        4) 写入 position actuator 的 ctrl
        """
        e_pos, e_rot = self.compute_pose_error(pos_des, rot_des)
        jacp, jacr = self.compute_jacobian()

        if rot_des is None:
            # 只做位置控制
            J = jacp
            e = self.kp_pos * e_pos
            A = J @ J.T + self.damping * np.eye(3)
            qdot = J.T @ np.linalg.solve(A, e)
        else:
            # 6D 位姿控制
            J = np.vstack([jacp, jacr])
            e = np.concatenate([
                self.kp_pos * e_pos,
                self.kp_rot * e_rot
            ])
            A = J @ J.T + self.damping * np.eye(6)
            qdot = J.T @ np.linalg.solve(A, e)

        q_des = self.data.qpos[:7].copy() + qdot * self.model.opt.timestep
        self.set_joint_target(q_des)

    def step(self, nstep=1):
        mujoco.mj_step(self.model, self.data, nstep=nstep)

    def run_position_demo(self, target_pos):
        """
        最小示例：让末端移动到目标位置
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                self.set_ee_pose(target_pos)   # 只控位置
                self.step()
                viewer.sync()

    def run_circle_demo(self, center, radius=0.03, omega=0.5):
        """
        末端画圆：只控制位置，姿态保持自由
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t = self.data.time
                pos_des = np.array([
                    center[0] + radius * np.cos(omega * t),
                    center[1] + radius * np.sin(omega * t),
                    center[2],
                ])

                self.set_ee_pose(pos_des)
                self.step()
                viewer.sync()              
    def set_action(self, action, **kwargs):

        return super().set_action(action, **kwargs)

if __name__ == "__main__":
    robot = Gen3Robot("gen3.xml")
    robot.set_up()

    ee_pos, ee_rot = robot.get_ee_pose()
    print("当前末端位置:", ee_pos)
    print("当前末端姿态:\n", ee_rot)

    # 目标点：在当前基础上往 x 方向移动 5cm
    target_pos = ee_pos + np.array([0.05, 0.0, 0.0])

    # 先试位置控制
    robot.run_position_demo(target_pos)



import mujoco
import time
import mujoco.viewer
import numpy as np
from gen3_robot.gen3_robot.base_controller import BaseController

import matplotlib.pyplot as plt


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
        
        self.ee_traj = []   # 用来记录末端轨迹
        self.reset_to_home()

    def reset_to_home(self):
        if self.key_id == -1:
            raise ValueError("没有找到 keyframe: home")
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

        print(f"End-Effector Position: {state['ee_pos']}")
        print(f"End-Effector Rotation: {state['ee_rot']}")
        
        return state["ee_pos"], state["ee_rot"]
    
    def set_joint_target(self, q_target):
        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape != (7,):
            raise ValueError("q_target 必须是长度为7的数组")

        ctrl = q_target.copy()

        for i in range(min(self.ndof, self.model.nu)):
            if self.model.actuator_ctrllimited[i]:
                low, high = self.model.actuator_ctrlrange[i]
                ctrl[i] = np.clip(ctrl[i], low, high)

        self.data.ctrl[:7] = ctrl

    def compute_pose_error(self, pos_des, rot_des=None):
        """
        pos_des: (3,)
        rot_des: (3,3) or None

        计算当前末端和目标末端差多少
        
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
        jacp = np.zeros((3, self.model.nv))  #位置雅可比
        jacr = np.zeros((3, self.model.nv))  #旋转雅可比
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        # 取前7列
        return jacp[:, :self.ndof], jacr[:, :self.ndof] #取前7列，对应7个关节

    def set_ee_pose(self, pos_des, rot_des=None):
        """
        末端位姿控制:
        1) 计算末端误差
        2) 用 Jacobian 反解关节速度
        3) 积分成关节目标
        4) 写入 position actuator 的 ctrl
        """
        e_pos, e_rot = self.compute_pose_error(pos_des, rot_des)  #位置和姿态误差
        jacp, jacr = self.compute_jacobian()    #3x7 的位置雅可比，3x7 的旋转雅可比 

        if rot_des is None:
            # 只做位置控制
            J = jacp  # 3x7 的位置雅可比
            e = self.kp_pos * e_pos   # 比例控制：误差 × 增益
            A = J @ J.T + self.damping * np.eye(3)  #阻尼最小二乘法
            '''
                雅可比矩阵： v = J · q̇
                逆运动学求解 ： q̇ = J⁺ · v_des
                为了避免奇异位形问题，使用阻尼伪逆：
                J⁺ = Jᵀ · (J·Jᵀ + λ²I)⁻¹
            
            '''
            qdot = J.T @ np.linalg.solve(A, e) 
        else:
            # 6D 位姿控制
            J = np.vstack([jacp, jacr])
            e = np.concatenate([
                self.kp_pos * e_pos,  #位置误差乘以位置增益
                self.kp_rot * e_rot   #姿态误差乘以姿态增益
            ])
            A = J @ J.T + self.damping * np.eye(6)
            qdot = J.T @ np.linalg.solve(A, e)

        q_des = self.data.qpos[:7].copy() + qdot * self.model.opt.timestep # 积分成关节位置
        self.set_joint_target(q_des) 

    def step(self, nstep=1):
        mujoco.mj_step(self.model, self.data, nstep=nstep)

    def run_position_demo(self, target_pos):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.clear_ee_traj()

            while viewer.is_running():
                self.set_ee_pose(target_pos)
                self.record_ee_traj()   # 每一步记录末端位置
                self.step()
                viewer.sync()

    def run_circle_demo(self, center, radius=0.03, omega=0.5):
        """
        末端画圆：只控制位置，姿态保持自由
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.clear_ee_traj()
            while viewer.is_running():
                t = self.data.time
                # 圆的参数方程
                pos_des = np.array([
                    center[0] + radius * np.cos(omega * t),
                    center[1] + radius * np.sin(omega * t),
                    center[2],
                ])

                self.set_ee_pose(pos_des)
                self.record_ee_traj()  # 记录末端轨迹
                self.step()
                viewer.sync()              
    def set_action(self, action, **kwargs):

        return super().set_action(action, **kwargs)
    def record_ee_traj(self):
        ee_pos, _ = self.get_ee_pose()
        self.ee_traj.append(ee_pos.copy())

    def clear_ee_traj(self):
        self.ee_traj = []

    def plot_ee_traj(self, show_3d=False):
        if not hasattr(self, 'ee_traj') or len(self.ee_traj) == 0:
            print("轨迹为空，先运行控制程序并记录轨迹")
            return

        traj = np.array(self.ee_traj)
        
        print("\n=== 轨迹信息 ===")
        print(f"轨迹点数: {len(traj)}")
        print(f"轨迹范围:")
        print(f"  X: {traj[:, 0].min():.3f} 到 {traj[:, 0].max():.3f}")
        print(f"  Y: {traj[:, 1].min():.3f} 到 {traj[:, 1].max():.3f}")
        print(f"  Z: {traj[:, 2].min():.3f} 到 {traj[:, 2].max():.3f}")
        print(f"起点: [{traj[0, 0]:.3f}, {traj[0, 1]:.3f}, {traj[0, 2]:.3f}]")
        print(f"终点: [{traj[-1, 0]:.3f}, {traj[-1, 1]:.3f}, {traj[-1, 2]:.3f}]")
        
        # 计算轨迹长度
        total_distance = 0
        for i in range(1, len(traj)):
            distance = np.linalg.norm(traj[i] - traj[i-1])
            total_distance += distance
        print(f"轨迹总长度: {total_distance:.3f} m")
        print("================\n")
        
        # 如果需要2D绘图，可以取消注释下面的代码
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(traj[:, 0], traj[:, 1], linewidth=2)
            plt.scatter(traj[0, 0], traj[0, 1], marker='o', s=50, label='start')
            plt.scatter(traj[-1, 0], traj[-1, 1], marker='x', s=50, label='end')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("End-Effector Trajectory (XY)")
            plt.axis("equal")
            plt.legend()
            plt.show()
        except ImportError:
            print("matplotlib不可用，跳过图形显示")

if __name__ == "__main__":
    robot = Gen3Robot("gen3.xml")
    robot.set_up()

    ee_pos, ee_rot = robot.get_ee_pose()
    print("当前末端位置:", ee_pos)
    print("当前末端姿态:\n", ee_rot)
    
    # 测试圆形轨迹
    print("\n" + "="*50)
    print("开始圆形轨迹测试")
    print("="*50)
    
    # 使用现有的draw_circle函数
    robot.run_circle_demo(center=[0, 0, 0.8], radius=0.05, omega=0.5)
    robot.plot_ee_traj()
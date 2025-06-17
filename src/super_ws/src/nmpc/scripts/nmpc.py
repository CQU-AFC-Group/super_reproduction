#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PolynomialTrajectory, PositionCommand
import casadi as ca


class NMPC:
    def __init__(self):
        rospy.init_node('nmpc', anonymous=True)

        self.current_odom : Optional[Odometry] = None
        self.current_traj : Optional[PolynomialTrajectory]= None

        self.ctr_freq = 10.0
        self.prediction_horizon = 10

        self.odom_sub = rospy.Subscriber('/lidar_slam/odom', Odometry, self.odom_callback)
        self.traj_sub = rospy.Subscriber('/planning_cmd/poly_traj', PolynomialTrajectory, self.traj_callback)
        self.cmd_pub = rospy.Publisher('/planning/pos_cmd_nmpc', PositionCommand, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0/self.ctr_freq), self.timer_callback)

        # 数据记录
        self.recorded_ref_odom = []  # 记录实际轨迹
        self.recorded_ref_path = []  # 记录参考轨迹
        self.recorded_ref_yaw = []  # 记录参考轨迹
        self.recorded_cmd_path = []  # 记录控制轨迹
        self.start_time = rospy.Time.now()

        # 控制参数
        self.Q = np.diag([10000, 10000, 10000])  # 状态权重
        self.R = np.diag([0.1, 0.1, 0.1])        # 控制输入权重
        # 定义系统模型 (三阶积分模型)
        self.setup_model()
        
    def setup_model(self):
        # 状态变量 [x, y, z, vx, vy, vz, ax, ay, az]
        x = ca.MX.sym('x', 9)
        # 控制输入 [jx, jy, jz] (jerk)
        u = ca.MX.sym('u', 3)
        
        # 动力学方程 (三阶积分模型)
        xdot = ca.vertcat(
            x[3],           # dx/dt = vx
            x[4],           # dy/dt = vy
            x[5],           # dz/dt = vz
            x[6],           # dvx/dt = ax
            x[7],           # dvy/dt = ay
            x[8],           # dvz/dt = az
            u[0],          # dax/dt = jx
            u[1],          # day/dt = jy
            u[2]           # daz/dt = jz
        )
        
        # 创建函数对象
        self.f = ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])

    def solve_mpc(self, x0, ref_trajectory):
        opti = ca.Opti()

        N = self.prediction_horizon
        dt = 1.0/self.ctr_freq
        
        # 决策变量
        X = opti.variable(9, N + 1)  # 状态序列
        U = opti.variable(3, N)    # 控制序列
        
        # 初始条件约束
        opti.subject_to(X[:3,0] == x0)

        # 动力学约束
        for k in range(N):
            x_next = X[:,k] + self.f(X[:,k], U[:,k]) * dt
            opti.subject_to(X[:,k+1] == x_next)
        
        # 成本函数
        cost = 0
        for k in range(N):
            # 跟踪误差成本
            error = X[:3,k] - ref_trajectory[:3,k]
            cost += error.T @ self.Q @ error

            # 控制输入成本
            cost += U[:,k].T @ self.R @ U[:,k]
        
        # 终端成本 (加大权重)
        error_terminal = X[:3,-1] - ref_trajectory[:3,-1]
        cost += 100 * error_terminal.T @ error_terminal
        
        opti.minimize(cost)
        
        # 求解器设置
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            return sol.value(U[:,0]), sol.value(X[:,1])
        except:
            print("求解失败，返回零控制")
            return np.zeros(3), None

    def odom_callback(self, msg):
        self.current_odom = msg
        self.recorded_ref_odom.append([
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            msg.pose.pose.position.z,
            (rospy.Time.now() - self.start_time).to_sec()
        ])

    def traj_callback(self, msg : PolynomialTrajectory):
        if msg.piece_num_pos == 0 or msg.piece_num_yaw == 0:
            return
        
        self.current_traj = msg

    def ref_traj(self):
        # 根据当前时间t计算参考轨迹
        if self.current_traj is None:
            return None,None

        t = (rospy.Time.now() - self.current_traj.start_WT_pos).to_sec()
        dt = 1.0/self.ctr_freq
        N = self.prediction_horizon
        D = self.current_traj.order_pos + 1
        ref = np.zeros((3, N))
        
        time_sum = sum(self.current_traj.time_pos)
        if t > time_sum:
            return None,None
        
        for j in range(N):
            remaining_t = t + dt*j
            if remaining_t > time_sum:
                remaining_t = time_sum
     
            # 查找当前时间对应的轨迹片段
            index = 0
            for i, duration in enumerate(self.current_traj.time_pos):
                if remaining_t <= duration:
                    index = i
                    break
                remaining_t -= duration
            
            # 计算当前位置参考点
            tn = 1.0
            for i in range(D-1, -1, -1):
                ref[0, j] += self.current_traj.coef_pos_x[8 * index + i] * tn
                ref[1, j] += self.current_traj.coef_pos_y[8 * index + i] * tn
                ref[2, j] += self.current_traj.coef_pos_z[8 * index + i] * tn
                tn *= remaining_t

        remaining_t = t
        if remaining_t > time_sum:
            remaining_t = time_sum

        index = 0
        for i, duration in enumerate(self.current_traj.time_yaw):
            if remaining_t <= duration:
                index = i
                break
            remaining_t -= duration
        
        # 计算当前位置参考点
        yaw = 0.0
        tn = 1.0
        for i in range(D-1, -1, -1):
            yaw += self.current_traj.coef_yaw[8 * index + i] * tn
            tn *= remaining_t
        
        return ref,yaw

    def timer_callback(self, event):
        if self.current_odom is None or self.current_traj is None:
            return

        # 计算参考轨迹
        ref_path,yaw = self.ref_traj()
        
        if ref_path is None:
            return

        self.recorded_ref_path.append([ref_path[0, 0], ref_path[1, 0], ref_path[2, 0],(rospy.Time.now() - self.start_time).to_sec()])
        self.recorded_ref_yaw.append([yaw, (rospy.Time.now() - self.start_time).to_sec()])

        # 计算控制输入
        x0 = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y,
            self.current_odom.pose.pose.position.z
        ])
        u, pred = self.solve_mpc(x0, ref_path)

        # 发布PositionCommand消息
        if pred is not None:
            cmd = PositionCommand()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "world"
            cmd.position.x = pred[0]
            cmd.position.y = pred[1]
            cmd.position.z = pred[2]
            cmd.velocity.x = pred[3]
            cmd.velocity.y = pred[4]
            cmd.velocity.z = pred[5]
            cmd.acceleration.x = pred[6]
            cmd.acceleration.y = pred[7]
            cmd.acceleration.z = pred[8]
            cmd.yaw = yaw
            self.recorded_cmd_path.append([pred[0], pred[1], pred[2], (rospy.Time.now() - self.start_time).to_sec()])

            self.cmd_pub.publish(cmd)
        else:
            print("nmpc error!")

    def plot_trajectories(self):
        """绘制记录的实际轨迹和参考轨迹"""
        if not self.recorded_ref_odom or not self.recorded_ref_path:
            rospy.loginfo("没有足够的轨迹数据用于绘图")
            return
            
        # 转换为numpy数组
        odom_array = np.array(self.recorded_ref_odom)
        ref_array = np.array(self.recorded_ref_path)
        yaw_array = np.array(self.recorded_ref_yaw)
        cmd_array = np.array(self.recorded_cmd_path)
        
        # 创建图形和子图
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        
        # 绘制X轴位置-时间图
        ax1.plot(odom_array[:, 3], odom_array[:, 0], 'b-', label='实际轨迹')
        ax1.plot(ref_array[:, 3], ref_array[:, 0], 'r--', label='参考轨迹')
        ax1.plot(cmd_array[:, 3], cmd_array[:, 0], 'g-.', label='控制轨迹')
        ax1.set_ylabel('X (m)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('四旋翼轨迹跟踪结果')
        
        # 绘制Y轴位置-时间图
        ax2.plot(odom_array[:, 3], odom_array[:, 1], 'b-', label='实际轨迹')
        ax2.plot(ref_array[:, 3], ref_array[:, 1], 'r--', label='参考轨迹')
        ax2.plot(cmd_array[:, 3], cmd_array[:, 1], 'g-.', label='控制轨迹')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.grid(True)
        
        # 绘制Z轴位置-时间图
        ax3.plot(odom_array[:, 3], odom_array[:, 2], 'b-', label='实际轨迹')
        ax3.plot(ref_array[:, 3], ref_array[:, 2], 'r--', label='参考轨迹')
        ax3.plot(cmd_array[:, 3], cmd_array[:, 2], 'g-.', label='控制轨迹')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('Z (m)')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(yaw_array[:, 1], yaw_array[:, 0], 'r--.', label='参考yaw')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('yaw (m)')
        ax4.legend()
        ax4.grid(True)
        
        # 调整布局
        plt.tight_layout()

        plt.savefig('/home/xzm/super_ws/src/nmpc/fig/figure.png', dpi=300)
        
        # 显示图形
        plt.show()
        

if __name__ == '__main__':
    try:
        node = NMPC()
        rospy.spin()  # 使用spin()保持节点运行，让定时器继续工作
        
        # 节点关闭后绘制轨迹
        node.plot_trajectories()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点运行出错: {str(e)}")
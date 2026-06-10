import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque
from quadrotor_sim.control_drone_main import (
    I_diag,
    L,
    QuadrotorModel,
    k_drag,
    k_roll,
    kf,
    km,
    m,
    omega_max,
)
from quadrotor_sim.so3_lee_controller import LeeSE3Controller
from quadrotor_sim.trajectory_planning import TrajectoryPlanner


class LeeGazeboController(Node):
    def __init__(self, target_pos=np.array([0, 0.0, 3]), target_yaw=0.0):
        super().__init__('lee_gazebo_controller')

        self.control_rate_hz = 50.0
        self.dt = 1.0 / self.control_rate_hz

        self.target_pos = target_pos
        self.target_yaw = target_yaw
        self.zero3 = np.zeros(3)
        self.num_steps = 0

        model = QuadrotorModel(
            m=m,
            L=L,
            I=I_diag,
            kf=kf,
            km=km,
            k_drag=k_drag,
            k_roll=k_roll,
            omega_max=omega_max,
        )
        self.controller = LeeSE3Controller(model, dt=self.dt)

        self.state = None

        self.odom_sub = self.create_subscription(
            Odometry,
            '/crazyflie/odom',
            self.odom_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            ThrustAndTorque,
            '/cf_control/control_command',
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('LeeGazeboController started')

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        # ROS quaternion order is x, y, z, w.
        # Your controller expects w, x, y, z.
        self.state = {
            'pos': np.array([p.x, p.y, p.z], dtype=float),
            'vel': np.array([v.x, v.y, v.z], dtype=float),
            'quat': np.array([q.w, q.x, q.y, q.z], dtype=float),
            'omega': np.array([w.x, w.y, w.z], dtype=float),
        }

        print('pose:', np.array([p.x, p.y, p.z]))

    def get_desired_state(self):
        state_data = {
            'in_pos': self.target_pos,
            'in_vel': self.zero3,
            'in_acc': self.zero3,
            'in_jerk': self.zero3,
            'in_snap': self.zero3,
            'in_yaw': self.target_yaw,
            'in_yaw_rate': 0.0,
            'in_yaw_acceleration': 0.0,
            'mass': m,
            'gravity': 9.81,
            'inertia': np.diag(I_diag),
        }

        return TrajectoryPlanner(filename=None, state_data=state_data).get_flat_outputs(
            test_line=0,
            state_data=state_data,
        )

    def control_loop(self):
        if self.state is None:
            return

        desired_state = self.get_desired_state()
        self.num_steps += 1
        thrust, torque = self.controller.control(
            self.state, desired_state, debug=True, num_steps=self.num_steps
        )

        msg = ThrustAndTorque()
        msg.collective_thrust = float(thrust)
        msg.torque = Vector3(
            x=float(torque[0]),
            y=float(torque[1]),
            z=float(torque[2]),
        )

        self.cmd_pub.publish(msg)

        # print('thrust:', thrust, 'torque:', torque, '\n',
        #       'desired pos:', desired_state['pos'], 'current pos:', self.state['pos'], '\n',
        #       'desired vel:', desired_state['vel'], 'current vel:', self.state['vel'], '\n',
        #       'desired yaw:', desired_state['yaw'], 'current yaw:', self.state['quat'], '\n',
        #       '---')


def main():
    rclpy.init()
    node = LeeGazeboController(target_pos=np.array([1, 0.0, 5]), target_yaw=0.0)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

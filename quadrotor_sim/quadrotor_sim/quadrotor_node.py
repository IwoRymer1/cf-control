import numpy as np
import rclpy
from control_drone_main import QuadrotorModel
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu


class QuadrotorNode(Node):
    def __init__(self, test_hover=False):
        super().__init__('quadrotor_model')

        # --- model ---
        self.model = QuadrotorModel(
            m=0.025 + 4 * 0.0008,
            L=0.0438,
            I=np.diag([16.6e-6, 16.7e-6, 29.3e-6]),
            kf=1.28192e-08,
            km=0.005964552 * 1.28192e-08,
            k_drag=8.06428e-05,
            k_roll=1e-7,
            omega_max=2618,
        )

        # sterowanie
        self.u = np.zeros(4)

        # --- sub i pub tylko jeśli nie test ---
        if not test_hover:
            self.create_subscription(Wrench, '/cmd_wrench', self.cmd_callback, 10)
            self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
            self.imu_pub = self.create_publisher(Imu, '/imu', 10)

            self.dt = 0.002  # 500 Hz
            self.create_timer(self.dt, self.update)
        else:
            # --- test hover ---
            self.dt = 0.001
            m = self.model.m
            self.u = np.array([m * 9.81 * 1, 0, 0, 0])  # hover thrust
            for _ in range(1000):
                self.model.step(self.u, self.dt)
                print(self.model.get_state()['pos'])

        # --- sub ---
        self.create_subscription(Wrench, '/cmd_wrench', self.cmd_callback, 10)

        # --- pub ---
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)

        # --- timer (symulacja) ---
        self.dt = 0.002  # 500 Hz
        self.create_timer(self.dt, self.update)

    def cmd_callback(self, msg):
        self.u[0] = msg.force.z
        self.u[1] = msg.torque.x
        self.u[2] = msg.torque.y
        self.u[3] = msg.torque.z

    def update(self):
        self.model.step(self.u, self.dt)
        self.publish_state()

    def publish_state(self):
        state = self.model.get_state()

        # --- Odometry ---
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'world'

        odom.pose.pose.position.x = state['pos'][0]
        odom.pose.pose.position.y = state['pos'][1]
        odom.pose.pose.position.z = state['pos'][2]

        odom.pose.pose.orientation.w = state['quat'][0]
        odom.pose.pose.orientation.x = state['quat'][1]
        odom.pose.pose.orientation.y = state['quat'][2]
        odom.pose.pose.orientation.z = state['quat'][3]

        odom.twist.twist.linear.x = state['vel'][0]
        odom.twist.twist.linear.y = state['vel'][1]
        odom.twist.twist.linear.z = state['vel'][2]

        odom.twist.twist.angular.x = state['omega'][0]
        odom.twist.twist.angular.y = state['omega'][1]
        odom.twist.twist.angular.z = state['omega'][2]

        self.odom_pub.publish(odom)

        # --- IMU ---
        imu = Imu()
        imu.header = odom.header

        imu.orientation = odom.pose.pose.orientation
        imu.angular_velocity = odom.twist.twist.angular

        self.imu_pub.publish(imu)


def main():
    import sys

    rclpy.init()

    test_hover = False
    if len(sys.argv) > 1 and sys.argv[1] == 'test_hover':
        test_hover = True

    node = QuadrotorNode(test_hover=test_hover)

    if not test_hover:
        rclpy.spin(node)
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

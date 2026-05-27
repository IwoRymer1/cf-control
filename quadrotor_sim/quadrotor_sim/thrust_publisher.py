#!/usr/bin/env python3


# chmod +x ~/ros2_ws/src/quadrotor_sim/quadrotor_sim/thrust_publisher.py
# source /opt/ros/humble/setup.bash
# source ~/ros2_ws/install/setup.bash

# ros2 run cf_control mixer

# another terminal
# source /opt/ros/humble/setup.bash
# source ~/ros2_ws/install/setup.bash

# python3 ~/ros2_ws/src/quadrotor_sim/quadrotor_sim/thrust_publisher.py

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque


class ThrustTest(Node):
    def __init__(self):
        super().__init__('thrust_test')

        self.pub = self.create_publisher(
            ThrustAndTorque,
            '/cf_control/control_command',
            10,
        )

        self.timer = self.create_timer(0.02, self.publish_command)  # 50 Hz

    def publish_command(self):
        msg = ThrustAndTorque()
        msg.collective_thrust = 0.3
        msg.torque = Vector3()
        msg.torque.x = 0.0
        msg.torque.y = 0.0
        msg.torque.z = 0.0

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ThrustTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

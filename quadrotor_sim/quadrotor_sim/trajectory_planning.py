import numpy as np

import pandas as pd

filename = 'trajectory_from_flat_output_test_data.csv'

def load_csv(filename):
    df = pd.read_csv(filename)
    
    #print(df.columns)
    # test_name	in_pos_x	in_pos_y	in_pos_z	in_vel_x	in_vel_y	in_vel_z	in_acc_x	in_acc_y	in_acc_z	in_jerk_x	in_jerk_y	in_jerk_z	in_snap_x	in_snap_y	in_snap_z	in_yaw	in_yaw_rate	in_yaw_acceleration	in_mass	in_gravity	in_I_xx	in_I_yy	in_I_zz	out_pos_x	out_pos_y	out_pos_z	out_quat_w	out_quat_x	out_quat_y	out_quat_z	out_vel_x	out_vel_y	out_vel_z	out_omega_x	out_omega_y	out_omega_z	out_thrust	out_torque_x	out_torque_y	out_torque_z
    #print(df.head())
    return df

def read_trajectory_data_print(filename, test_line):
    # first test in second line
    line_in_file = test_line
    df = load_csv(filename)
    row = df.iloc[line_in_file]
    in_pos = np.array([row['in_pos_x'], row['in_pos_y'], row['in_pos_z']])
    in_vel = np.array([row['in_vel_x'], row['in_vel_y'], row['in_vel_z']])
    in_acc = np.array([row['in_acc_x'], row['in_acc_y'], row['in_acc_z']])
    in_jerk = np.array([row['in_jerk_x'], row['in_jerk_y'], row['in_jerk_z']])
    in_snap = np.array([row['in_snap_x'], row['in_snap_y'], row['in_snap_z']])
    in_yaw = row['in_yaw']
    in_yaw_rate = row['in_yaw_rate']
    in_yaw_acceleration = row['in_yaw_acceleration']
    mass = row['in_mass']
    gravity = row['in_gravity']
    I = np.array([row['in_I_xx'], row['in_I_yy'], row['in_I_zz']])
    out_pos = np.array([row['out_pos_x'], row['out_pos_y'], row['out_pos_z']])
    out_quat = np.array([row['out_quat_w'], row['out_quat_x'], row['out_quat_y'], row['out_quat_z']])
    out_vel = np.array([row['out_vel_x'], row['out_vel_y'], row['out_vel_z']])
    out_omega = np.array([row['out_omega_x'], row['out_omega_y'], row['out_omega_z']])
    out_thrust = row['out_thrust']
    out_torque = np.array([row['out_torque_x'], row['out_torque_y'], row['out_torque_z']])

    print("Input trajectory data:")
    print("Position:", in_pos)
    print("Velocity:", in_vel)
    print("Acceleration:", in_acc)
    print("Jerk:", in_jerk)
    print("Snap:", in_snap)
    print("Yaw:", in_yaw)
    print("Yaw rate:", in_yaw_rate)
    print("Yaw acceleration:", in_yaw_acceleration)
    print("Mass:", mass)
    print("Gravity:", gravity)
    print("Inertia:", I)
    
    print("\nExpected output data:")
    print("Position:", out_pos)
    print("Quaternion:", out_quat)
    print("Velocity:", out_vel)
    print("Angular velocity:", out_omega)
    print("Thrust:", out_thrust)
    print("Torque:", out_torque)


class TrajectoryPlanner:
    def __init__(self, filename):
        self.filename = filename
        self.trajectory_data = load_csv(filename)
        self.in_pos = self.trajectory_data[['in_pos_x', 'in_pos_y', 'in_pos_z']].values
        self.in_vel = self.trajectory_data[['in_vel_x', 'in_vel_y', 'in_vel_z']].values
        self.in_acc = self.trajectory_data[['in_acc_x', 'in_acc_y', 'in_acc_z']].values
        self.in_jerk = self.trajectory_data[['in_jerk_x', 'in_jerk_y', 'in_jerk_z']].values
        self.in_snap = self.trajectory_data[['in_snap_x', 'in_snap_y', 'in_snap_z']].values
        self.in_yaw = self.trajectory_data['in_yaw'].values
        self.in_yaw_rate = self.trajectory_data['in_yaw_rate'].values
        self.in_yaw_acceleration = self.trajectory_data['in_yaw_acceleration'].values
        self.mass = self.trajectory_data['in_mass'].values
        self.gravity = self.trajectory_data['in_gravity'].values
        self.inertia = self.trajectory_data[['in_I_xx', 'in_I_yy', 'in_I_zz']].values
        self.out_pos = self.trajectory_data[['out_pos_x', 'out_pos_y', 'out_pos_z']].values
        self.out_quat = self.trajectory_data[['out_quat_w', 'out_quat_x', 'out_quat_y', 'out_quat_z']].values
        self.out_vel = self.trajectory_data[['out_vel_x', 'out_vel_y', 'out_vel_z']].values
        self.out_omega = self.trajectory_data[['out_omega_x', 'out_omega_y', 'out_omega_z']].values
        self.out_thrust = self.trajectory_data['out_thrust'].values
        self.out_torque = self.trajectory_data[['out_torque_x', 'out_torque_y', 'out_torque_z']].values
    
    def rot_to_quat(self, R):
        qw = np.sqrt(1 + np.trace(R)) / 2
        qx = (R[2,1] - R[1,2]) / (4*qw)
        qy = (R[0,2] - R[2,0]) / (4*qw)
        qz = (R[1,0] - R[0,1]) / (4*qw)
        return np.array([qw, qx, qy, qz])

    def get_flat_outputs(self, test_line):
        in_pos = self.in_pos[test_line]
        in_vel = self.in_vel[test_line]
        in_acc = self.in_acc[test_line]
        in_jerk = self.in_jerk[test_line]
        in_snap = self.in_snap[test_line]
        in_yaw = self.in_yaw[test_line]
        in_yaw_rate = self.in_yaw_rate[test_line]
        in_yaw_acceleration = self.in_yaw_acceleration[test_line]
        mass = self.mass[test_line]
        gravity = self.gravity[test_line]
        inertia = self.inertia[test_line]

        #as arrays
        in_jerk = np.array(in_jerk)
        in_snap = np.array(in_snap)

        # output calculation
        out_pos = np.asarray(in_pos)
        out_vel = np.asarray(in_vel)

        # output quaterion
        Tzb = in_acc + np.array([0, 0, gravity])
        out_thrust = mass * np.linalg.norm(Tzb)

        zb = Tzb / np.linalg.norm(Tzb)
        out_yaw = in_yaw
        
        # desired heading direction
        xc = np.array([np.cos(out_yaw), np.sin(out_yaw), 0])

        #based on that we place the body frame
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        R = np.column_stack((xb, yb, zb))
        out_quat = self.rot_to_quat(R)

        # desired angular velocity
        


if __name__ == "__main__":
    read_trajectory_data_print(filename, test_line=0)
    pass


import numpy as np
import trajectory_planning as traj_plan

import control_drone_main as QuadrotorModel

m_body = 0.025
m_prop = 0.0008
m = m_body + 4 * m_prop

I_body = np.array(
    [
        [16.571710e-6, 0.830806e-6, 0.718277e-6],
        [0.830806e-6, 16.655602e-6, 1.800197e-6],
        [0.718277e-6, 1.800197e-6, 29.261652e-6],
    ]
)

I_diag = np.diag([16.6e-6, 16.7e-6, 29.3e-6])

L = 0.0438

kf = 1.28192e-08
km = 0.005964552 * kf
k_drag = 8.06428e-05
k_roll = 1e-7
omega_max = 2618


class LeeSE3Controller:
    def __init__(self, model, dt=0.01):
        self.m = model.m
        self.g = model.g
        self.J = model.I
        self.dt = dt

        # tunable params
        self.kx = np.diag([6, 6, 8])
        self.kv = np.diag([4, 4, 5])

        self.kR = np.diag([2, 2, 0.5])
        self.kOmega = np.diag([0.05, 0.05, 0.05])
        self.k_psi = 3.0

        self.Rd_prev = np.eye(3)

        self.M_max = 1e-3

    def hat(self, x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    def vee(self, M):
        return np.array([M[2, 1], M[0, 2], M[1, 0]])

    def quat_to_rot(self, q):
        w, x, y, z = q
        R = np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )
        return R

    def control(self, state, desired_state):
        x = state['pos']
        v = state['vel']
        q = state['quat']
        omega = state['omega']

        R = self.quat_to_rot(q)

        xd = desired_state['pos']
        vd = desired_state['vel']
        ad = desired_state['acc']

        # optional yaw input (default = 0)
        yaw = desired_state.get('yaw', 0.0)
        yaw_rate = desired_state.get('yaw_rate', 0.0)

        e3 = np.array([0.0, 0.0, 1.0])

        # --- position control ---
        ex = x - xd
        ev = v - vd

        a_cmd = -self.kx @ ex - self.kv @ ev + ad + self.g * e3
        F_des = self.m * a_cmd

        if np.linalg.norm(F_des) < 1e-6:
            b3d = np.array([0.0, 0.0, 1.0])
        else:
            b3d = F_des / np.linalg.norm(F_des)

        f = np.dot(F_des, R @ e3)
        f = max(0.0, f)

        b1c = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        b1c = b1c - np.dot(b1c, b3d) * b3d
        b1c = b1c / np.linalg.norm(b1c)

        b2d = np.cross(b3d, b1c)
        norm_b2d = np.linalg.norm(b2d)

        if norm_b2d < 1e-6:
            b2d = np.array([0.0, 1.0, 0.0])
        else:
            b2d = b2d / norm_b2d

        b1d = np.cross(b2d, b3d)
        b1d /= np.linalg.norm(b1d)

        Rd = np.column_stack((b1d, b2d, b3d))

        psi_d = np.arctan2(Rd[1, 0], Rd[0, 0])
        psi = np.arctan2(R[1, 0], R[0, 0])

        yaw_error = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))

        # =========================================================
        # ANALYTIC DESIRED ANGULAR VELOCITY
        # =========================================================

        Omega_d_world = np.array([0.0, 0.0, yaw_rate])
        Omegad = Rd.T @ Omega_d_world

        # yaw-only motion → no angular acceleration
        Omegad_dot = np.zeros(3)

        # =========================================================
        # ATTITUDE ERRORS
        # =========================================================

        eR = 0.5 * self.vee(Rd.T @ R - R.T @ Rd)

        # yaw_correction = np.array([0.0, 0.0, self.k_psi * yaw_error])

        eOmega = omega - R.T @ Rd @ Omegad

        # =========================================================
        # CONTROL MOMENT
        # =========================================================

        M = (
            -self.kR @ eR
            - self.kOmega @ eOmega
            + np.cross(omega, self.J @ omega)
            - self.J @ (self.hat(omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ Omegad_dot)
        )

        M = np.clip(M, -self.M_max, self.M_max)
        return f, M


def test_controller(test_line):
    desired_state = planner.get_flat_outputs(test_line)
    state = {
        'pos': desired_state['pos'],
        'vel': desired_state['vel'],
        'quat': desired_state['quat'],
        'omega': desired_state['omega'],
    }
    f, M = controller.control(state, desired_state)
    return f, M


def test_all_lines(filename):

    file = traj_plan.load_csv(filename)
    num_tests = len(file)
    results = []
    for test_line in range(num_tests):
        f, M = test_controller(test_line)
        results.append((f, M))
    return results


def run_simulation(test_line=None, dt=0.01, total_time=5.0):

    if test_line is None:
        desired_state = {
            'pos': np.array([1, 1, 0]),
            'vel': np.array([0, 0, 0]),
            'omega': np.array([0, 0, 0]),
            'R': np.eye(3),
            'omega_dot': np.zeros(3),
            'acc': np.array([0, 0, 0]),
            'quat': np.array([1, 0, 0, 0]),
        }

        state = {
            'pos': np.zeros(3),
            'vel': np.zeros(3),
            'omega': np.zeros(3),
            'quat': np.array([1, 0, 0, 0]),
        }
    else:
        desired_state = planner.get_flat_outputs(test_line)
        state = {
            'pos': desired_state['pos'],
            'vel': desired_state['vel'],
            'quat': desired_state['quat'],
            'omega': desired_state['omega'],
        }

    num_steps = int(total_time / dt)
    for step in range(num_steps):
        f, M = controller.control(state, desired_state)
        quad.set_state(**state)
        u = np.asarray([f, M[0], M[1], M[2]])
        quad.step(u, dt)
        state = quad.get_state()
        print(
            f'Step {step}: pos={state["pos"]}, vel={state["vel"]}, quat={state["quat"]}, omega={state["omega"]}'
        )


def test_takeoff():
    desired_state = {
        'pos': np.array([0, 0, 0]),
        'vel': np.zeros(3),
        'omega': np.zeros(3),
        'R': np.eye(3),
        'omega_dot': np.zeros(3),
        'acc': np.zeros(3),
        'quat': np.array([1, 0, 0, 0]),
    }
    state = {
        'pos': np.zeros(3),
        'vel': np.zeros(3),
        'omega': np.zeros(3),
    }
    f, M = controller.control(state, desired_state)
    print(f'Takeoff control output: f={f}, M={M}')


def smooth_yaw_test(dt=0.01, total_time=3.0, initial_yaw=0.0, final_yaw=np.pi / 4, yaw_rate=0.1):
    desired_state = {
        'pos': np.array([0, 0, 0]),
        'vel': np.zeros(3),
        'omega': np.zeros(3),
        'R': np.eye(3),
        'omega_dot': np.zeros(3),
        'acc': np.zeros(3),
        'yaw': final_yaw,
        'yaw_rate': yaw_rate,
        'quat': np.array([1, 0, 0, 0]),
    }
    state = {
        'pos': np.zeros(3),
        'vel': np.zeros(3),
        'omega': np.zeros(3),
        'quat': np.array([1, 0, 0, 0]),
    }
    f, M = controller.control(state, desired_state)
    print(f'Smooth yaw test control output: f={f}, M={M}')

    num_steps = int(total_time / dt)
    reached_final = False
    print_every_n = 10
    for step in range(num_steps):
        f, M = controller.control(state, desired_state)
        quad.set_state(**state)
        u = np.asarray([f, M[0], M[1], M[2]])
        quad.step(u, dt)
        state = quad.get_state()

        Rotation_matrix = controller.quat_to_rot(state['quat'])
        yaw = np.arctan2(Rotation_matrix[1, 0], Rotation_matrix[0, 0])
        yaw_deg = np.degrees(yaw)
        if step % print_every_n == 0:
            print(
                f'Step {step}: pos={state["pos"]}, vel={state["vel"]}, quat={state["quat"]}, omega={state["omega"]}'
            )
            print(f'Current yaw: {yaw_deg:.2f} degrees')
            yaw_error = desired_state['yaw'] - yaw
            print(f'Yaw error: {np.degrees(yaw_error):.2f}')
            print(f'control values: f= {f} , M = {M}')


if __name__ == '__main__':
    filename = 'trajectory_from_flat_output_test_data.csv'

    planner = traj_plan.TrajectoryPlanner(filename)

    quad = QuadrotorModel.QuadrotorModel(
        m=m, L=L, I=I_diag, kf=kf, km=km, k_drag=k_drag, k_roll=k_roll, omega_max=omega_max
    )
    dt = 0.01
    controller = LeeSE3Controller(quad, dt)
    #smooth_yaw_test(dt=dt, total_time=2.0, initial_yaw=0.0, final_yaw=-np.pi / 3, yaw_rate=0)
    run_simulation()

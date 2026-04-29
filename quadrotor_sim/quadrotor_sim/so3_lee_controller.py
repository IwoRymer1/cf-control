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
    def __init__(self, model):
        self.m = model.m
        self.g = model.g
        self.J = model.I

        # tunable params
        self.kx = np.diag([6, 6, 8])
        self.kv = np.diag([4, 4, 5])

        self.kR = np.diag([8, 8, 1])
        self.kOmega = np.diag([0.15, 0.15, 0.1])

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

        Rd = desired_state['R']
        Omegad = desired_state['omega']
        Omegad_dot = desired_state['omega_dot']

        e3 = np.array([0, 0, 1])

        # translational errors
        ex = x - xd
        ev = v - vd

        A = -self.kx @ ex - self.kv @ ev - self.m * self.g * e3 + self.m * ad

        f = A @ (R.T @ e3)

        # attitude errors
        eR = 0.5 * self.vee(Rd.T @ R - R.T @ Rd)
        eOmega = omega - R.T @ Rd @ Omegad

        # control moment
        M = (
            -self.kR @ eR
            - self.kOmega @ eOmega
            + np.cross(omega, self.J @ omega)
            - self.J @ (self.hat(omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ Omegad_dot)
        )

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
            'pos': np.array([0, 0, 1]),
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
        'pos': np.array([0, 0, -1]),
        'vel': np.zeros(3),
        'omega': np.zeros(3),
        'R': np.eye(3),
        'omega_dot': np.zeros(3),
        'acc': np.zeros(3),
    }
    state = {
        'pos': np.zeros(3),
        'vel': np.zeros(3),
        'omega': np.zeros(3),
    }
    f, M = controller.control(state, desired_state)
    print(f'Takeoff control output: f={f}, M={M}')


if __name__ == '__main__':
    filename = 'trajectory_from_flat_output_test_data.csv'

    filename = 'trajectory_from_flat_output_test_data.csv'
    planner = traj_plan.TrajectoryPlanner(filename)

    quad = QuadrotorModel.QuadrotorModel(
        m=m, L=L, I=I_diag, kf=kf, km=km, k_drag=k_drag, k_roll=k_roll, omega_max=omega_max
    )

    controller = LeeSE3Controller(quad)
    run_simulation()

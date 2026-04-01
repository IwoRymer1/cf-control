import numpy as np

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


class QuadrotorModel:
    def __init__(self, m, L, I, kf, km, k_drag, k_roll, omega_max):

        self.m = m
        self.g = 9.81
        self.L = L
        self.I = I
        self.invI = np.linalg.inv(I)

        self.kf = kf
        self.km = km
        self.k_drag = k_drag
        self.k_roll = k_roll
        self.omega_max = omega_max

        # --- stan ---
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)

        # kwaternion [w, x, y, z]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.omega = np.zeros(3)

    def quat_mul(self, q, p):
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = p
        return np.array(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ]
        )

    def normalize_quat(self, q):
        return q / np.linalg.norm(q)

    def quat_to_rot(self, q):
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def dynamics(self, u):
        """
        u = [T, tau_x, tau_y, tau_z]
        """

        T = u[0]
        tau = u[1:4]

        # --- translacja ---
        R = self.quat_to_rot(self.q)
        thrust_body = np.array([0, 0, T])
        thrust_world = R @ thrust_body

        pos_dot = self.vel
        vel_dot = np.array([0, 0, -self.g]) + thrust_world / self.m
        vel_dot -= self.k_drag * self.vel  # opór powietrza

        # --- rotacja (kwaternion) ---
        omega_quat = np.array([0, *self.omega])
        q_dot = 0.5 * self.quat_mul(self.q, omega_quat)

        # --- dynamika obrotowa ---
        omega_dot = self.invI @ (tau - np.cross(self.omega, self.I @ self.omega))
        omega_dot -= self.k_roll * self.omega  # tłumienie obrotów

        return pos_dot, vel_dot, q_dot, omega_dot

    def step(self, u, dt):
        pos_dot, vel_dot, q_dot, omega_dot = self.dynamics(u)

        # Euler (można podmienić na RK4)
        # self.pos += dt * pos_dot
        # self.vel += dt * vel_dot
        # self.q += dt * q_dot
        # self.omega += dt * omega_dot

        # RK 4
        k1_pos, k1_vel, k1_q, k1_omega = self.dynamics(u)
        k2_pos, k2_vel, k2_q, k2_omega = self.dynamics(u)
        k3_pos, k3_vel, k3_q, k3_omega = self.dynamics(u)
        k4_pos, k4_vel, k4_q, k4_omega = self.dynamics(u)
        self.pos += (dt / 6) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
        self.vel += (dt / 6) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
        self.q += (dt / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        self.omega += (dt / 6) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

        # normalizacja kwaternionu
        self.q = self.normalize_quat(self.q)

    def get_state(self):
        return {
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            'quat': self.q.copy(),
            'omega': self.omega.copy(),
        }

    def set_state(self, pos, vel, quat, omega):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.q = self.normalize_quat(np.array(quat))
        self.omega = np.array(omega)


quad = QuadrotorModel(
    m=m, L=L, I=I_diag, kf=kf, km=km, k_drag=k_drag, k_roll=k_roll, omega_max=omega_max
)

# hover (powinno wisieć)
u = np.array([m * 9.81 * 1.0, 0.5, 0, 0])

for i in range(1000):
    quad.step(u, dt=0.001)
    if i % 100 == 0:
        print(quad.get_state()['pos'], quad.get_state()['vel'], quad.get_state()['quat'])


# task:
# obtain state from flat output

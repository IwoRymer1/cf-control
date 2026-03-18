import numpy as np

m_body = 0.025
m_prop = 0.0008
m = m_body + 4 * m_prop

I_body = np.array([
    [16.571710e-6,  0.830806e-6,  0.718277e-6],
    [0.830806e-6,  16.655602e-6,  1.800197e-6],
    [0.718277e-6,  1.800197e-6, 29.261652e-6]
])

I_diag = np.diag([16.6e-6, 16.7e-6, 29.3e-6])

L = 0.0438

kf = 1.28192e-08
km = 0.005964552 * kf
k_drag = 8.06428e-05
k_roll = 1e-7
omega_max = 2618

class UAVModel():
    def __init__(self, m=1.0, L=0.2, I=np.diag([0.01, 0.01, 0.02]), 
                 kf=1e-5, km = 1e-6, k_drag=8.06428e-05, k_roll=1e-7,
                 omega_max = 2618):
        self.m = m
        self.g = 9.81
        self.L = L
        self.I = I
        self.invI = np.linalg.inv(self.I)

        self.kf = kf
        self.km = km

        self.k_drag = k_drag
        self.k_roll = k_roll

        self.omega_max = omega_max

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.angles = np.zeros(3)
        self.omega = np.zeros(3)

        self.tau_z = (1, -1, 1, -1)



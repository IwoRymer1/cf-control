import control_drone_main as model
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


def test_1():
    quad = model.QuadrotorModel(
        m=m, L=L, I=I_diag, kf=kf, km=km, k_drag=k_drag, k_roll=k_roll, omega_max=omega_max
    )

    u = np.array([m * 9.81 * 1.0, 0, 0, 0])

    for i in range(1000):
        quad.step(u, dt=0.001)
        if i % 100 == 0:
            print(quad.get_state()['pos'], quad.get_state()['vel'], quad.get_state()['quat'])
    quad_1_out = quad.get_state()['pos']
    assert np.all(quad_1_out == [0, 0, 0])


def test_2():
    quad = model.QuadrotorModel(
        m=m, L=L, I=I_diag, kf=kf, km=km, k_drag=k_drag, k_roll=k_roll, omega_max=omega_max
    )

    u = np.array([m * 9.81 * 1.1, 0, 0, 0])

    for i in range(1000):
        quad.step(u, dt=0.001)
        if i % 100 == 0:
            print(quad.get_state()['pos'], quad.get_state()['vel'], quad.get_state()['quat'])
    quad_1_out = quad.get_state()['pos']
    assert quad_1_out[2] > 0


def test_3():
    quad = model.QuadrotorModel(
        m=m, L=L, I=I_diag, kf=kf, km=km, k_drag=k_drag, k_roll=k_roll, omega_max=omega_max
    )

    u = np.array([m * 9.81 * 0.5, 0, 0, 0])

    for i in range(1000):
        quad.step(u, dt=0.001)
        if i % 100 == 0:
            print(quad.get_state()['pos'], quad.get_state()['vel'], quad.get_state()['quat'])
    quad_1_out = quad.get_state()['pos']
    assert quad_1_out[2] < 0


if __name__ == '__main__':
    test_1()
    test_2()
    test_3()

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def kinematics(omega, phi, magnetic_dipole, magnetic_field, inertia, inertiaInvert):
    torque_magnetic = np.cross(phi * magnetic_dipole, magnetic_field)
    derivation_term = np.cross(omega, np.matmul(inertia, omega))

    return np.matmul(inertiaInvert, torque_magnetic - derivation_term)


if __name__ == '__main__':
    omega_0 = np.array([0.0, 15.0 * np.pi / 180.0, 0.0])  # in rad/s
    a = 10.0  # cm
    rho = 2000.0  # in kg/m^3
    h = 0.1  # in s
    magnetic_dipole = 2.57  # in A/m^2
    earth_magnetic_field = np.array([0.0, 0.0, 40e-6])
    initial_attitude = np.array([0.0, 0.0, 1.0])

    mass = a * a * a * (rho / 1_000_000)
    inertia = 1.0 / 6.0 * mass * a * a
    inertiaMatrix = np.array([[inertia, 0.0, 0.0], [0.0, inertia, 0.0], [0.0, 0.0, inertia]])
    inertiaMatrixInvert = np.array([[1.0 / inertia, 0.0, 0.0], [0.0, 1.0 / inertia, 0.0], [0.0, 0.0, 1.0 / inertia]])

    omega_timeseries = [omega_0]
    phi_timeseries = [np.array([0.0, 0.0, 0.0])]
    timestamps = [0.0]

    for i in tqdm(range(10000000)):
        rotation = R.from_euler('zyx', phi_timeseries[i])
        rotation_matrix = rotation.as_matrix()
        attitude = np.matmul(rotation_matrix, initial_attitude)
        next_omega = omega_timeseries[i] + h * kinematics(omega_timeseries[i], attitude, magnetic_dipole,
                                                          earth_magnetic_field, inertiaMatrix, inertiaMatrixInvert)
        omega_timeseries.append(next_omega)
        phi_timeseries.append(phi_timeseries[i] + omega_timeseries[i] * h)
        timestamps.append((i + 1) * h)

    omega_x = []
    omega_y = []
    omega_z = []
    for omega_value in omega_timeseries:
        omega_x.append(omega_value[0])
        omega_y.append(omega_value[1])
        omega_z.append(omega_value[2])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(timestamps, omega_x, label="x")
    ax.plot(timestamps, omega_y, label="y")
    ax.plot(timestamps, omega_z, label="z")
    ax.legend()
    plt.show()
    print(omega_timeseries)

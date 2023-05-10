import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import h5py


def kinematics(omega, magnetic_dipole, magnetic_field, inertia, inertiaInvert, base, m_hyst_x, m_hyst_y):
    """

    :param omega: Rotation rate in earth centric coordinates. rad/s
    :param phi: The attitude vector (i.e. the z-axis in the body centric coordinate system.)
    :param magnetic_dipole: Magnetic moment of the permanent magnet.
    :param magnetic_field: Earths magnetic field in earth centric coordinates.
    :param inertia: Inertia matrix.
    :param inertiaInvert: Inverse of the inertia matrix
    :param base: The base of the body centric system in earth centric coordinates
    :return: Change of the rotation rate in earth centric coordinates. rad/s
    """

    torque_magnetic = np.cross(np.array([0.0, 0.0, 1.0]) * magnetic_dipole, magnetic_field)
    torque_hysteresis_x = np.cross(np.array([1.0, 0.0, 0.0]) * m_hyst_x, magnetic_field)
    torque_hysteresis_y = np.cross(np.array([0.0, 1.0, 0.0]) * m_hyst_y, magnetic_field)
    # derivation_term = np.cross(omega, np.matmul(inertia, omega))

    # return np.matmul(inertiaInvert, torque_hysteresis_x + torque_hysteresis_y + torque_magnetic - derivation_term)
    return np.matmul(inertiaInvert, torque_hysteresis_x + torque_hysteresis_y + torque_magnetic)


def hysteresis_loop_x(p, Bs, H, Hc):
    dH = H - hysteresis_loop_x.H_prev

    if dH > 0.0:
        Hc_actual = -1.0 * Hc
    else:
        Hc_actual = Hc
    hysteresis_loop_x.H_prev = H

    return (2.0 / np.pi * Bs * np.arctan(p * H + Hc_actual))


hysteresis_loop_x.H_prev = 1.0


def hysteresis_loop_y(p, Bs, H, Hc):
    dH = H - hysteresis_loop_y.H_prev

    if dH > 0.0:
        Hc_actual = -1.0 * Hc
    else:
        Hc_actual = Hc
    hysteresis_loop_y.H_prev = H

    return (2.0 / np.pi * Bs * np.arctan(p * H + Hc_actual))


hysteresis_loop_y.H_prev = 1.0


def build_rotation_matrix(theta):
    return np.array(
        [[cos(theta[1]) * cos(theta[0]), cos(theta[1]) * sin(theta[0]), - 1.0 * sin(theta[1])],
         [sin(theta[2]) * sin(theta[1]) * cos(theta[0]) - cos(theta[2]) * sin(theta[0]),
          sin(theta[2]) * sin(theta[1]) * sin(theta[0]) + cos(theta[2]) * cos(theta[0]), sin(theta[2]) * cos(theta[1])],
         [cos(theta[2]) * sin(theta[1]) * cos(theta[0]) + sin(theta[2]) * sin(theta[0]),
          cos(theta[2]) * sin(theta[1]) * sin(theta[0]) - sin(theta[2]) * cos(theta[0]),
          cos(theta[2]) * cos(theta[1])]])


def build_rotation_matrix_derivative(theta):
    return np.array(
        [[0.0, sin(theta[2]), cos(theta[2])],
         [0.0, cos(theta[2]) * cos(theta[1]), -1.0 * sin(theta[2]) * cos(theta[1])],
         [cos(theta[1]), sin(theta[2]) * sin(theta[1]), cos(theta[2]) * sin(theta[1])]]) * 1.0 / cos(theta[1])


def hysteresis_rod_moment(B_hyst, V, mu_0):
    return (B_hyst * V) / mu_0


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------- Simulation parameters --
    # The initial turn rate of the satellite in earth centric coordinates.
    n_0 = np.array([10.0, 15.0, 10.0])  # in °/s
    omega_0 = n_0 * np.pi / 180.0  # in rad/s

    # Edge length of the satellite.
    a = 0.1  # m

    # Density of the satellite
    rho = 2000.0  # in kg/m^3

    # Timestep of the simulation loop.
    h = 0.001  # in seconds

    # Duration that shall be simulated.
    duration = 24  # in hours

    # Magnetic dipole moment of the magnetic rod inside the cubesat.
    magnetic_dipole = 2.57  # in A/m^2

    # Magnetic field of the earth in LEO.
    earth_magnetic_field = np.array([0.0, 0.0, 40e-6])  # in T

    # The orthonormal base of the body centric coordinate system.
    initial_base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Coercivity of the hysteresis rod.
    Hc = 0.96  # A/m

    # Remancence of the hysteresis rod.
    Br = 0.35  # T

    # Saturation flux of the hysteresis rod.
    Bs = 0.74  # T

    # Length of the hysteresis rod.
    l_hyst = 0.09  # Meter

    # Diameter of the hysteresis rod.
    d_hyst = 0.001  # Meter

    # Data is written in chunks to disk. This is the chunk size.
    cache_size = 1000

    # ------------------------------------------------------------------------------------------ Precalculated values --
    # Mass of the satellite (assuming it is a perfectly and solid cube).
    mass = a * a * a * rho

    # Inertia matrices of the satellite.
    inertia = 1.0 / 6.0 * mass * a * a
    inertiaMatrix = np.array([[inertia, 0.0, 0.0], [0.0, inertia, 0.0], [0.0, 0.0, inertia]])
    inertiaMatrixInvert = np.array([[1.0 / inertia, 0.0, 0.0], [0.0, 1.0 / inertia, 0.0], [0.0, 0.0, 1.0 / inertia]])

    # Factor that is used in the calculation of the flux through the hysteresis rod.
    p = 1.0 / Hc * np.tan((np.pi * Br) / (2 * Bs))

    # Volume of the hysteresis rod.
    V_hyst = d_hyst * d_hyst * np.pi / 4.0 * l_hyst

    # Permeability of the vacuum.
    mu_0 = 4 * np.pi * 1e-7

    # Count of simulation iterations.
    iterations = int(duration * 60 * 60 / h)

    # ----------------------------------------------------------------------------------------------- Simulation loop --

    with h5py.File('C:\Temp\data.hdf5', 'w') as f:
        omega = f.create_dataset('omega', (iterations + 1, 3), dtype='float32', chunks=(1000, 3), compression='gzip')
        theta = f.create_dataset('theta', (iterations + 1, 3), dtype='float32', chunks=(1000, 3), compression='gzip')
        timestamps = f.create_dataset('timestamps', (iterations + 1,), dtype='float32', compression='gzip')

        omega[0] = omega_0
        omega_current = omega[0]
        omega_next = np.array([0.0, 0.0, 0.0])
        omega_cache = np.zeros(shape=(cache_size, 3), dtype=float)
        theta[0] = np.array([0.0, 0.0, 0.0])
        theta_current = theta[0]
        theta_next = np.array([0.0, 0.0, 0.0])
        theta_cache = np.zeros(shape=(cache_size, 3), dtype=float)
        timestamps[0] = 0.0
        timestamps_cache = np.zeros(shape=(cache_size,), dtype=float)
        cache_counter = 1
        cache_idx = 0
        for i in tqdm(range(iterations)):
            # Rotate the body centric system.

            rotation_matrix = build_rotation_matrix(theta_current)
            inverse_rotation_matrix = rotation_matrix.transpose()
            base = np.matmul(inverse_rotation_matrix, initial_base.transpose()).transpose()

            # Calculate the dipole moment of the hysteresis rods ...
            # ... Calculate the field in the direction of the rods ...
            H_x = np.dot(base[0], earth_magnetic_field / mu_0)
            H_y = np.dot(base[1], earth_magnetic_field / mu_0)
            # ... Calculate the flux induced in the hysteresis rods ...
            B_hyst_x = hysteresis_loop_x(p, Bs, H_x, Hc)
            B_hyst_y = hysteresis_loop_y(p, Bs, H_y, Hc)
            # ... Get the dipole moment from the hysteresis flux.
            m_hyst_x = hysteresis_rod_moment(B_hyst_x, V_hyst, mu_0)
            m_hyst_y = hysteresis_rod_moment(B_hyst_y, V_hyst, mu_0)

            # Calculate the next omega.
            omega_next = omega_current + h * kinematics(omega_current, magnetic_dipole,
                                                        np.matmul(rotation_matrix, earth_magnetic_field),
                                                        inertiaMatrix, inertiaMatrixInvert, base, m_hyst_x,
                                                        m_hyst_y)
            omega_cache[cache_counter] = omega_next

            # Calculate the next theta.
            theta_next = (theta_current + np.matmul(build_rotation_matrix_derivative(theta_current),
                                                    omega_current) * h) % (2.0 * np.pi)
            theta_cache[cache_counter] = theta_next

            # Calculate the timestamp.
            timestamps_cache[cache_counter] = (i + 1) * h

            # Check if the cache has to be written to disk.
            cache_counter += 1
            if cache_counter > (cache_size - 1):
                # Store the results
                omega[cache_idx * cache_size: (cache_idx + 1) * cache_size, :] = omega_cache
                theta[cache_idx * cache_size: (cache_idx + 1) * cache_size, :] = theta_cache
                timestamps[cache_idx * cache_size: (cache_idx + 1) * cache_size] = timestamps_cache

                cache_counter = 0
                cache_idx += 1

            theta_current = theta_next
            omega_current = omega_next

        # Write the data of the last chunk to disk.
        omega[cache_idx * cache_size: cache_idx * cache_size + cache_counter, :] = omega_cache[0:cache_counter, :]
        theta[cache_idx * cache_size: cache_idx * cache_size + cache_counter, :] = theta_cache[0:cache_counter, :]
        timestamps[cache_idx * cache_size: cache_idx * cache_size + cache_counter] = timestamps_cache[0:cache_counter]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(timestamps[:], omega[:, 0], label="w_x", color='r')
        ax.plot(timestamps[:], omega[:, 1], label="w_y", color='g')
        ax.plot(timestamps[:], omega[:, 2], label="w_z", color='b')
        ax.legend()
        plt.show()

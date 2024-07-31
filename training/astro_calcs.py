# Sarah Wielgosz
# MLDS Onboarding

import numpy as np
import math
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# Specific Gravitation Constants
MU_EARTH = 3.986004418 * 10 ** 14
G = 6.67408 * 10 ** -11 # m^3 kg^-1 s^-2
RADIUS_EARTH = 6378000 # m

# Natural Basis
DIR_X = np.array([1, 0, 0])
DIR_Y = np.array([0, 1, 0])
DIR_Z = np.array([0, 0, 1])

def cartesian_to_standard(r, v, mu=MU_EARTH):
    """Converts Cartian orbital elements to standard orbital elements.

    Args:
        r (ndarray): Position vector (in meters)
        v (ndarray): Velocity vector (in meters)
        mu (float): Specific gravity of celestial body (in m^3 s^-2)

    Returns:
        A list containing the orbital elements: [a, e, i, omega, w, u]
    """
    r = r.astype('float64')
    v = v.astype('float64')

    pos = np.linalg.norm(r)
    vel = np.linalg.norm(v)

    _momentum = np.cross(r, v)
    _line_of_nodes = np.cross(DIR_Z, _momentum)
    _eccentricity = 1 / mu * ((vel ** 2 - mu / pos) * r - (np.dot(r, v)) * v)

    e = np.linalg.norm(_eccentricity)
    
    energy = vel ** 2 / 2 - mu / pos
    a = - mu / (2 * energy)

    cos_i = np.dot(_momentum, DIR_Z) / np.linalg.norm(_momentum)
    i = np.degrees(np.arccos(cos_i))

    cos_omega = np.dot(_line_of_nodes, DIR_X) / np.linalg.norm(_line_of_nodes)
    omega = np.degrees(np.arccos(cos_omega))

    if np.dot(_line_of_nodes, DIR_Y) < 0:
        omega = 360 - omega

    cos_w = np.dot(_line_of_nodes, _eccentricity) / (np.linalg.norm(_line_of_nodes) * e)
    w = np.degrees(np.arccos(cos_w))

    if np.dot(_eccentricity, DIR_Z) < 0:
        w = 360 - w
    
    cos_u = np.dot(_eccentricity, r) / (e * pos)
    u = np.degrees(np.arccos(cos_u))

    if np.dot(r, v) < 0:
        u = 360 - u
    
    return [a, e, i, omega, w, u]


def standard_to_cartesian(a, e, i, raan, w, u, mu=MU_EARTH):
    """_summary_

    Args:
        a (float): Semi-major axis (m)
        e (float): Eccentricity ()
        i (float): Inclination (deg)
        raan (float): RAAN (deg)
        w (float): Argument of periapsis (deg)
        u (float): True anamoly (deg)

    Returns:
        r
        v
    """
    
    r = a * (1 - e ** 2) / (1 + e * np.cos(np.radians(u)))

    _r = np.array([r * np.cos(np.radians(u)), r * np.sin(np.radians(u)), 0])
    _v = math.sqrt(mu / (a * (1 - e ** 2))) * np.array([-np.sin(np.radians(u)), e + np.cos(np.radians(u)), 0])

    rad_raan = np.radians(raan)
    rad_i = np.radians(i)
    rad_w = np.radians(w)

    R_3_raan = np.array([[np.cos(rad_raan) , np.sin(rad_raan), 0],
                         [-np.sin(rad_raan), np.cos(rad_raan) , 0],
                         [0                , 0               , 1]])

    R_1_i = np.array([[1, 0, 0],
                      [0, np.cos(rad_i), np.sin(rad_i)],
                      [0, -np.sin(rad_i), np.cos(rad_i)]])

    R_3_w = np.array([[np.cos(rad_w) , np.sin(rad_w), 0],
                      [-np.sin(rad_w), np.cos(rad_w), 0],
                      [0             , 0         , 1]])

    R_eff = R_3_w @ R_1_i @ R_3_raan
    R_eff = np.matrix.transpose(R_eff)
    
    _pos = R_eff @ _r
    _vel = R_eff @ _v
    return [_pos, _vel]


def propogate_2BP(t, state, mu=MU_EARTH):
    """State space representation of Newton's Law of Gravitation. Only implemented for Earth.
        Selected state variables are [r_x, r_y, r_z, v_x, v_y, v_z]

    Args:
        t (float): Current time step
        state (arr): State vector of form [r_x, r_y, r_z, v_x, v_y, v_z]

    Returns:
        Value of velocity and acceleration at the given time and state in form:
            [v_x, v_y, v_z, a_x, a_y, a_z]
    """

    _r = state[0:3]
    _v = state[3:]

    # a = - mu / norm(r) ^ 3 * r
    _a = - mu / (np.linalg.norm(_r) ** 3) * _r

    val = np.array([_v[0], _v[1], _v[2], _a[0], _a[1], _a[2]])
    return val


def calculate_orbital_period(a, mu=MU_EARTH):
    """Calculates orbital period of an orbit.

    Args:
        a (float): Semi-major axis of orbit
        mu (float, optional): Gravitional constant for body. Defaults to MU_EARTH.

    Returns:
        float: Orbital peroid in seconds
    """
    return 2 * math.pi * math.sqrt(a ** 3 / mu )


def calculate_orbital_energy(r, v, mu=MU_EARTH):
    """Calculates orbital energy

    Args:
        r (float): Radius magnitude
        v (float): Velocity magnitude
    
    Returns:
        float: Energy of orbit
    """
    return v ** 2 / 2 - mu / r

def calculate_angular_momentum(_r, _v):
    """Calculates angular momentum

    Args:
        _r (arr): Position vector
        _v (arr): Velocity vector

    Returns:
        arr: Angular Momentum
    """
    return np.cross(_r, _v)


def calculate_true_anomoly(_e, _r, _v):
    """Calculate true anamoly

    Args:
        _e (_type_): _description_
        _r (_type_): Radius vector
        _v (_type_): Velocity vectore

    Returns:
        _type_: _description_
    """
    
    v = np.arccos(np.dot(_e, _r) / (np.linalg.norm(_e) * np.linalg.norm(_r)))

    if np.dot(_r, _v) < 0:
        v = 2*math.pi - v
    
    return v

def calculate_final_state(_r,_v,tof):
    """Calculate final state of spacecraft

    Args:
        _r (arr[float]): Initial position vector (m)
        _v (arr[float]): Initial velocity vectory (m/s)
        tof (float): Time of flight (s)
        
    Returns: 
        _rv (arr[float]): Final state vector
    """

    init_state = np.array([_r[0], _r[1], _r[2], _v[0], _v[1], _v[2]])
    tol = 10**-13

    orbit = solve_ivp(propogate_2BP, [0, tof], init_state, method="RK45", atol=tol, rtol=tol)

    # pos = []
    # vel = []
    # acc = []
    # for i in range(orbit.t.shape[0]):
    #     pos.append(math.sqrt(orbit.y[0][i] ** 2 + orbit.y[1][i] ** 2 + orbit.y[2][i] ** 2))
    #     vel.append(math.sqrt(orbit.y[3][i] ** 2 + orbit.y[4][i] ** 2 + orbit.y[5][i] ** 2))
    #     a_x = - MU_EARTH / (pos[-1] ** 3) * orbit.y[0][i]
    #     a_y = - MU_EARTH / (pos[-1] ** 3) * orbit.y[1][i]
    #     a_z = - MU_EARTH / (pos[-1] ** 3) * orbit.y[2][i]
    #     acc.append(math.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2))

    # #plt.style.use('_mpl-gallery')
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    # ax1.plot(orbit.t, pos)
    # ax1.set_xlabel('Time (s)', size=10)
    # ax1.set_ylabel('Position (m)', size=10)
    # ax1.set_title("Postion Vs Time of Orbit A (t = 2P)")

    # ax2.plot(orbit.t, vel)
    # ax2.set_xlabel('Time (s)', size=10)
    # ax2.set_ylabel('Velocity (m/s)', size=10)
    # ax2.set_title("Velocity Vs Time of Orbit A (t = 2P)")


    # ax3.plot(orbit.t, acc)
    # ax3.set_xlabel('Time (s)', size=10)
    # ax3.set_ylabel('Acceleration (m/s^2)', size=10)
    # ax3.set_title("Acceleration Vs Time of Orbit A (t = 2P)")

    # plt.tight_layout()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2]) 
    # ax.set_xlabel('x (m)', size=10)
    # ax.set_ylabel('y (m)', size=10)
    # ax.set_zlabel('z (m)', size=10)
    # ax.set_title("Orbital Path of Orbit A")
    
    # plt.tight_layout()

    # # label periapsis
    # mins, _ = find_peaks(np.array(pos)*-1)

    # min_inds = [0, *mins, len(pos) - 1]
    # t_s = [orbit.t[i] for i in min_inds]
    # y_s = [pos[i] for i in min_inds]

    # ax1.plot(t_s, y_s, "ro")

    # init_energy = vel[0] ** 2 / 2 - MU_EARTH / pos[0]
    # energy_deviation = [calculate_orbital_energy(pos[i], vel[i]) - init_energy for i in range(len(orbit.t))]

    # fig2 = plt.figure()
    # ax4 = plt.axes()

    # ax4.plot(orbit.t, energy_deviation)
    # ax4.set_xlabel('Time (s)', size=10)
    # ax4.set_ylabel('Deviation from Initial Orbital Energy (J/kg)', size=10)
    # ax4.set_title("Deviation from Initial Orbital Energy Vs Time of Orbit A (t = 2P)")

    # _rad = []
    # _vel = []
    # _angular_momentum = []
    # for i in range(len(orbit.t)):
    #     _rad_c = np.array([orbit.y[0][i], orbit.y[1][i], orbit.y[2][i]])
    #     _vel_c = np.array([orbit.y[3][i], orbit.y[4][i], orbit.y[5][i]])
    #     _rad.append(_rad_c)
    #     _vel.append(_vel_c)
    #     _angular_momentum.append(calculate_angular_momentum(_rad_c, _vel_c))

    # init_momentum = np.linalg.norm(_angular_momentum[0])
    # momentum_deviation = []
    # for item in _angular_momentum:
    #     momentum_deviation.append(np.linalg.norm(item) - init_momentum)

    
    # fig3, momentum_axes = plt.subplots(3, 1)

    # for i, ax_symbol in enumerate(["x", "y", "z"]):
    #     momentum_axes[i].plot(orbit.t, [j[i] for j in _angular_momentum])
    #     momentum_axes[i].set_xlabel('Time (s)', size=10)
    #     momentum_axes[i].set_ylabel(f'Angular Momentum (m^2/s)', size=10)
    #     momentum_axes[i].set_title(f'{ax_symbol.upper()}-Component of Angular Momentum Vs Time')

    # plt.tight_layout()
    
    # fig4 = plt.figure()
    # ang_dev_axes = plt.axes()
    # ang_dev_axes.plot(orbit.t, momentum_deviation)
    
    # ang_dev_axes.set_xlabel('Time (s)', size=10)
    # ang_dev_axes.set_ylabel(f'Deviation of Angular Momentum (m^2/s)', size=10)
    # ang_dev_axes.set_title('Deviation of Angular Momentum from Initial Value Vs Time')

    # plt.tight_layout()

    # true_anomoly = []
    # for i in range(len(orbit.t)):
    #     _e = 1 / MU_EARTH * ((vel[i] ** 2 - MU_EARTH / pos[i]) * _rad[i] - (np.dot(_rad[i], _vel[i])) * _vel[i])
    #     true_anomoly.append(calculate_true_anomoly(_e, _rad[i], _vel[i]))

    # fig5 = plt.figure()
    # t_anom_axes = plt.axes()

    # t_anom_axes.plot(orbit.t, true_anomoly)
    # t_anom_axes.set_xlabel('Time (s)', size=10)
    # t_anom_axes.set_ylabel(f'True Anomoly (rads)', size=10)
    # t_anom_axes.set_title('True Anomoly Vs Time')

    # plt.show()
    final_state = [orbit.y[0][-1], orbit.y[1][-1], orbit.y[2][-1], orbit.y[3][-1], orbit.y[4][-1], orbit.y[5][-1]]
    return final_state
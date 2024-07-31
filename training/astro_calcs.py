import numpy as np
import math
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# Specific Gravitation Constants
MU_EARTH = 3.986004418 * 10 ** 5 # km^3 s^-2
G = 6.67408 * 10 ** -11 # m^3 kg^-1 s^-2
RADIUS_EARTH = 6378 # km

# Natural Basis
DIR_X = np.array([1, 0, 0])
DIR_Y = np.array([0, 1, 0])
DIR_Z = np.array([0, 0, 1])

def cartesian_to_standard(r, v, mu=MU_EARTH):
    """Converts Cartian orbital elements to standard orbital elements.

    Args:
        r (array[float]): Position vector (m)
        v (array[float]): Velocity vector (m)
        mu (float): Specific gravity of celestial body (m^3 s^-2)

    Returns:
        A list containing the orbital elements: [a, e, i, omega, w, u]
    """

    # Convert initial state to float
    r = r.astype('float64')
    v = v.astype('float64')

    # Eccentricity, e
    h = np.cross(r, v) # angular momentum
    n = np.cross(DIR_Z, h) # line of nodes
    e_vec = 1 / mu * ((np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)) * r - (np.dot(r, v)) * v) # points of periapsis
    e = np.linalg.norm(e_vec) # eccentricity
    
    # Semi-major axis, a
    energy = np.linalg.norm(v) ** 2 / 2 - mu / np.linalg.norm(r)
    a = - mu / (2 * energy)

    # Angle of inclination, i
    cos_i = np.dot(h, DIR_Z) / np.linalg.norm(h)
    i = np.degrees(np.arccos(cos_i))

    # Right ascension of ascending node, omega (deg)
    cos_omega = np.dot(n, DIR_X) / np.linalg.norm(n)
    omega = np.degrees(np.arccos(cos_omega))

    if np.dot(n, DIR_Y) < 0:
        omega = 360 - omega

    # Argument of periapsis, w
    cos_w = np.dot(n, e_vec) / (np.linalg.norm(n) * e)
    w = np.degrees(np.arccos(cos_w))

    if np.dot(e_vec, DIR_Z) < 0:
        w = 360 - w
    
    # Trua anomaly, nu
    cos_nu = np.dot(e_vec, r) / (e * np.linalg.norm(r))
    nu = np.degrees(np.arccos(cos_nu))

    if np.dot(r, v) < 0:
        nu = 360 - nu
    
    return [a, e, i, omega, w, nu]


def standard_to_cartesian(a, e, i, raan, w, nu, mu=MU_EARTH):
    """Converts orbital elements to Cartesian coordinates

    Args:
        a (float): Semi-major axis (m)
        e (float): Eccentricity ()
        i (float): Inclination (deg)
        raan (float): RAAN (deg)
        w (float): Argument of periapsis (deg)
        nu (float): True anamoly (deg)

    Returns:
        r
        v
    """
    
    # Position and velocity in perifocal frame
    p = a * (1 - e ** 2) # periapsis
    r_vec = np.array([p * np.cos(np.radians(nu)) / (1 + e * np.cos(np.radians(nu))), p * np.sin(np.radians(nu)) / (1 + e * np.cos(np.radians(nu))), 0])
    v_vec = math.sqrt(mu / (a * (1 - e ** 2))) * np.array([-np.sin(np.radians(nu)), e + np.cos(np.radians(nu)), 0])

    # Rotations
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
    
    # Apply transformation to return to inertial frame
    r = R_eff @ r_vec
    v = R_eff @ v_vec
    return [r, v]


def propagate_2BP(t, X, mu=MU_EARTH):
    """State space representation of Newton's Law of Gravitation
        Selected state variables are [r_x, r_y, r_z, v_x, v_y, v_z]

    Args:
        t (float): Current time step
        X (arr): State vector of form [r_x, r_y, r_z, v_x, v_y, v_z]

    Returns:
        Value of velocity and acceleration at the given time and state in form:
            [v_x, v_y, v_z, a_x, a_y, a_z]
    """

    # Extract position and velocity
    r = X[0:3]
    v = X[3:6]

    # Calculate acceleration for new state vector
    a = - mu / (np.linalg.norm(r) ** 3) * r

    dXdt = np.array([v[0], v[1], v[2], a[0], a[1], a[2]])
    return dXdt

def calculate_orbit(r,v,tof):
    """Calculate final state of spacecraft

    Args:
        r (arr[float]): Initial position vector (m)
        v (arr[float]): Initial velocity vectory (m/s)
        tof (float): Time of flight (s)
        
    Returns: 
        final_state (arr[float]): Final state vector
    """

    init_state = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
    tol = 10**-13

    sol = solve_ivp(propagate_2BP, [0, tof], init_state, method="RK45", atol=tol, rtol=tol, t_eval = np.linspace(0, tof, 1000))

    # Convert Cartesian coordinates to orbital elements for each timestep
    orbital_elements = np.array([cartesian_to_standard(sol.y[0:3, i], sol.y[3:6, i]) for i in range(len(sol.t))])

    # final_state = [orbit.y[0][-1], orbit.y[1][-1], orbit.y[2][-1], orbit.y[3][-1], orbit.y[4][-1], orbit.y[5][-1]]
    return sol, orbital_elements

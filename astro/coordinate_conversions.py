import numpy as np
from . import constants
import math

# Natural Basis
DIR_X = np.array([1, 0, 0])
DIR_Y = np.array([0, 1, 0])
DIR_Z = np.array([0, 0, 1])


def cartesian_to_standard(r, v, mu=constants.MU_EARTH):
    """Converts Cartesian orbital elements to standard orbital elements.

    Args:
        r (array[float]): Position vector (m)
        v (array[float]): Velocity vector (m)
        mu (float): Specific gravity of celestial body (m^3 s^-2)

    Returns:
        A list containing the orbital elements: [a, e, i, omega, w, u]
    """

    # Convert initial state to float
    r = r.astype("float64")
    v = v.astype("float64")

    # Eccentricity, e
    h = np.cross(r, v)  # angular momentum
    n = np.cross(DIR_Z, h)  # line of nodes
    e_vec = (
        1
        / mu
        * ((np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)) * r - (np.dot(r, v)) * v)
    )  # points of periapsis
    e = np.linalg.norm(e_vec)  # eccentricity

    # Semi-major axis, a
    energy = np.linalg.norm(v) ** 2 / 2 - mu / np.linalg.norm(r)
    a = -mu / (2 * energy)

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

    # True anomaly, nu
    cos_nu = np.dot(e_vec, r) / (e * np.linalg.norm(r))
    nu = np.degrees(np.arccos(cos_nu))

    if np.dot(r, v) < 0:  # CHANGE THIS TO <=
        nu = 360 - nu

    if np.isclose(nu, 360.0):
        nu = 0.0

    return [a, e, i, omega, w, nu]


def standard_to_cartesian(a, e, i, raan, w, nu, mu=constants.MU_EARTH):
    """Converts orbital elements to Cartesian coordinates

    Args:
        a (float): Semi-major axis (m)
        e (float): Eccentricity ()
        i (float): Inclination (deg)
        raan (float): RAAN (deg)
        w (float): Argument of periapsis (deg)
        nu (float): True anomaly (deg)

    Returns:
        r
        v
    """

    # Position and velocity in perifocal frame
    p = a * (1 - e**2)  # periapsis
    r_vec = np.array(
        [
            p * np.cos(np.radians(nu)) / (1 + e * np.cos(np.radians(nu))),
            p * np.sin(np.radians(nu)) / (1 + e * np.cos(np.radians(nu))),
            0,
        ]
    )
    v_vec = math.sqrt(mu / (a * (1 - e**2))) * np.array(
        [-np.sin(np.radians(nu)), e + np.cos(np.radians(nu)), 0]
    )

    # Rotations
    rad_raan = np.radians(raan)
    rad_i = np.radians(i)
    rad_w = np.radians(w)

    R_3_raan = np.array(
        [
            [np.cos(rad_raan), np.sin(rad_raan), 0],
            [-np.sin(rad_raan), np.cos(rad_raan), 0],
            [0, 0, 1],
        ]
    )

    R_1_i = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rad_i), np.sin(rad_i)],
            [0, -np.sin(rad_i), np.cos(rad_i)],
        ]
    )

    R_3_w = np.array(
        [
            [np.cos(rad_w), np.sin(rad_w), 0],
            [-np.sin(rad_w), np.cos(rad_w), 0],
            [0, 0, 1],
        ]
    )

    R_eff = R_3_w @ R_1_i @ R_3_raan
    R_eff = np.matrix.transpose(R_eff)

    # Apply transformation to return to inertial frame
    r = R_eff @ r_vec
    v = R_eff @ v_vec
    return r, v
    # return np.concatenate((r, v))

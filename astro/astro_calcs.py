import numpy as np
from scipy.integrate import solve_ivp
from . import constants
from . import coordinate_conversions
import math

def propagate_2BP(t, X, mu=constants.MU_EARTH): 
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
    a = -mu / (np.linalg.norm(r) ** 3) * r

    dXdt = np.array([v[0], v[1], v[2], a[0], a[1], a[2]])
    return dXdt


def calculate_orbit(r, v, tof, mu=constants.MU_EARTH):
    """Calculate final state of spacecraft

    Args:
        r (arr[float]): Initial position vector (m)
        v (arr[float]): Initial velocity vector (m/s)
        tof (float): Time of flight (s)

    Returns:
        final_state (arr[float]): Final state vector
    """

    init_state = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
    tol = 10**-13

    sol = solve_ivp(
        propagate_2BP,
        [0, tof],
        init_state,
        method="RK45",
        atol=tol,
        rtol=tol,
        t_eval=np.linspace(0, tof, 1000),
        args=(mu,)
    )

    # Convert Cartesian coordinates to orbital elements for each timestep
    orbital_elements = np.array(
        [
            coordinate_conversions.cartesian_to_standard(sol.y[0:3, i], sol.y[3:6, i])
            for i in range(len(sol.t))
        ]
    )

    return sol, orbital_elements

def calculate_orbital_period(a, mu=constants.MU_EARTH):
    """Calculates orbital period of an orbit.

    Args:
        a (float): Semi-major axis of orbit
        mu (float, optional): Gravitational constant for body. Defaults to constants.MU_EARTH.

    Returns:
        float: Orbital period in seconds
    """
    return 2 * math.pi * math.sqrt(a ** 3 / mu )
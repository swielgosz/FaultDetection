import numpy as np
from scipy.integrate import solve_ivp
from . import constants
from . import coordinate_conversions
import math


def propagate_with_radial_thrust(
    t, X, mu=constants.MU_EARTH, thrust_magnitude=0.0001, thrust_time_range=(0, np.inf)
):
    """State space representation of orbit propagation with radial thrust in a specified time range.

    Parameters:
    - t: Current time (seconds).
    - X: State vector [rx, ry, rz, vx, vy, vz].
    - mu: Gravitational parameter of the central body.
    - thrust_magnitude: Magnitude of the radial thrust (km/s^2).
    - thrust_time_range: Tuple specifying the time range (start, end) during which thrust is applied.
    """
    # Extract position and velocity
    r = X[0:3]  # Position vector
    v = X[3:6]  # Velocity vector

    # Check if the current time falls within the thrust time range
    start_time, end_time = thrust_time_range
    if start_time <= t <= end_time:
        # Radial thrust (toward or away from the central body)
        thrust_direction = r / np.linalg.norm(r)  # Normalize the position vector
        thrust_acceleration = thrust_magnitude * thrust_direction
    else:
        thrust_acceleration = np.array(
            [0.0, 0.0, 0.0]
        )  # No thrust outside the specified range

    # Gravitational acceleration
    a_gravity = -mu / (np.linalg.norm(r) ** 3) * r

    # Total acceleration (gravity + thrust)
    a_total = a_gravity + thrust_acceleration

    # Time derivative of the state vector
    dXdt = np.array([v[0], v[1], v[2], a_total[0], a_total[1], a_total[2]])

    return dXdt


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


def calculate_orbit(r0, v0, tof, mu=constants.MU_EARTH, propagate_fn=None, **kwargs):
    """
    Propagate an orbit using the provided initial state and time of flight (tof).
    Optionally, use a custom propagation function if `propagate_fn` is provided.

    Args:
        r0 (array): Initial position vector in km.
        v0 (array): Initial velocity vector in km/s.
        tof (float): Time of flight in seconds.
        mu (float): Gravitational parameter.
        propagate_fn (function): Custom propagation function (optional).

    Returns:
        sol (OdeSolution): Solution object containing time and solution values.
        cartesian_sol (array): Solution of the orbit in Cartesian coordinates.
        orbital_elements (array): Orbital elements at each time step.
    """
    # Use the provided propagation function or default to the 2-body propagation
    if propagate_fn is None:
        # Default to the 2-body problem (without thrust)
        propagate_fn = propagate_2BP

    # Define the time vector for the integration
    t_span = np.linspace(0, tof, num=1000)  # Adjust the number of points as needed

    # Set the initial state vector [r, v]
    initial_state = np.concatenate((r0, v0))

    tol = 10**-13

    # Perform the integration (use the chosen propagation function)
    sol = solve_ivp(
        propagate_fn,
        (0, tof),
        initial_state,
        atol=tol,
        rtol=tol,
        t_eval=t_span,
        args=(mu, *kwargs.values()),
    )

    # Extract the Cartesian solution
    cartesian_sol = sol.y[:6].T  # Extract position and velocity

    # Convert Cartesian coordinates to orbital elements at each time step
    orbital_elements = np.array(
        [
            coordinate_conversions.cartesian_to_standard(sol.y[0:3, i], sol.y[3:6, i])
            for i in range(len(sol.t))
        ]
    )

    return sol, cartesian_sol, orbital_elements


def calculate_orbital_period(a, mu=constants.MU_EARTH):
    """Calculates orbital period of an orbit.

    Args:
        a (float): Semi-major axis of orbit
        mu (float, optional): Gravitational constant for body. Defaults to constants.MU_EARTH.

    Returns:
        float: Orbital period in seconds
    """
    return 2 * math.pi * math.sqrt(a**3 / mu)

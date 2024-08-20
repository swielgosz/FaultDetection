import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import astro_calcs, constants, visualization
import numpy as np


def num_1():

    # Initial state
    r0 = np.array([14737, 2559, 450])  # km
    v0 = np.array([-1, 5.7, 1])  # km/s

    # Calculate one period (s)
    a = astro_calcs.coordinate_conversions.cartesian_to_standard(r0, v0)[0]
    tof = astro_calcs.calculate_orbital_period(a, mu=constants.MU_EARTH)
    cartesian_sol, orbital_elements = astro_calcs.calculate_orbit(
        r0, v0, tof, constants.MU_EARTH
    )
    visualization.plot_3d_orbit(cartesian_sol)


if __name__ == "__main__":
    num_1()

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import astro
from astro import astro_calcs, constants, visualization
import numpy as np


def generate_single_faulted_orbit():
    dataset_oe = []
    dataset_cartesian = []

    # Orbital parameters
    a = 15000 + constants.RADIUS_EARTH  # Semi-major axis in km
    e = 0.3  # Eccentricity
    i = 10.0  # Inclination in degrees
    raan = 0.0  # RAAN in degrees
    w = 10.0  # Argument of periapsis in degrees
    nu = 0.0  # True anomaly in degrees

    # Convert to Cartesian coordinates
    r0, v0 = astro.coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    tof = astro_calcs.calculate_orbital_period(
        a, mu=constants.MU_EARTH
    )  # Orbital period in seconds

    # Use the propagate_with_radial_thrust function for propagation with thrust
    sol, cartesian_sol, orbital_elements = astro_calcs.calculate_orbit(
        r0,
        v0,
        tof,
        constants.MU_EARTH,
        propagate_fn=astro_calcs.propagate_with_radial_thrust,
    )

    # Visualize the orbit
    visualization.plot_3d_orbit(sol)

    # Record time, orbital elements
    for t, elems in zip(sol.t, orbital_elements):
        dataset_oe.append([t, *elems])

    # Record time, Cartesian coordinates
    for t, elems in zip(sol.t, cartesian_sol):
        dataset_cartesian.append([t, *elems])

    # Ensure the directories exist
    output_path_oe = "../datasets/dataset_oe_with_thrust.npy"
    output_path_cartesian = "../datasets/dataset_cartesian_with_thrust.npy"
    output_dir_oe = os.path.dirname(output_path_oe)
    output_dir_cartesian = os.path.dirname(output_path_cartesian)

    if not os.path.exists(output_dir_oe):
        os.makedirs(output_dir_oe)
    if not os.path.exists(output_dir_cartesian):
        os.makedirs(output_dir_cartesian)

    # Save the datasets
    np.save(output_path_oe, dataset_oe)
    print(f"\nOrbital elements dataset saved to {output_path_oe}")
    np.save(output_path_cartesian, dataset_cartesian)
    print(f"\nCartesian dataset saved to {output_path_cartesian}")


if __name__ == "__main__":
    generate_single_faulted_orbit()
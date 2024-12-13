# %%
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import astro
from astro import astro_calcs, constants, visualization
import numpy as np


def generate_single_healthy_orbit():

    dataset_oe = []
    dataset_cartesian = []
    # Initial state
    # r0 = np.array([14737, 2559, 450])  # km
    # v0 = np.array([-1, 5.7, 1])  # km/s

    a = 15000 + constants.RADIUS_EARTH
    e = 0.3
    i = 10.0
    raan = 0.0
    w = 10.0
    nu = 0.0

    # Calculate one period (s)
    # a = astro_calcs.coordinate_conversions.cartesian_to_standard(r0, v0)[0]
    r0, v0 = astro.coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    tof = astro_calcs.calculate_orbital_period(a, mu=constants.MU_EARTH)
    sol, cartesian_sol, orbital_elements = astro_calcs.calculate_orbit(
        r0, v0, tof, constants.MU_EARTH
    )
    visualization.plot_3d_orbit(sol)

    # Record time, orbital elements
    for t, elems in zip(sol.t, orbital_elements):
        dataset_oe.append([t, *elems])

    # Record time, cartesian coordinates
    for t, elems in zip(sol.t, cartesian_sol):
        dataset_cartesian.append([t, *elems])

    # Ensure the directories exists
    output_path_oe = "../datasets/dataset_oe.npy"
    # data_path = os.path.join(script_dir, "../datasets/dataset_oe.npy")
    # data_np = np.load(data_path)
    output_path_cartesian = "../datasets/dataset_cartesian.npy"
    output_dir_oe = os.path.dirname(output_path_oe)
    output_dir_cartesian = os.path.dirname(output_path_cartesian)
    if not os.path.exists(output_dir_oe):
        os.makedirs(output_dir_oe)
    if not os.path.exists(output_dir_cartesian):
        os.makedirs(output_dir_cartesian)

    # Save the arrays as .npy files
    np.save(output_path_oe, dataset_oe)
    print(f"\nOrbital elements dataset saved to {output_path_oe}")
    np.save(output_path_cartesian, dataset_cartesian)
    print(f"\nCartesian dataset saved to {output_path_cartesian}")


if __name__ == "__main__":
    generate_single_healthy_orbit()

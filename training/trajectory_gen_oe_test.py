import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import random
from astro import astro_calcs, constants, visualization

def generate_dataset_oe(output_path, altitude_range, e_range, i_range, raan_range, w_range, num_trajs):
    dataset = []

    # Gravitational parameter for Earth
    mu = constants.MU_EARTH  # Ensure you have the correct gravitational parameter
    min_radius = constants.RADIUS_EARTH  # Earth's radius in km

    for idx in range(num_trajs):
        while True:  # Loop until valid parameters are found
            # Random orbital elements within range
            altitude = random.uniform(*altitude_range)
            b = min_radius + altitude  # Distance from center of the Earth
            e = random.uniform(*e_range)

            # Calculate semi-major axis based on eccentricity
            a = b / np.sqrt(1 - e**2) if e < 1 else None  # Prevent division by zero for e=1

            if a is not None and a >= min_radius + altitude:  # Check if the semi-major axis is valid
                break  # Exit the loop if valid parameters are found

        i = random.uniform(*i_range)
        raan = random.uniform(*raan_range)
        w = random.uniform(*w_range)
        nu = 0
        tof = astro_calcs.calculate_orbital_period(a, mu)
        initial_state = np.array([a, e, i, raan, w, nu])

        # Initial state in Cartesian
        r0, v0 = astro_calcs.coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)

        # Generate trajectory data
        sol, cartesian_sol, orbital_elements = astro_calcs.calculate_orbit(r0, v0, tof, constants.MU_EARTH)
        trajectory_data = []
        for t, elems in zip(sol.t, orbital_elements):
            trajectory_data.append([t, *elems])

        # Store initial state and trajectory data
        dataset.append({"initial_state": initial_state, "trajectory": trajectory_data})

        print(f"Progress: {idx/num_trajs * 100:.2f}%", end="\r")

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the dataset as a .npy file
    np.save(output_path, dataset)
    print(f"\nDataset saved to {output_path}")


if __name__ == "__main__":
    altitude_range = [160, 35786]  # LEO to GEO altitudes in km
    e_range = [0, 0.5]  # Eccentricity range
    i_range = [0, np.pi]  # Inclination range (0 to 180 degrees)
    raan_range = [0, 2 * np.pi]  # RAAN range (0 to 360 degrees)
    w_range = [0, 2 * np.pi]  # Argument of periapsis range (0 to 360 degrees)
    num_trajs = 100

    generate_dataset_oe(
        "../datasets/dataset_generalized.npy",
        altitude_range,
        e_range,
        i_range,
        raan_range,
        w_range,
        num_trajs,  # Number of trajectories
    )

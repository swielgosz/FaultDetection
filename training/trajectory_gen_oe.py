import sys
import os
import numpy as np
import random
from astro import astro_calcs, constants

def generate_dataset_oe(output_path, altitude_range, e_range, inc_range, raan_range, arg_periapsis_range, num_trajs):
    dataset = []

    # Gravitational parameter for Earth
    mu = constants.MU_EARTH  # Ensure you have the correct gravitational parameter

    for i in range(num_trajs):
        # Generate a random semi-minor axis (altitude in km)
        b = random.uniform(*altitude_range)
        r = constants.RADIUS_EARTH + b  # Distance from center of the Earth

        # Generate random eccentricity
        e = random.uniform(*e_range)

        # Calculate semi-major axis
        a = b / np.sqrt(1 - e**2)

        # Calculate velocity based on semi-major axis
        v_max = np.sqrt(mu * (2/r - 1/a))
        v = random.uniform(0, v_max)

        # Generate random inclination, RAAN, and argument of periapsis
        inclination = random.uniform(*inc_range)
        raan = random.uniform(*raan_range)
        arg_periapsis = random.uniform(*arg_periapsis_range)

        # Initial state
        initial_state = (np.array([r, 0, 0]), np.array([0, v, 0]))  # Example initial state
        
        # Time of flight
        tof = random.uniform(10 * 60, 2 * 60 * 60)  # Random time of flight in seconds
        
        # Generate trajectory data
        trajectory_data = generate_trajectory_data(initial_state, tof, mu)
        
        # Store initial state and trajectory data
        dataset.append({
            "initial_state": initial_state,
            "trajectory": trajectory_data
        })

        print(f"Progress: {i/num_trajs * 100:.2f}%", end="\r")

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the dataset as a .npy file
    np.save(output_path, dataset)
    print(f"\nDataset saved to {output_path}")

if __name__ == "__main__":
    altitude_range = [100, 35786]  # LEO to GEO altitudes in km
    e_range = [0, 0.5]  # Eccentricity range
    inc_range = [0, np.pi]  # Inclination range (0 to 180 degrees)
    raan_range = [0, 2 * np.pi]  # RAAN range (0 to 360 degrees)
    arg_periapsis_range = [0, 2 * np.pi]  # Argument of periapsis range (0 to 360 degrees)

    generate_dataset_oe(
        "../datasets/dataset_trajectory.npy",
        altitude_range,
        e_range,
        inc_range,
        raan_range,
        arg_periapsis_range,
        100,  # Number of trajectories
    )

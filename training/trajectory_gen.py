import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import astro_calcs, constants, utils
import numpy as np
import random

def generate_dataset(output_path, _r_range, _v_range, tof_range, num_trajs):

    dataset = []

    for i in range(num_trajs):
        _r_gen = np.array(
            [
                utils.signed_rand_float(_r_range),
                utils.signed_rand_float(_r_range),
                utils.signed_rand_float(_r_range),
            ]
        )
        _v_gen = np.array(
            [
                utils.signed_rand_float(_v_range),
                utils.signed_rand_float(_v_range),
                utils.signed_rand_float(_v_range),
            ]
        )
        tof = random.uniform(*tof_range)
        # tof = 48*60*60 # 48 hrs

        cartesian_sol, orbital_elements = astro_calcs.calculate_orbit(
            _r_gen, _v_gen, tof
        )

        # Record time of flight, time, orbital elements
        for t, elems in zip(cartesian_sol.t, orbital_elements):
            dataset.append([tof, t, *elems])

        print(f"Progress: {i/num_trajs * 100:.2f}%", end="\r")


    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the array as a .npy file
    np.save(output_path, dataset)
    print(f"\nDataset saved to {output_path}")


if __name__ == "__main__":

    r0 = np.array([14737, 2559, 450])  # km
    v0 = np.array([-1, 5.7, 1])  # km/s
    
    generate_dataset(
        "../datasets/dataset_cartesian.npy",
        [constants.RADIUS_EARTH + 100, 10 * 10**3],
        [3, 20],
        [10 * 60, 2 * 60 * 60],
        1 * 10**2,
    )

#%%
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import visualization, coordinate_conversions

# Relative path to the .npy file
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, "../datasets/dataset_oe.npy")

# Load the NumPy array
data = np.load(relative_path)

# Inspect the shape and structure of the data
print(data.shape)
print(data[:5])  # Print the first 5 rows to understand the data structure

# Extract features and labels if needed
# Adjust the slicing according to your dataset's structure
# For example, assuming the data has columns [tof, time, a, e, i, omega, w, u]
time = data[:, 0]
orbital_elements = data[:, 1:]

cartesian_converted = []

for i in range(orbital_elements.shape[0]):
    a, e, i, raan, w, nu = orbital_elements[i]
    r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    cartesian_converted.append(np.concatenate((r, v)))

cartesian_converted = np.array(cartesian_converted)

# Plot the converted Cartesian data
cartesian_converted = np.column_stack((time,cartesian_converted))

# # Relative path to the .npy file for existing Cartesian data
relative_path_cartesian = "../datasets/dataset_cartesian.npy"
data_cartesian = np.load(relative_path_cartesian)

# # Plot the actual Cartesian data for comparison
visualization.compare_orbits(cartesian_converted,data_cartesian)

# %%

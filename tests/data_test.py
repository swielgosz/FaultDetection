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

# # Plot each orbital element separately
# plt.figure(figsize=(12, 12))

# # Plot Semi-Major Axis (a)
# plt.subplot(3, 2, 1)
# plt.plot(time, orbital_elements[:, 0], label="Semi-Major Axis (a)", color="b")
# plt.xlabel("Time")
# plt.ylabel("Semi-Major Axis (a)")
# plt.title("Semi-Major Axis vs Time")
# plt.legend()

# # Plot Eccentricity (e)
# plt.subplot(3, 2, 2)
# plt.plot(time, orbital_elements[:, 1], label="Eccentricity (e)", color="g")
# plt.xlabel("Time")
# plt.ylabel("Eccentricity (e)")
# plt.title("Eccentricity vs Time")
# plt.legend()

# # Plot Inclination (i)
# plt.subplot(3, 2, 3)
# plt.plot(time, orbital_elements[:, 2], label="Inclination (i)", color="r")
# plt.xlabel("Time")
# plt.ylabel("Inclination (i)")
# plt.title("Inclination vs Time")
# plt.legend()

# # Plot RAAN (omega)
# plt.subplot(3, 2, 4)
# plt.plot(time, orbital_elements[:, 3], label="RAAN (omega)", color="c")
# plt.xlabel("Time")
# plt.ylabel("RAAN (omega)")
# plt.title("RAAN vs Time")
# plt.legend()


# # Plot Argument of periapsis (w)
# plt.subplot(3, 2, 5)
# plt.plot(time, orbital_elements[:, 4], label="Argument of periapsis (w)", color="m")
# plt.xlabel("Time")
# plt.ylabel("Argument of periapsis (w)")
# plt.title("Argument of Periapsis vs Time")
# plt.legend()

# # Plot True Anomaly (nu)
# plt.subplot(3, 2, 6)
# plt.plot(time, orbital_elements[:, 5], label="True Anomaly (nu)", color="m")
# plt.xlabel("Time")
# plt.ylabel("True Anomaly (nu)")
# plt.title("True Anomaly vs Time")
# plt.legend()

# plt.tight_layout()
# plt.show()


# Relative path to the .npy file
relative_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")

# Load the NumPy array
data = np.load(relative_path)

# Inspect the shape and structure of the data
print(data.shape)
print(data[:5])  # Print the first 5 rows to understand the data structure

# Extract features and labels if needed
# Adjust the slicing according to your dataset's structure
# For example, assuming the data has columns [tof, time, a, e, i, omega, w, u]
time = data[:, 0]
cartesian_elements = data[:, 1:]

# Plot each orbital element separately
# plt.figure(figsize=(12, 12))

cartesian_converted = []

for i in range(orbital_elements.shape[0]):
    a, e, i, raan, w, nu = orbital_elements[i]
    r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    cartesian_converted.append(np.concatenate((r, v)))

cartesian_converted = np.array(cartesian_converted)

# Plot the converted Cartesian data
visualization.plot_3d_orbit_from_np(cartesian_converted)

# Relative path to the .npy file for existing Cartesian data
relative_path_cartesian = "../datasets/dataset_cartesian.npy"
data_cartesian = np.load(relative_path_cartesian)

# Plot the actual Cartesian data for comparison
visualization.plot_3d_orbit_from_np(data_cartesian)

# %%

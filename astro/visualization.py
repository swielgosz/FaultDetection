import matplotlib.pyplot as plt
import numpy as np
from . import constants


def plot_sphere(ax, radius, center=(0, 0, 0), color="b", alpha=1):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def plot_3d_orbit(solution):
    """
    Plot a 3D orbit from either an OdeSolution or a NumPy array.

    Parameters:
        solution: OdeSolution or np.ndarray
            If OdeSolution, the orbit is extracted from the `.y` attribute.
            If np.ndarray, it is assumed to contain Cartesian coordinates.
    """

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Extract Cartesian coordinates based on the solution type
    if hasattr(solution, "t") and hasattr(solution, "y"):
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
    elif isinstance(solution, np.ndarray):
        x, y, z = solution[:, 0], solution[:, 1], solution[:, 2]
    else:
        raise ValueError("Input must be an OdeSolution or a NumPy array.")

    ax.plot3D(x, y, z)

    # Plot the sphere representing Earth
    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0, 0, 0))

    # Get the limits for each axis
    x_limits = [np.min(x), np.max(x)]
    y_limits = [np.min(y), np.max(y)]
    z_limits = [np.min(z), np.max(z)]

    # Determine the range
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    # Find the maximum range
    max_range = max(x_range, y_range, z_range)

    # Set the same limits for all axes
    sphere_center = (0, 0, 0)
    ax.set_xlim([sphere_center[0] - max_range / 2, sphere_center[0] + max_range / 2])
    ax.set_ylim([sphere_center[1] - max_range / 2, sphere_center[1] + max_range / 2])
    ax.set_zlim([sphere_center[2] - max_range / 2, sphere_center[2] + max_range / 2])

    # Add labels and aspect ratio
    ax.set_xlabel("x (m)", size=10)
    ax.set_ylabel("y (m)", size=10)
    ax.set_zlabel("z (m)", size=10)
    ax.set_title("Orbital Path", size=12)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


def compare_orbits(true_orbit, predicted_orbit=None):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Function to extract Cartesian coordinates
    def get_coordinates(orbit):
        if hasattr(orbit, "t") and hasattr(orbit, "y"):  # OdeSolution
            return orbit.y[0], orbit.y[1], orbit.y[2]
        elif isinstance(orbit, np.ndarray):  # NumPy array
            return orbit[:, 0], orbit[:, 1], orbit[:, 2]
        else:
            raise ValueError("Input must be an OdeSolution or a NumPy array.")

    # Plot the true orbit
    x_true, y_true, z_true = get_coordinates(true_orbit)
    ax.plot3D(x_true, y_true, z_true, label="True Orbit", color="b")

    # Plot the predicted orbit if provided
    if predicted_orbit is not None:
        x_pred, y_pred, z_pred = get_coordinates(predicted_orbit)
        ax.plot3D(
            x_pred, y_pred, z_pred, label="Predicted Orbit", color="r", linestyle="--"
        )

    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0, 0, 0))

    # Get the limits for each axis from true orbit
    x_limits_true = [np.min(x_true), np.max(x_true)]
    y_limits_true = [np.min(y_true), np.max(y_true)]
    z_limits_true = [np.min(z_true), np.max(z_true)]

    # Initialize limits with true orbit limits
    x_limits = x_limits_true
    y_limits = y_limits_true
    z_limits = z_limits_true

    # Adjust limits with predicted orbit if provided
    if predicted_orbit is not None:
        x_limits_pred = [np.min(x_pred), np.max(x_pred)]
        y_limits_pred = [np.min(y_pred), np.max(y_pred)]
        z_limits_pred = [np.min(z_pred), np.max(z_pred)]

        # Use the larger range for each axis
        x_limits = [
            min(x_limits_true[0], x_limits_pred[0]),
            max(x_limits_true[1], x_limits_pred[1]),
        ]
        y_limits = [
            min(y_limits_true[0], y_limits_pred[0]),
            max(y_limits_true[1], y_limits_pred[1]),
        ]
        z_limits = [
            min(z_limits_true[0], z_limits_pred[0]),
            max(z_limits_true[1], z_limits_pred[1]),
        ]

    # Determine the range
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    # Find the maximum range
    max_range = max(x_range, y_range, z_range)

    # Set the same limits for all axes
    sphere_center = (0, 0, 0)
    ax.set_xlim([sphere_center[0] - max_range / 2, sphere_center[0] + max_range / 2])
    ax.set_ylim([sphere_center[1] - max_range / 2, sphere_center[1] + max_range / 2])
    ax.set_zlim([sphere_center[2] - max_range / 2, sphere_center[2] + max_range / 2])

    ax.set_xlabel("x", size=10)
    ax.set_ylabel("y", size=10)
    ax.set_zlabel("z", size=10)
    ax.set_title("True and Predicted Orbits")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()

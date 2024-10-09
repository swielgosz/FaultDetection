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


def plot_3d_orbit(cartesian_orbit):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(cartesian_orbit.y[0], cartesian_orbit.y[1], cartesian_orbit.y[2])
    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0, 0, 0))

    # Get the limits for each axis
    x_limits = [np.min(cartesian_orbit.y[0]), np.max(cartesian_orbit.y[0])]
    y_limits = [np.min(cartesian_orbit.y[1]), np.max(cartesian_orbit.y[1])]
    z_limits = [np.min(cartesian_orbit.y[2]), np.max(cartesian_orbit.y[2])]

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

    ax.set_xlabel("x (m)", size=10)
    ax.set_ylabel("y (m)", size=10)
    ax.set_zlabel("z (m)", size=10)
    ax.set_title("Orbital Path of Orbit A")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def plot_3d_orbit_from_np(cartesian_orbit):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(cartesian_orbit[:, 1], cartesian_orbit[:, 2], cartesian_orbit[:, 3])
    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0, 0, 0))

    # Get the limits for each axis
    x_limits = [np.min(cartesian_orbit[:, 1]), np.max(cartesian_orbit[:, 1])]
    y_limits = [np.min(cartesian_orbit[:, 2]), np.max(cartesian_orbit[:, 2])]
    z_limits = [np.min(cartesian_orbit[:, 3]), np.max(cartesian_orbit[:, 3])]

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

    ax.set_xlabel("x (m)", size=10)
    ax.set_ylabel("y (m)", size=10)
    ax.set_zlabel("z (m)", size=10)
    ax.set_title("test")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def compare_orbits(true_orbit, predicted_orbit=None):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Plot the true orbit
    ax.plot3D(
        true_orbit[:, 1],
        true_orbit[:, 2],
        true_orbit[:, 3],
        label="True Orbit",
        color="b",
    )

    # Plot the predicted orbit if provided
    if predicted_orbit is not None:
        ax.plot3D(
            predicted_orbit[:, 1],
            predicted_orbit[:, 2],
            predicted_orbit[:, 3],
            label="Predicted Orbit",
            color="r",
            linestyle="--",
        )

    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0, 0, 0))

    # Get the limits for each axis from true orbit
    x_limits_true = [np.min(true_orbit[:, 1]), np.max(true_orbit[:, 1])]
    y_limits_true = [np.min(true_orbit[:, 2]), np.max(true_orbit[:, 2])]
    z_limits_true = [np.min(true_orbit[:, 3]), np.max(true_orbit[:, 3])]

    # Initialize limits with true orbit limits
    x_limits = x_limits_true
    y_limits = y_limits_true
    z_limits = z_limits_true

    # Adjust limits with predicted orbit if provided
    if predicted_orbit is not None:
        x_limits_pred = [np.min(predicted_orbit[:, 1]), np.max(predicted_orbit[:, 1])]
        y_limits_pred = [np.min(predicted_orbit[:, 2]), np.max(predicted_orbit[:, 2])]
        z_limits_pred = [np.min(predicted_orbit[:, 3]), np.max(predicted_orbit[:, 3])]

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

    ax.set_xlabel("x (m)", size=10)
    ax.set_ylabel("y (m)", size=10)
    ax.set_zlabel("z (m)", size=10)
    ax.set_title("True and Predicted Orbits")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_orbits_scatter(true_orbit, predicted_orbit=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the true orbit as a scatter plot
    ax.scatter(
        true_orbit[:, 1],
        true_orbit[:, 2],
        true_orbit[:, 3],
        label="True Orbit",
        color="b",
        s=10,  # Adjust the size of the scatter points as needed
    )

    # Plot the predicted orbit if provided
    if predicted_orbit is not None:
        ax.scatter(
            predicted_orbit[:, 1],
            predicted_orbit[:, 2],
            predicted_orbit[:, 3],
            label="Predicted Orbit",
            color="r",
            s=10,  # Adjust the size of the scatter points as needed
        )

    # Add a sphere to the plot
    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0, 0, 0))

    # Get the limits for each axis from true orbit
    x_limits_true = [np.min(true_orbit[:, 1]), np.max(true_orbit[:, 1])]
    y_limits_true = [np.min(true_orbit[:, 2]), np.max(true_orbit[:, 2])]
    z_limits_true = [np.min(true_orbit[:, 3]), np.max(true_orbit[:, 3])]

    # Initialize limits with true orbit limits
    x_limits = x_limits_true
    y_limits = y_limits_true
    z_limits = z_limits_true

    # Adjust limits with predicted orbit if provided
    if predicted_orbit is not None:
        x_limits_pred = [np.min(predicted_orbit[:, 1]), np.max(predicted_orbit[:, 1])]
        y_limits_pred = [np.min(predicted_orbit[:, 2]), np.max(predicted_orbit[:, 2])]
        z_limits_pred = [np.min(predicted_orbit[:, 3]), np.max(predicted_orbit[:, 3])]

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

    ax.set_xlabel("x (m)", size=10)
    ax.set_ylabel("y (m)", size=10)
    ax.set_zlabel("z (m)", size=10)
    ax.set_title("True and Predicted Orbits")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()

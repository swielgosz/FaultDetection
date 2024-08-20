import matplotlib.pyplot as plt
import numpy as np
from . import constants
def plot_sphere(ax, radius, center=(0, 0, 0), color='b', alpha=0.5):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_3d_orbit(cartesian_orbit):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(cartesian_orbit.y[0], cartesian_orbit.y[1], cartesian_orbit.y[2]) 
    plot_sphere(ax, radius=constants.RADIUS_EARTH, center=(0,0,0), color='r', alpha=0.3)

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
    sphere_center = (0,0,0)
    ax.set_xlim([sphere_center[0] - max_range/2, sphere_center[0] + max_range/2])
    ax.set_ylim([sphere_center[1] - max_range/2, sphere_center[1] + max_range/2])
    ax.set_zlim([sphere_center[2] - max_range/2, sphere_center[2] + max_range/2])


    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Orbit A")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
# %%

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import visualization, coordinate_conversions
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt
import keras
import time
from keras import metrics
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
print(tf.__version__)

# %%
SEED = 1


def set_seeds(seed=SEED):
    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

# %% Load data from .npy file
#    Data file contains timestep (s), semi-major axis a (km), eccentricity e (-), inclination i (deg), RAAN (deg), argument of periapsis w (deg), and true anomaly nu (deg)
#    Orbital elements recorded for each timestep. tof will be constant in each row for one orbit

# Loading orbital element data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../datasets/dataset_oe.npy")
data_np = np.load(data_path)
data = pd.DataFrame(
    data_np, columns=["t", "a", "e", "i", "RAAN", "w", "nu"]
)  # convert to data frame

nu_ground_truth = data[["nu"]] # nu in degree
t_max = data[["t"]].max().values[0]

# Load cartesian data for plotting
cartesian_data_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")
cartesian_data_np = np.load(cartesian_data_path)

data.describe().transpose()

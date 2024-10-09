# %%

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import visualization, coordinate_conversions
import numpy as np

import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

# %%
SEED = 8

def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# Call the above function with seed value
set_global_determinism(seed=SEED)
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

nu_record = data[["nu"]]
t_max = data[['t']].max().values[0]

# Load cartesian data for plotting
cartesian_data_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")
cartesian_data_np = np.load(cartesian_data_path)

data.describe().transpose()
#%%
# Record non time-varying orbital elements
a = data_np[0, 1]  # Semi-major axis
e = data_np[0, 2]  # Eccentricity
i = data_np[0, 3]  # Inclination
raan = data_np[0, 4]  # RAAN
w = data_np[0, 5]  # Argument of periapsis

# Separate features and labels
features = data[["t"]]
labels = data[["nu"]]

# Split data into train and test sets
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Separate features from labels
train_features = train_data[["t"]]
train_labels = train_data[["nu"]]
test_features = test_data[["t"]]
test_labels = test_data[["nu"]]

# Normalize features and labels using only the training data
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

train_features = pd.DataFrame(feature_scaler.fit_transform(train_features), columns=train_features.columns)

train_labels = pd.DataFrame(
    label_scaler.fit_transform(train_labels), columns=train_labels.columns
)

test_features = pd.DataFrame(
    feature_scaler.transform(test_features), columns=test_features.columns
)

test_labels = pd.DataFrame(
    label_scaler.transform(test_labels), columns=test_labels.columns
)

# Define the periodic loss function
def periodic_loss(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    # Minimize the angular error considering the wraparound at 360 degrees
    error = tf.minimum(error, 360.0 - error)  
    return tf.reduce_mean(error)

# Define and compile the model
def build_and_compile_model(input_shape_):
    model = keras.Sequential(
        [
            # tf.keras.Input(shape=(input_shape,)),
            tf.keras.layers.Flatten(input_shape=(input_shape_,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            # layers.Dense(32, activation="relu"),
            # layers.Dense(32, activation="relu"),
            # layers.Dense(16, activation="relu"),
            # layers.Dense(8, activation="relu"),
            # layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dense(1),
        ]
    )
    model.compile(loss=periodic_loss, optimizer=tf.keras.optimizers.Adam(0.001))
    return model


input_shape_ = train_features.shape[1]
dnn_model = build_and_compile_model(input_shape_)
dnn_model.summary()

# %%
BATCH_SIZE = 500
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2, # if validation split is low (~0.1), results are very bad (error is ~100 deg instead of ~50 at beginning/end of orbit)
    verbose=1,
    epochs=100,
    batch_size=BATCH_SIZE,
)

# %%


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

# Create time array and reshape to 2D array
time_ = np.linspace(0, t_max, 10000).reshape(-1, 1)
time_ = time_[:-1]

# Convert to DataFrame with the same column name as used during scaling
time_df = pd.DataFrame(time_, columns=["t"])

# Normalize the time data
time_scaled = pd.DataFrame(
    feature_scaler.transform(time_df), columns=["t"]
)
print("Sample of time scaled data:", feature_scaler.inverse_transform(time_scaled))
print(t_max)
predictions_scaled = dnn_model.predict(time_scaled)
predictions = label_scaler.inverse_transform(predictions_scaled)

# Convert predictions to Cartesian coordinates
cartesian_coords = []
for pred in predictions:
    nu = pred[0]
    r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    cartesian_coords.append(np.concatenate([r, v]))
cartesian_coords = np.array(cartesian_coords)

# Concatenate time data with predictions
data_with_time = data_np[:, 0]  # Extract time column
predictions_with_time = np.column_stack((data_with_time, cartesian_coords))

# Plot orbits
visualization.compare_orbits(cartesian_data_np, predictions_with_time)
# %%
error = predictions - nu_record
# for idx in range(len(error)):
#     if abs(error[idx]) > 180:
#         error[idx] = 360 - abs(error[idx])

plt.figure(figsize=(10, 6))
plt.scatter(nu_record, error, alpha=0.5, edgecolors='k')
plt.xlabel('True Values (nu)')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs. True Values')
plt.grid(True)
plt.show()

time = np.linspace(0, t_max, len(predictions))  # Create a time array if not already available

plt.figure(figsize=(10, 6))
plt.plot(time, error, label='Prediction Error')
plt.xlabel('Time')
plt.ylabel('Prediction Error')
plt.title('Prediction Error Over Time')
plt.legend()
plt.grid(True)
plt.show()
# %%
train_loss = dnn_model.evaluate(train_features, train_labels, verbose=0)
test_loss = dnn_model.evaluate(test_features, test_labels, verbose=0)

print(f'Training Loss: {train_loss}')
print(f'Test Loss: {test_loss}')
# %%

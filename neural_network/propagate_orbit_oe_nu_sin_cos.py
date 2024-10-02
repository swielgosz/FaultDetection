# %%

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import visualization, coordinate_conversions
import numpy as np

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)


# %% Load data from .npy file
# Data file contains timestep (s), semi-major axis a (km), eccentricity e (-), inclination i (deg), RAAN (deg), argument of periapsis w (deg), and true anomaly nu (deg)

# Loading orbital element data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../datasets/dataset_oe.npy")
data_np = np.load(data_path)
data = pd.DataFrame(
    data_np, columns=["t", "a", "e", "i", "RAAN", "w", "nu"]
)  # convert to data frame
nu_record = data[["nu"]]
t_max = data[['t']].max().values[0]

# Convert true anomaly (nu) to sine and cosine components
data["sin_nu"] = np.sin(np.deg2rad(data["nu"]))  # convert to radians first
data["cos_nu"] = np.cos(np.deg2rad(data["nu"]))

# Load cartesian data for plotting
cartesian_data_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")
cartesian_data_np = np.load(cartesian_data_path)

# Record non time-varying orbital elements
a = data_np[0, 1]  # Semi-major axis
e = data_np[0, 2]  # Eccentricity
i = data_np[0, 3]  # Inclination
raan = data_np[0, 4]  # RAAN
w = data_np[0, 5]  # Argument of periapsis

# Separate features and labels
features = data[["t"]]
labels = data[["sin_nu", "cos_nu"]]  # using sine and cosine of true anomaly

num_samples = 100  # Adjust this based on your needs

# Select samples from the beginning and end
start_samples = data.iloc[:num_samples]
end_samples = data.iloc[-num_samples:]

# Get the remaining samples from the middle
remaining_data = data.iloc[num_samples:-num_samples]

# Randomly sample from the remaining data
train_middle_samples = remaining_data.sample(frac=0.8, random_state=0)

# Combine selected samples
train_data = pd.concat([start_samples, end_samples, train_middle_samples])

# Split data into train and test sets
# train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Separate features from labels
train_features = train_data[["t"]]
train_labels = train_data[["sin_nu", "cos_nu"]]
test_features = test_data[["t"]]
test_labels = test_data[["sin_nu", "cos_nu"]]

# Normalize features and labels using only the training data
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
train_features = pd.DataFrame(feature_scaler.fit_transform(train_features), columns=train_features.columns)
test_features = pd.DataFrame(feature_scaler.transform(test_features), columns=test_features.columns)

# No need to scale sin and cos since they are already between [-1, 1]

# Define and compile the model
def build_and_compile_model(input_shape_):
    model = keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(input_shape_,)),
            layers.Dense(16, activation="tanh"),
            layers.Dense(16, activation="tanh"),
            layers.Dense(16, activation="tanh"),
            layers.Dense(2),  # Predicting two values: sin_nu and cos_nu
        ]
    )
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    return model

input_shape_ = train_features.shape[1]
dnn_model = build_and_compile_model(input_shape_)
dnn_model.summary()

# %% Train the model
BATCH_SIZE = 1000
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1,
    epochs=500,
    batch_size=BATCH_SIZE,
)

def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot loss
plot_loss(history)

# %% Predictions and conversion back to nu
time_ = np.linspace(0, t_max, 10000).reshape(-1, 1)
time_ = time_[:-1]
time_df = pd.DataFrame(time_, columns=["t"])
time_scaled = pd.DataFrame(feature_scaler.kj(time_df), columns=["t"])

predictions_scaled = dnn_model.predict(time_scaled)
# predictions_scaled = dnn_model.predict(train_features)
sin_nu_pred, cos_nu_pred = predictions_scaled[:, 0], predictions_scaled[:, 1]

# Convert sin and cos predictions back to true anomaly
predictions_nu = np.rad2deg(np.arctan2(sin_nu_pred, cos_nu_pred))  # get nu back in degrees
predictions_nu = np.where(predictions_nu < 0, predictions_nu + 360, predictions_nu)

# Convert predictions to Cartesian coordinates
cartesian_coords = []
for nu in predictions_nu:
    r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    cartesian_coords.append(np.concatenate([r, v]))
cartesian_coords = np.array(cartesian_coords)

# Concatenate time data with predictions
data_with_time = data_np[:, 0]  # Extract time column
predictions_with_time = np.column_stack((data_with_time, cartesian_coords))

# Plot orbits
#%%
visualization.compare_orbits(cartesian_data_np, predictions_with_time)

nu_record_np = nu_record.values.flatten()

# Check lengths
assert len(predictions_nu) == len(nu_record_np), "Length mismatch between predictions and true values."
error = nu_record_np - predictions_nu

# Adjust error values for wrapping around angles
for idx in range(len(error)):
    if abs(error[idx]) > 180:
        error[idx] = 360 - abs(error[idx])

plt.figure(figsize=(10, 6))
plt.scatter(nu_record, error, alpha=0.5, edgecolors='k')
plt.xlabel('True Values (nu)')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs. True Values')
plt.grid(True)
plt.show()

time = np.linspace(0, t_max, len(predictions_nu))  # Create a time array if not already available
#%%
plt.figure(figsize=(10, 6))
plt.plot(time, error, label='Prediction Error')
plt.xlabel('Time')
plt.ylabel(r'$\nu_{\text{actual}} - \nu_{\text{predicted}}$ [deg]')
plt.title('Prediction Error')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(time,nu_record_np, label='Actual nu', color='b')
plt.plot(time, predictions_nu, label='Predicted nu', color='r', linestyle='--')
# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('True Anomaly [deg]')
plt.title('Actual vs Predicted True Anomaly')
plt.legend()
# Add grid for better readability
plt.grid(True)

# Show plot
plt.show()

# %%
train_loss = dnn_model.evaluate(train_features, train_labels, verbose=0)
test_loss = dnn_model.evaluate(test_features, test_labels, verbose=0)

print(f'Training Loss: {train_loss}')
print(f'Test Loss: {test_loss}')

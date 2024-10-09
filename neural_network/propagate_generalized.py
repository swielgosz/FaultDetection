#%%
import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from astro import coordinate_conversions, constants, astro_calcs, visualization

#%%
# Load data from .npy file
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../datasets/dataset_generalized.npy")
data_np = np.load(data_path, allow_pickle=True)

# Prepare input features (initial state + time) and output labels (final state)
features = []
labels = []

# Create features and labels
# Convert degrees to radians to remove need for scaling
for entry in data_np:
    initial_state = entry["initial_state"]  # a, e, i, RAAN, w, nu
    trajectory = entry["trajectory"]  # time, a, e, i, RAAN, w, nu
    cos_nu0 = np.cos(np.deg2rad(initial_state[5]))
    sin_nu0 = np.sin(np.deg2rad(initial_state[5]))

    # For each time step in the trajectory
    for state in trajectory:
        time = state[0]
        nu = state[6]
        if np.isnan(nu):
            continue
        sin_nu = np.sin(np.deg2rad(nu))
        cos_nu = np.cos(np.deg2rad(nu))

        features.append(
            np.array(
                [
                    time,  # Time as a scalar
                    initial_state[0],  # a as a scalar
                    initial_state[1],  # e as a scalar
                    np.deg2rad(initial_state[2]),  # i in radians
                    np.deg2rad(initial_state[3]),  # RAAN in radians
                    np.deg2rad(initial_state[4]),  # w in radians
                    cos_nu0,  # cos(nu0) as a scalar
                    sin_nu0,  # sin(nu0) as a scalar
                ]
            )
        )

        labels.append(np.array([sin_nu, cos_nu]))  # Ensure labels are arrays too

# Convert features and labels to DataFrames
features = pd.DataFrame(features)
labels = pd.DataFrame(labels)

# Define train and test features and labels
train_features = features.sample(frac=0.8, random_state=0)
test_features = features.drop(train_features.index)

train_labels = labels.iloc[train_features.index]
test_labels = labels.iloc[test_features.index]

# Normalize time and semi-major axis
feature_scaler = MinMaxScaler()
train_features.iloc[:, [0, 1]] = feature_scaler.fit_transform(
    train_features.iloc[:, [0, 1]]
)
test_features.iloc[:, [0, 1]] = feature_scaler.transform(test_features.iloc[:, [0, 1]])

# Define and compile the model
def build_and_compile_model(input_shape_):
    model = keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(input_shape_,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(7),  # Predicting two values: sin_nu and cos_nu
        ]
    )
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    return model

input_shape_ = train_features.shape[1]
dnn_model = build_and_compile_model(input_shape_)
dnn_model.summary()

# Train the model
BATCH_SIZE = 1000
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1,
    epochs=50,
    batch_size=BATCH_SIZE,
)

# Plot training loss
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

# %%
# Test with specified conditions
def test_model():
    # Define the initial conditions
    a = 15000 + constants.RADIUS_EARTH  # Semi-major axis in km
    e = 0.3  # Eccentricity
    i = np.deg2rad(10.0)  # Inclination in radians
    raan = 0.0  # RAAN in radians
    w = np.deg2rad(10.0)  # Argument of periapsis in radians
    nu = 0.0  # True anomaly in radians
    tof = astro_calcs.calculate_orbital_period(a, mu=constants.MU_EARTH)

    # Generate 1000 time steps from 0 to tof
    time_steps = np.linspace(0, tof, 10000)

    # Prepare features for all time steps
    features = []
    for time in time_steps:
        cos_nu0 = np.cos(nu)
        sin_nu0 = np.sin(nu)

        feature_vector = np.array([time, a, e, np.rad2deg(i), np.rad2deg(raan), np.rad2deg(w), cos_nu0, sin_nu0])
        features.append(feature_vector)

    features = np.array(features)
    features_normalized = feature_scaler.transform(features)  # Normalize the inputs

    # Make predictions for all time steps
    predictions_scaled = dnn_model.predict(features_normalized)
    sin_nu_pred, cos_nu_pred = predictions_scaled[:, 0], predictions_scaled[:, 1]
    predictions_nu = np.rad2deg(np.arctan2(sin_nu_pred, cos_nu_pred))  # get nu back in degrees
    predictions_nu = np.where(predictions_nu < 0, predictions_nu + 360, predictions_nu)

    # Concatenate predictions_scaled (all but last two columns) with predictions_nu
    predictions_combined = np.concatenate([predictions_scaled[:, :-2], predictions_nu[:, np.newaxis]], axis=1)

    # Combine with time steps
    time_vector = time_steps[:, np.newaxis]  # Reshape for concatenation
    predictions_with_time = np.hstack((time_vector, predictions_combined))  # Add time as the first column

    return predictions_with_time

# Get predictions
predictions = test_model()
print("Predicted (a, e, i, RAAN, w, cos(nu), sin(nu)) for 10000 time steps:")
print(predictions)

#%%
# Load cartesian data for plotting
cartesian_data_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")
cartesian_data_np = np.load(cartesian_data_path)
visualization.compare_orbits(cartesian_data_np, predictions)
# # Predictions
# predictions_scaled = dnn_model.predict(test_features)

# # Extract predictions
# predicted_orbital_elements = predictions_scaled  # This will be the predicted (a, e, i, RAAN, w, cos(nu), sin(nu))

# # Convert predictions to Cartesian coordinates for analysis
# cartesian_coords = []
# for idx in range(len(predicted_orbital_elements)):
#     a, e, i, raan, w, cos_nu, sin_nu = predicted_orbital_elements[idx]
#     nu_pred = np.rad2deg(np.arctan2(sin_nu, cos_nu))  # Convert back to true anomaly
#     r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu_pred)
#     cartesian_coords.append(np.concatenate([r, v]))
# cartesian_coords = np.array(cartesian_coords)

# # Evaluation
# train_loss = dnn_model.evaluate(train_features, train_labels, verbose=0)
# test_loss = dnn_model.evaluate(test_features, test_labels, verbose=0)

# print(f'Training Loss: {train_loss}')
# print(f'Test Loss: {test_loss}')
# %%

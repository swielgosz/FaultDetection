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
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(2),  # Predicting two values: sin_nu and cos_nu
        ]
    )
    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
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
    constant_oe = []
    for time in time_steps:
        cos_nu0 = np.cos(nu)
        sin_nu0 = np.sin(nu)

        feature_vector = np.array([time, a, e, i, raan, w, cos_nu0, sin_nu0])
        features.append(feature_vector)
        constant_oe_vector = np.array(
            [a, e, np.rad2deg(i), np.rad2deg(raan), np.rad2deg(w)]
        )
        constant_oe.append(constant_oe_vector)

    features = np.array(features)
    features = pd.DataFrame(features)
    constant_oe = np.array(constant_oe)

    features_normalized = feature_scaler.transform(features.iloc[:, [0, 1]])
    features_normalized = pd.DataFrame(
        features_normalized, columns=[features.columns[0], features.columns[1]]
    )
    if features.shape[1] > 2:  # Check if there are additional columns
        features_normalized = pd.concat(
            [features_normalized, features.iloc[:, 2:]], axis=1
        )

    # Predict true anomaly
    predictions_scaled = dnn_model.predict(features_normalized)
    sin_nu_pred, cos_nu_pred = predictions_scaled[:, 0], predictions_scaled[:, 1]
    predictions_nu = np.rad2deg(
        np.arctan2(sin_nu_pred, cos_nu_pred)
    )  # get nu back in degrees
    predictions_nu = np.where(predictions_nu < 0, predictions_nu + 360, predictions_nu)

    # Concatenate predictions
    predictions = np.concatenate((constant_oe, predictions_nu[:, np.newaxis]), axis=1)
    time_vector = time_steps[:, np.newaxis]
    predictions_with_time = np.hstack((time_vector, predictions))

    return predictions_with_time


# Get predictions
predictions = test_model()
# %% Visualize results


# Convert predictions to Cartesian coordinates for plotting
def convert_predictions_to_cartesian(predicted_orbital_elements):
    cartesian_coords = []
    for idx in range(len(predicted_orbital_elements)):
        time, a, e, i, raan, w, nu_pred = predicted_orbital_elements[idx]
        # print(time)
        print(nu_pred)
        r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu_pred)
        cartesian_coords.append(
            np.concatenate([[time], r, v])
        )  # Concatenate position and velocity vectors
    return np.array(cartesian_coords)


cartesian_predictions = convert_predictions_to_cartesian(predictions)
cartesian_data_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")
cartesian_data_np = np.load(cartesian_data_path)
visualization.compare_orbits(cartesian_data_np, cartesian_predictions)


# Calculate the prediction error of true anomaly
oe_data_path = os.path.join(script_dir, "../datasets/dataset_oe.npy")
oe_data_np = np.load(oe_data_path)
nu_record = oe_data_np[
    :, -1
]  # Assuming the last column in predictions is the predicted nu
error = predictions[:, -1] - nu_record

# Plot the prediction error against true values
plt.figure(figsize=(10, 6))
plt.scatter(nu_record, error, s=2)  # Use scatter to visualize the error
plt.xlabel("True Values (nu) [Degrees]")
plt.ylabel("Prediction Error [Degrees]")
plt.title("Prediction Error vs. True Values")
plt.legend()
plt.grid(True)
plt.show()

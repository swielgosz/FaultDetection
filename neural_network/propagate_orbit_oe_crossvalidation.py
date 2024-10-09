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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print(tf.__version__)

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
t_max = data[["t"]].max().values[0]
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
labels = data[["nu"]]

# %% Define cross-validation parameters
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=0)
fold = 1
fold_results = []

# Define and compile the model
def build_and_compile_model(input_shape_):
    model = keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(input_shape_,)),
            layers.Dense(16, activation="tanh"),
            layers.Dense(16, activation="tanh"),
            layers.Dense(2, activation="tanh"),
            layers.Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    return model

input_shape_ = features.shape[1]

# Perform k-fold cross-validation
for train_index, val_index in kf.split(data):
    train_data, val_data = data.iloc[train_index], data.iloc[val_index]
    
    # Separate features and labels
    train_features = train_data[["t"]]
    train_labels = train_data[["nu"]]
    val_features = val_data[["t"]]
    val_labels = val_data[["nu"]]
    
    # Normalize features and labels
    feature_scaler = MinMaxScaler()
    label_scaler = MinMaxScaler()

    train_features = pd.DataFrame(feature_scaler.fit_transform(train_features), columns=train_features.columns)
    val_features = pd.DataFrame(feature_scaler.transform(val_features), columns=val_features.columns)
    train_labels = pd.DataFrame(label_scaler.fit_transform(train_labels), columns=train_labels.columns)
    val_labels = pd.DataFrame(label_scaler.transform(val_labels), columns=val_labels.columns)
    
    # Build and compile the model
    model = build_and_compile_model(input_shape_)
    
    # Train the model
    history = model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=100,
        batch_size=1000,
        verbose=0
    )
    
    # Evaluate on validation set
    val_predictions = model.predict(val_features)
    val_labels = label_scaler.inverse_transform(val_labels)
    val_predictions = label_scaler.inverse_transform(val_predictions)
    
    # Calculate error
    val_error = mean_squared_error(val_labels, val_predictions)
    fold_results.append(val_error)
    
    print(f"Fold {fold} - Validation MSE: {val_error:.4f}")
    fold += 1

# Average performance across all folds
print(f"Average Validation MSE: {np.mean(fold_results):.4f}")

# %% Plot loss for the last fold
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% Predict and plot final results (using the last fold's model)
# Use the last trained model to make predictions
final_model = build_and_compile_model(input_shape_)
final_model.fit(
    features,
    labels,
    epochs=100,
    batch_size=100,
    verbose=0
)

# Create time array and reshape to 2D array
time_ = np.linspace(0, t_max, 10000).reshape(-1, 1)

# Convert to DataFrame with the same column name as used during scaling
time_df = pd.DataFrame(time_, columns=["t"])

# Normalize the time data
time_scaled = pd.DataFrame(feature_scaler.transform(time_df), columns=["t"])
predictions_scaled = final_model.predict(time_scaled)
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
#%%
visualization.compare_orbits(cartesian_data_np, predictions_with_time)
visualization.compare_orbits(predictions_with_time)

# %% Plot prediction errors
error = predictions - nu_record
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [True Anomaly (deg)]")
_ = plt.ylabel("Count")

plt.figure(figsize=(10, 6))
plt.scatter(nu_record, error, alpha=0.5, edgecolors="k")
plt.xlabel("True Values (nu)")
plt.ylabel("Prediction Error")
plt.title("Prediction Error vs. True Values")
plt.grid(True)
plt.show()

time = np.linspace(
    0, t_max, len(predictions)
)  # Create a time array if not already available

plt.figure(figsize=(10, 6))
plt.plot(time, error, label="Prediction Error")
plt.xlabel("Time")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Over Time")
plt.legend()
plt.grid(True)
plt.show()

# %% Evaluate the final model on the entire dataset
train_loss = final_model.evaluate(features, labels, verbose=0)
print(f"Final Model Loss: {train_loss}")

# %%

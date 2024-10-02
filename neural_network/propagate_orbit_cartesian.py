# %%

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astro import visualization
import numpy as np

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
print(tf.__version__)

runflag = 2
# %% Load data from .npy file
#    Data file contains timestep (s), semi-major axis a (km), eccentricity e (-), inclination i (deg), RAAN (deg), argument of periapsis w (deg), and true anomaly nu (deg)
#    Orbital elements recorded for each timestep. tof will be constant in each row for one orbit

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../datasets/dataset_cartesian.npy")
data_np = np.load(data_path)
data = pd.DataFrame(data_np, columns=["t", "x", "y", "z", "vx", "vy", "vz"])
data.tail()
t_max = data['t'].max()
# Separate features and labels
features = data[["t"]]
labels = data[["x", "y", "z", "vx", "vy", "vz"]]
data.describe().transpose()
#%%
# Normalize features and labels
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

# Fit scaler on all data (this assumes all columns should be scaled)
features_scaled = pd.DataFrame(feature_scaler.fit_transform(features), columns=features.columns)
labels_scaled = pd.DataFrame(label_scaler.fit_transform(labels), columns=labels.columns)


# Combine the scaled features and labels
data_scaled = pd.concat([features_scaled, labels_scaled], axis=1)
data = data_scaled


train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# %% Inspect the data

train_data.describe().transpose()

# %% Split features from labels

if runflag == 1:
    train_features = train_data[["t"]]
    test_features = test_data[["t"]]
    # train_features = features_with_initial_conditions
else:
    train_features = train_data
    test_features = test_data

train_labels = train_data[["x", "y", "z", "vx", "vy", "vz"]]
test_labels = test_data[["x", "y", "z", "vx", "vy", "vz"]]

print("Training features shape:", train_features.shape)
print("Training labels shape:", train_labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
# %%

BATCH_SIZE = 1000


def build_and_compile_model(input_shape_):
    model = keras.Sequential(
        [   
            tf.keras.layers.Flatten(input_shape=(input_shape_,)),
            tf.keras.layers.Dense(40, activation="tanh"),
            tf.keras.layers.Dense(40, activation="tanh"),
            tf.keras.layers.Dense(40, activation="tanh"),
            tf.keras.layers.Dense(6),
        ]
    )

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return model

input_shape_ = train_features.shape[1]
dnn_model = build_and_compile_model(input_shape_)
dnn_model.summary()

# %%
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.3,
    verbose=1,
    epochs=100,
    batch_size=BATCH_SIZE,
)

# history = dnn_model.fit(train_dataset,epochs=100)
# %%
# dnn_model.evaluate(data['t'])

def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

# %%
# Plotting predicted vs true orbit
# Make predictions on test data
# test_features=data['t']
# print(test_features)
predictions_scaled = dnn_model.predict(test_features)
predictions = label_scaler.inverse_transform(predictions_scaled)
print('Predictions')
print(predictions)
# Extract time data for test features
test_times = test_data['t'].values

# Concatenate time data with predictions
predictions_with_time = np.column_stack((test_times, predictions))
# Plot orbits
visualization.compare_orbits(data_np,predictions_with_time)
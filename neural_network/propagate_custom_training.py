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

# %% Prepare data 
# Record non time-varying orbital elements
a = data_np[0, 1]  # Semi-major axis
e = data_np[0, 2]  # Eccentricity
i = data_np[0, 3]  # Inclination
raan = data_np[0, 4]  # RAAN
w = data_np[0, 5]  # Argument of periapsis

# Separate features and labels
features = data[["t"]]
labels = data[["nu"]]

# First split: 20% test, 80% train (we'll split train again for validation)
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=0
)

# Second split on the 80% train data: 80% train, 20% validation
train_features, val_features, train_labels, val_labels = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=0
)

# Normalize the data
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

train_features = feature_scaler.fit_transform(train_features)
train_labels = label_scaler.fit_transform(train_labels)

val_features = feature_scaler.transform(val_features)
val_labels = label_scaler.transform(val_labels)

test_features = feature_scaler.transform(test_features)
test_labels = label_scaler.transform(test_labels)

#%% Model
def periodic_activation(x):
    scale_max = 1.0
    return tf.math.floormod(x, scale_max)

inputs = keras.Input(shape=(train_features.shape[1],), name="nu")

x1 = layers.Dense(16, activation="relu")(inputs)
x2 = layers.Dense(16, activation="relu")(x1)
x3 = layers.Dense(16, activation="relu")(x2)
x4 = layers.Dense(16, activation="relu")(x3)
outputs = layers.Dense(1, activation=periodic_activation)(x4)
model = keras.Model(inputs=inputs, outputs=outputs)

#Instantiate an optimizer
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

#Instantiate a loss function
loss_fn = keras.losses.mean_absolute_error

# Prepare the training dataset
batch_size = 1000
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
train_dataset = train_dataset.batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
val_dataset = val_dataset.batch(batch_size)

# Define metrics for tracking performance
train_metric = metrics.MeanAbsoluteError()
val_metric = metrics.MeanAbsoluteError()

# Train the model
epochs = 100

for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch_train, training=True)
            # loss_value = loss_fn(y_batch_train, y_pred)
            loss_value = tf.reduce_mean(loss_fn(y_batch_train, y_pred))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_metric.update_state(y_batch_train, y_pred)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # End-of-epoch training metrics
    train_mae = train_metric.result()
    print("Training MAE over epoch: %.4f" % (float(train_mae),))

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_y_pred = model(x_batch_val, training=False)
        # Update val metrics
        val_metric.update_state(y_batch_val, val_y_pred)
    val_mae = val_metric.result()

    print("Validation acc: %.4f" % (float(val_mae),))
    print("Time taken: %.2fs" % (time.time() - start_time))
# %% Visualize results
# Create time array and reshape to 2D array
time_ = np.linspace(0, t_max, len(data[["t"]])).reshape(-1, 1)

# Convert to DataFrame with the same column name as used during scaling
time_df = pd.DataFrame(time_, columns=["t"])

# Normalize the time data
time_scaled = pd.DataFrame(feature_scaler.transform(time_df), columns=["t"])
predictions_scaled = model.predict(time_scaled)
predictions = label_scaler.inverse_transform(predictions_scaled)

# Calculate error between true and predicted values
error = predictions - nu_ground_truth.values

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
# %% Visualize error
error = predictions - nu_ground_truth

plt.figure(figsize=(10, 6))
plt.scatter(nu_ground_truth, error, s=2)
plt.xlabel("True Values (nu)")
plt.ylabel("Prediction Error")
plt.title("Prediction Error vs. True Values")
plt.ylim(-10, 10)  # Set y-axis limits
plt.grid(True)
plt.show()

# %% Evaluate losses
# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn,metrics = ['mean_absolute_error'])
train_loss = model.evaluate(train_features, train_labels, verbose=0)
test_loss = model.evaluate(test_features, test_labels, verbose=0)
print("Train loss:", train_loss)
print("Test loss:", test_loss)
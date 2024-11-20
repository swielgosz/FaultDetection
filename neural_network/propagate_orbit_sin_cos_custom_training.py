# %%

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

data["sin_nu"] = np.sin(np.deg2rad(data["nu"]))  # Convert to radians first
data["cos_nu"] = np.cos(np.deg2rad(data["nu"]))  # Convert to radians first

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
features = data[["t"]]  # Using only time as a feature
labels = data[["sin_nu", "cos_nu"]]  # Use sin and cos of nu as targets

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
val_features = feature_scaler.transform(val_features)
test_features = feature_scaler.transform(test_features)

#%% Model

inputs = keras.Input(shape=(train_features.shape[1],))
x = tf.keras.layers.Flatten()(inputs)
x = layers.Dense(32, activation="tanh")(x)
x = layers.Dense(16, activation="tanh")(x)
x = layers.Dense(8, activation="tanh")(x)
outputs = layers.Dense(2)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

#Instantiate an optimizer
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

#Instantiate a loss function
loss_fn = keras.losses.mean_squared_error

# Prepare the training dataset
batch_size = 1000
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
train_dataset = train_dataset.batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
val_dataset = val_dataset.batch(batch_size)

# Define metrics for tracking performance
train_metric = metrics.MeanSquaredError()
val_metric = metrics.MeanSquaredError()

# Define learning rate update 
def manual_reduce_lr_on_plateau(current_loss, optimizer, best_loss, wait, factor=0.2, patience=100, min_lr=1e-6):
    if current_loss < best_loss:
        best_loss = current_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            lr = float(optimizer.learning_rate.numpy())
            new_lr = max(lr * factor, min_lr)
            optimizer.learning_rate.assign(new_lr)
            print(f"Reduced learning rate to {new_lr:.6f}")
            wait = 0
    return best_loss, wait

# Initialize variables
best_loss = float("inf")
wait = 0

@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y, y_pred)
        # loss_value = tf.reduce_mean(loss_fn(y, y_pred))
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Update training metric.
    train_metric.update_state(y, y_pred)
    return loss_value

@tf.function
def test_step(x, y):
    val_y_pred = model(x, training=False)
    val_metric.update_state(y, val_y_pred)


# Train the model
epochs = 500

for epoch in range(epochs):
    print(epoch)
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
     
        # Log every 200 batches.
        # if step % 200 == 0:
        #     print(
        #         "Training loss (for one batch) at step %d: %.4f"
        #         % (step, float(loss_value))
        #     )
        #     print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # End-of-epoch training metrics
    train_mse = train_metric.result()
    train_metric.reset_state()
    # print("Training MAE over epoch: %.4f" % (float(train_mae),))

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)
    val_mse = val_metric.result()
    val_metric.reset_state()
    best_loss, wait = manual_reduce_lr_on_plateau(val_mse, optimizer, best_loss, wait)
    # print("Validation acc: %.4f" % (float(val_mae),))
    # print("Time taken: %.2fs" % (time.time() - start_time))
# %% Visualize results
# Create time array and normalize it
time_ = np.linspace(0, t_max, len(data[["t"]])).reshape(-1, 1)
time_df = pd.DataFrame(time_, columns=["t"])
time_scaled = pd.DataFrame(feature_scaler.transform(time_df), columns=["t"])

# Predict sin and cos of nu
predictions_scaled = model.predict(time_scaled)
predictions = predictions_scaled  # Already scaled between [-1, 1]

# Convert predictions back to true anomaly
pred_sin_nu, pred_cos_nu = predictions[:, 0], predictions[:, 1]
pred_nu = np.rad2deg(np.arctan2(pred_sin_nu, pred_cos_nu))  # Convert sin and cos to nu
pred_nu = pred_nu % 360.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# Calculate error between true and predicted values
error = pred_nu - nu_ground_truth.values.flatten()

# Visualize the error
plt.scatter(nu_ground_truth, error, s=2)
plt.xlabel("True Values (nu)")
plt.ylabel("Prediction Error")
plt.title("Prediction Error vs. True Values")
plt.ylim(-10, 10)  # Set y-axis limits
plt.grid(True)
plt.show()

# Visualize the orbits as before
cartesian_coords = []
for nu in pred_nu:
    r, v = coordinate_conversions.standard_to_cartesian(a, e, i, raan, w, nu)
    cartesian_coords.append(np.concatenate([r, v]))
cartesian_coords = np.array(cartesian_coords)
data_with_time = data_np[:, 0]  # Extract time column
predictions_with_time = np.column_stack((data_with_time, cartesian_coords))
# Plot orbits
visualization.compare_orbits(cartesian_data_np, predictions_with_time)
# %% Evaluate losses
# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn,metrics = ['mean_absolute_error'])
train_loss = model.evaluate(train_features, train_labels, verbose=0)
test_loss = model.evaluate(test_features, test_labels, verbose=0)
print("Train loss:", train_loss)
print("Test loss:", test_loss)

# %% Visualize error
pred_nu = pred_nu % 360.0
error = pred_nu - nu_ground_truth.values.flatten()

plt.figure(figsize=(10, 6))
plt.scatter(nu_ground_truth, error, s=2)
plt.xlabel("True Values (nu)")
plt.ylabel("Prediction Error")
plt.title("Prediction Error vs. True Values")
plt.ylim(-10, 10)  # Set y-axis limits
plt.grid(True)
plt.show()

print(max(pred_nu))
print(min(pred_nu))
# %%
plt.figure(figsize=(10, 6))
plt.plot(
    time_,
    nu_ground_truth,
    label="True True Anomaly (Ground Truth)",
    color="dodgerblue",
)
plt.plot(
    time_,
    pred_nu,
    label="Predicted True Anomaly",
    color="crimson",
    linestyle="dashed",
)
plt.xlabel("Time (Original Scale)")
plt.ylabel("True Anomaly (degrees)")
plt.title("True Anomaly Prediction vs Ground Truth")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("true_anomaly_comparison_unscaled_time.png")
plt.show()
# %%
rmse = np.sqrt(np.mean(error**2))
print(f'RMSE: {rmse:.2f}')
# %%

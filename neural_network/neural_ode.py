# %%
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../datasets/dataset_oe_diffrax.npy")
# data_path = os.path.join(script_dir, "../datasets/dataset_oe_diffrax.npy")
data_np = np.load(data_path)

# Prepare the target (nu) and time features
nu_ground_truth = data_np[:, -1]  # nu in degree
t_max = np.max(data_np[:, 0])

# Convert nu to sin and cos for training labels
sin_nu = np.sin(np.deg2rad(data_np[:, -1]))  # Convert to radians first
cos_nu = np.cos(np.deg2rad(data_np[:, -1]))  # Convert to radians first

# Load cartesian data (optional for plotting)
cartesian_data_path = os.path.join(
    # script_dir, "../datasets/dataset_cartesian_diffrax.npy"
    script_dir,
    "../datasets/dataset_cartesian_diffrax.npy",
)
cartesian_data_np = np.load(cartesian_data_path)

# Record non-time-varying orbital elements
a = data_np[0, 1]  # Semi-major axis
e = data_np[0, 2]  # Eccentricity
i = data_np[0, 3]  # Inclination
raan = data_np[0, 4]  # RAAN
w = data_np[0, 5]  # Argument of periapsis

# Separate features and labels
features = data_np[:, [0]]  # Time as a feature
labels = np.column_stack((sin_nu, cos_nu))  # sin and cos of nu as targets
# # Split the data into training, validation, and test sets
# train_features, test_features, train_labels, test_labels = train_test_split(
#     features, labels, test_size=0.2, random_state=0, shuffle=False)
# train_features, val_features, train_labels, val_labels = train_test_split(
#     train_features, train_labels, test_size=0.2, random_state=0, shuffle=False)

train_features = features
train_labels = labels
# Normalize the data
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

train_features = feature_scaler.fit_transform(train_features)
# val_features = feature_scaler.transform(val_features)
# test_features = feature_scaler.transform(test_features)
ts = train_features  # Time feature
ys = train_labels  # Target (sin_nu and cos_nu)

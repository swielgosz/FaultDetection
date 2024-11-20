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

import sys
import os
import numpy as np
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../datasets/dataset_generalized.npy")
data_np = np.load(data_path, allow_pickle=True)
print(data_np)
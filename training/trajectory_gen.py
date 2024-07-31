from astro_calcs import *
from scipy.integrate import solve_ivp
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import math
import random
from scipy.signal import find_peaks
import csv

def generate_dataset(output_path, _r_range, _v_range, tof_range, num_trajs):
# def generate_dataset(output_path):

    dataset = []

    for i in range(num_trajs):
        _r_gen = np.array([utils.signed_rand_float(_r_range), utils.signed_rand_float(_r_range), utils.signed_rand_float(_r_range)])
        _v_gen = np.array([utils.signed_rand_float(_v_range), utils.signed_rand_float(_v_range), utils.signed_rand_float(_v_range)])
        tof = random.uniform(*tof_range)

        state = calculate_final_state(_r_gen, _v_gen, tof)
        dataset.append([*_r_gen.tolist(), *_v_gen.tolist(), tof, *state])

        print(i/num_trajs * 100, end="\r")

    with open(output_path, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataset)

if __name__ == "__main__":
    generate_dataset("../datasets/dataset.csv", [RADIUS_EARTH + 100 * 10 ** 3, 50 * 10 ** 6], [3 * 10 **3, 10 * 10 ** 3], [10*60, 2*60*60], 1 * 10 ** 5)
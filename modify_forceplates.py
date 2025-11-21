import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import opensim
import requests

# Paths for OpenSimAD
baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

# Paths for your existing data processing
# sys.path.append("./../../")
script_folder, _ = os.path.split(os.path.abspath(__file__))

import utils as ut
from utilsProcessing import lowPassFilter
from utilsPlotting import plot_dataframe
import utilsKinematics

# Configuration for OpenCap Session
session_id = 'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38'
trial_name = 'estocada_izquierda_1'
data_folder = os.path.abspath(os.path.join(script_folder, 'Data', session_id))

# Parameters for OpenSimAD Processing
motion_type = 'sit_to_stand'
case = '0'
time_window = [1.3, 2.2]  # Customize this time window as needed
treadmill_speed = 0
contact_side = 'all'

# Prepare OpenSimAD simulation settings
settings = processInputsOpenSimAD(baseDir, data_folder, session_id, trial_name, 
                                  motion_type, time_window, None, 
                                  treadmill_speed, contact_side)

# Run OpenSimAD tracking to get simulated kinematics
run_tracking(baseDir, data_folder, session_id, settings, case=case, solveProblem=True)

# Load the simulated center of mass acceleration
kinematics_simulated_path = os.path.join(data_folder, 'SimulatedKinematics.npz')
kinematics_simulated = np.load(kinematics_simulated_path)
center_of_mass_acc = kinematics_simulated['center_of_mass_acc']

# Load and process force data from Google Drive
force_dir = os.path.join(data_folder, 'MeasuredForces', trial_name)
force_path = os.path.join(force_dir, f'{trial_name}_forces.mot')
force_gdrive_url = 'https://drive.usercontent.google.com/u/2/uc?id=1-8bc4yZv0Ot8i0D6cczgXXQPBCyFrdCy&export=download'
response = requests.get(force_gdrive_url)
with open(force_path, 'wb') as f:
    f.write(response.content)

forces_structure = ut.storage_to_numpy(force_path)
force_data = forces_structure.view(np.float64).reshape(forces_structure.shape + (-1,))
force_headers = forces_structure.dtype.names

# Synchronize data with cross correlation using optimized kinematics
def cross_corr_improved(y1, y2, multCorrGaussianStd=None, window_size=None, use_fft=True, visualize=False):
    from scipy.signal import correlate, gaussian, resample
    from scipy.fft import fft, ifft

    # Ensure equal length with resampling
    if len(y1) != len(y2):
        n = max(len(y1), len(y2))
        y1 = resample(y1, n)
        y2 = resample(y2, n)

    # Normalize signals
    y1 = (y1 - np.mean(y1)) / np.std(y1)
    y2 = (y2 - np.mean(y2)) / np.std(y2)

    # Calculate cross correlation
    if use_fft:
        corr = ifft(fft(y1) * np.conj(fft(y2))).real
    else:
        corr = correlate(y1, y2, mode='same')

    if window_size:
        window = np.hanning(window_size)
        window_padded = np.zeros_like(corr)
        start = (len(corr) - window_size) // 2
        window_padded[start:start + window_size] = window
        corr = corr * window_padded

    if multCorrGaussianStd:
        gauss_window = gaussian(len(corr), multCorrGaussianStd)
        corr = corr * gauss_window

    shift = len(y1) // 2
    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)
    lag = argmax_corr - shift

    if visualize:
        plt.plot(corr)
        plt.title('Improved Cross Correlation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.axvline(x=argmax_corr, color='r', linestyle='--', label='Max Lag')
        plt.legend()
        plt.show()

    return max_corr, lag

# Downsample and align data for synchronization
force_columns = ut.get_columns(['R_ground_force_vy', 'L_ground_force_vy'], force_headers)
forces_for_cross_corr = np.sum(force_data[:, force_columns], axis=1, keepdims=True)

# Adjust framerate and synchronize
framerate_forces = 1 / np.diff(force_data[:2, 0])[0]
framerate_kinematics = 1 / np.diff(kinematics_simulated['time'][:2])[0]
time_forces_downsamp, forces_for_cross_corr_downsamp = ut.downsample(forces_for_cross_corr,
                                                                     force_data[:, 0],
                                                                     framerate_forces,
                                                                     framerate_kinematics)

# Get synchronization lag
_, lag = cross_corr_improved(np.squeeze(center_of_mass_acc), np.squeeze(forces_for_cross_corr_downsamp), 
                             multCorrGaussianStd=50, window_size=200, use_fft=True, visualize=True)

# Adjust time in force data
force_data_aligned = np.copy(force_data)
force_data_aligned[:, 0] = force_data[:, 0] - lag / framerate_kinematics

# Save synchronized force data
aligned_force_path = os.path.join(force_dir, f'{trial_name}_syncd_forces.mot')
ut.numpy_to_storage(force_headers, force_data_aligned, aligned_force_path, datatype=None)

# Plot to verify alignment
plt.figure()
plt.plot(kinematics_simulated['time'], center_of_mass_acc, label='COM Acceleration')
plt.plot(force_data_aligned[:, 0], forces_for_cross_corr, label='Vertical Ground Reaction Force')
plt.legend()
plt.show()

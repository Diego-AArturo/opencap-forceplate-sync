"""
Force plate integration module for single-leg movements.

This module integrates force plate data with kinematic data from OpenCap,
synchronizing temporal and spatial coordinates, and running inverse dynamics
analysis in OpenSim.

Author: ForcePlateIntegration Team
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim
import requests
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt, butter, filtfilt
from scipy.interpolate import interp1d

sys.path.append("./../../")
script_folder, _ = os.path.split(os.path.abspath(__file__))

import src.utils.utils as ut
from src.utils.utilsProcessing import lowPassFilter
from src.utils.utilsPlotting import plot_dataframe
from src.utils import utilsKinematics


def save_results(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: Data structure to save (dict, list, etc.)
        filename: Output filename (str)
    """
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def lowpass_filter(signal, cutoff=3, fs=60, order=4):
    """
    Apply a Butterworth low-pass filter to a signal.
    
    Args:
        signal: Input signal array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered signal array
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def detect_upforce(signal, threshold_factor=0.1):
    """
    Detect the first significant change in force plate signal.
    
    This function identifies when force starts to increase from baseline,
    indicating initial contact or force application.
    
    Args:
        signal: Force plate signal array
        threshold_factor: Threshold factor based on standard deviation
    
    Returns:
        Index of first force increase, or None if not found
    """
    baseline = np.mean(signal[:10])
    
    for i in range(1, len(signal)):
        if signal[i] > baseline + np.std(signal) * threshold_factor:
            return i
    return None


def detect_interception(signal1, signal2, threshold=0.028, method='first'):
    """
    Detect the intersection (convergence) point between two signals of the same size.
    
    Args:
        signal1: First signal array
        signal2: Second signal array
        threshold: Maximum difference value to consider intersection
        method: Selection method ('first', 'closest', or 'centered')
    
    Returns:
        Index of intersection point, or None if not found
    """
    diff = np.abs(signal1 - signal2)
    close_idxs = np.where(diff < threshold)[0]

    if len(close_idxs) == 0:
        print("No intersection found under threshold.")
        return None

    if method == 'first':
        index = close_idxs[0]
    elif method == 'closest':
        index = np.argmin(diff)
    elif method == 'centered':
        mid = len(signal1) // 2
        index = close_idxs[np.argmin(np.abs(close_idxs - mid))]
    else:
        raise ValueError("Invalid method. Use 'first', 'closest', or 'centered'.")

    return index if index else None


def detect_flat_segment(fall_zone, search_start, min_duration=5, tolerance=2e-3):
    """
    Detect the start of a region where the derivative remains approximately constant (≈ 0).
    
    Args:
        fall_zone: Derivative signal array
        search_start: Starting index for search
        min_duration: Minimum number of samples with derivative ≈ 0
        tolerance: Tolerance to consider derivative as 0
    
    Returns:
        Index of flat segment start, or None if not found
    """
    flat = np.isclose(fall_zone, 0, atol=tolerance).astype(int)

    count = 0
    for i in range(len(flat)):
        if flat[i] == 1:
            count += 1
            if count >= min_duration:
                return search_start + i - min_duration + 1
        else:
            count = 0

    print("No sufficiently long flat zone found.")
    return None


def calculate_spatial_calibration(heel_pos, cop_pos, force_signal, method='weighted_average'):
    """
    Calculate spatial calibration between heel position and center of pressure.
    
    Args:
        heel_pos: Heel positions [N x 3]
        cop_pos: Center of pressure positions [N x 3]
        force_signal: Vertical force signal for weighting
        method: Calibration method ('weighted_average', 'contact_only', 'dynamic')
    
    Returns:
        offset: Correction vector [x, y, z]
    """
    if method == 'weighted_average':
        weights = np.abs(force_signal) / np.max(np.abs(force_signal))
        weights = weights.reshape(-1, 1)
        offset = np.average(heel_pos - cop_pos, axis=0, weights=weights.flatten())
        
    elif method == 'contact_only':
        threshold = 0.1 * np.max(np.abs(force_signal))
        contact_mask = np.abs(force_signal) > threshold
        
        if np.any(contact_mask):
            offset = np.mean(heel_pos[contact_mask] - cop_pos[contact_mask], axis=0)
        else:
            offset = np.zeros(3)
            
    elif method == 'dynamic':
        window_size = min(50, len(force_signal) // 10)
        offsets = []
        
        for i in range(0, len(force_signal) - window_size, window_size // 2):
            window_end = min(i + window_size, len(force_signal))
            window_force = force_signal[i:window_end]
            
            if np.max(np.abs(window_force)) > 0.05 * np.max(np.abs(force_signal)):
                window_offset = np.mean(heel_pos[i:window_end] - cop_pos[i:window_end], axis=0)
                offsets.append(window_offset)
        
        if offsets:
            offset = np.mean(offsets, axis=0)
        else:
            offset = np.zeros(3)
    
    else:
        raise ValueError(f"Unrecognized method: {method}")
    
    return offset


def detect_heel_strike(signal, signal2, threshold_factor=0.6, half_system=True):
    """
    Detect the first heel contact with the ground after takeoff.
    
    Uses maximum fall rate (second derivative) to identify heel strike.
    
    Args:
        signal: Normalized vertical heel signal (primary leg)
        signal2: Normalized vertical heel signal (reference leg)
        threshold_factor: Threshold factor to adjust takeoff sensitivity
        half_system: If True, only analyze first half of signal
    
    Returns:
        Index of heel strike impact, or None if not detected
    """
    if half_system:
        half = int(len(signal) / 2)
        signal = medfilt(signal[:half], kernel_size=5)
        signal2 = signal2[:half]
    else:
        signal = medfilt(signal[:], kernel_size=5)
        signal2 = signal2[:]

    baseline = max(signal[50:]) - min(signal[:])
    threshold = baseline * threshold_factor + min(signal[:])
    rising_window = 10
    takeoff_idx = None
    
    for i in range(30, len(signal) - rising_window):
        future = signal[i+1] - signal[i]
        if np.all(signal[i:i + rising_window] > threshold) and future > 0:
            takeoff_idx = i
            break

    if takeoff_idx is None:
        print("Takeoff not detected.")
        return None

    search_start = takeoff_idx + 20
    search_end = min(takeoff_idx + 150, len(signal))

    deriv = np.gradient(signal)
    fall_zone = deriv[search_start:search_end]
    
    if len(fall_zone) == 0:
        print("Empty fall zone.")
        return None

    impact_idx = np.argmin(fall_zone) + search_start

    if half_system:
        landing_search_start = takeoff_idx + 10
        idx_interception = detect_interception(
            signal[landing_search_start:],
            signal2[landing_search_start:]
        )
        
        if idx_interception is None:
            return impact_idx
            
        local_min_idx = idx_interception + landing_search_start

        if local_min_idx is not None and impact_idx is not None:
            if local_min_idx > impact_idx:
                min_idx = np.argmin(signal[impact_idx:local_min_idx+3]) + impact_idx
            else:
                min_idx = local_min_idx
        elif local_min_idx is None:
            min_idx = impact_idx
        elif impact_idx is None:
            min_idx = local_min_idx
        else:
            print("Could not determine impact.")
            return None
    else:
        min_idx = detect_flat_segment(fall_zone, search_start)

    return impact_idx


# Load lag times from file if it exists
if os.path.exists('lag_times.json'):
    with open('lag_times.json', 'r') as f:
        lag_times = json.load(f)
else:
    lag_times = []


def IntegrateForcepalte_legs(session_id, trial_name, force_gdrive_url, participant_id, legs):
    """
    Integrate force plate data with kinematic data for single-leg movements.
    
    This function:
    1. Downloads force plate data from Google Drive
    2. Synchronizes force plate and kinematic data temporally
    3. Applies spatial corrections to align force vectors with heel positions
    4. Runs inverse dynamics analysis in OpenSim
    5. Saves synchronized force data and results
    
    Args:
        session_id: OpenCap session ID (str)
        trial_name: Name of the trial to process (str)
        force_gdrive_url: Google Drive URL to download force plate data (str)
        participant_id: Participant identifier (str)
        legs: Leg identifier ('R' for right, 'L' for left)
    
    Returns:
        None (saves results to disk)
    
    Note:
        View example session at:
        https://app.opencap.ai/session/9eea5bf0-a550-4fa5-bc69-f5f072765848
    """
    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', participant_id))
    force_dir = os.path.join(data_folder, 'MeasuredForces', trial_name)
    force_path = os.path.join(force_dir, f'{trial_name}_forces.mot')
    os.makedirs(force_dir, exist_ok=True)

    # Download force plate data from Google Drive
    response = requests.get(force_gdrive_url)
    with open(force_path, 'wb') as f:
        f.write(response.content)

    # Configuration parameters
    lowpass_filter_frequency = 30
    filter_force_data = True
    filter_kinematics_data = True

    # Coordinate system transformations
    r_C0_to_forceOrigin_exp_C = {'R': [0, -.191, .083],
                                 'L': [0, -.191, .083]}
    R_forcePlates_to_C = {'R': R.from_euler('y', -90, degrees=True),
                          'L': R.from_euler('y', -90, degrees=True)}

    visualize_synchronization = False
    save_plot = True
    run_ID = True

    def get_columns(list1, list2):
        """Helper function to get column indices."""
        return [i for i, item in enumerate(list2) if item in list1]

    # Download or load kinematic data
    if not os.path.exists(os.path.join(data_folder, 'MarkerData')):
        _, model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name, _ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

    # Initialize kinematics analysis
    kinematics = utilsKinematics.kinematics(
        data_folder, trial_name,
        modelName=model_name,
        lowpass_cutoff_frequency_for_coordinate_values=10
    )

    # Load OpenCap settings and metadata
    opencap_settings = ut.get_main_settings(data_folder, trial_name)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))
    mass = opencap_metadata['mass_kg']
    
    if 'verticalOffset' in opencap_settings.keys():
        vertical_offset = opencap_settings['verticalOffset']
    else:
        vertical_offset = 0

    # Load force plate data
    forces_structure = ut.storage_to_numpy(force_path)
    force_data = forces_structure.view(np.float64).reshape(forces_structure.shape + (-1,))
    force_headers = forces_structure.dtype.names

    # Filter force data if enabled
    if filter_force_data:
        force_data[:, 1:] = lowPassFilter(
            force_data[:, 0], force_data[:, 1:],
            lowpass_filter_frequency, order=4
        )

    # Rotate force plate data to match coordinate system
    quantity = ['ground_force_v', 'ground_torque_', 'ground_force_p']
    directions = ['x', 'y', 'z']
    for q in quantity:
        for leg in ['R', 'L']:
            force_columns = get_columns([leg + '_' + q + d for d in directions], force_headers)
            rot = R_forcePlates_to_C[leg]
            force_data[:, force_columns] = rot.inv().apply(force_data[:, force_columns])

    # Transform center of pressure coordinates
    r_G0_to_C0_expC = np.array((0, -vertical_offset, 0))

    for leg in ['R', 'L']:
        force_columns = get_columns([leg + '_ground_force_p' + d for d in directions], force_headers)
        r_forceOrigin_to_COP_exp_C = force_data[:, force_columns]
        r_G0_to_COP_exp_G = (r_G0_to_C0_expC +
                            r_C0_to_forceOrigin_exp_C[leg] +
                            r_forceOrigin_to_COP_exp_C)
        force_data[:, force_columns] = r_G0_to_COP_exp_G

    # Get center of mass accelerations
    center_of_mass_acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
    center_of_mass_acc_filtered = center_of_mass_acc
    time_kinematics_filtered = kinematics.time

    # Get marker data for heel positions
    marker_data = kinematics.get_marker_dict(
        session_dir=f'Data\\{participant_id}',
        trial_name=trial_name,
        lowpass_cutoff_frequency=4
    )
    time_heel = marker_data['time']
    pos_Rheel = marker_data['markers']['RHeel']
    pos_Lheel = marker_data['markers']['LHeel']
    
    # Initialize lag time (will be calculated later)
    lag_time = 1000
    
    lag_times.append({
        'participante': participant_id,
        'movimiento': trial_name,
        'lag': lag_time
    })
    save_results(lag_times, 'lag_times.json')
    
    # Normalize heel vertical positions
    pos_Rheel_y = (pos_Rheel[:, 1] - np.min(pos_Rheel[:, 1])) / (
        np.max(pos_Rheel[:, 1]) - np.min(pos_Rheel[:, 1])
    )
    pos_Lheel_y = (pos_Lheel[:, 1] - np.min(pos_Lheel[:, 1])) / (
        np.max(pos_Lheel[:, 1]) - np.min(pos_Lheel[:, 1])
    )

    # Filter heel positions
    pos_Rheel_y_filtered = lowPassFilter(
        time_heel, pos_Rheel_y,
        lowpass_cutoff_frequency=3, order=4
    )
    pos_Lheel_y_filtered = lowPassFilter(
        time_heel, pos_Lheel_y,
        lowpass_cutoff_frequency=3, order=4
    )
    pos_Rheel_y_filtered = lowpass_filter(pos_Rheel_y_filtered)
    pos_Lheel_y_filtered = lowpass_filter(pos_Lheel_y_filtered)
    
    # Detect heel strike based on movement type
    if any(word in trial_name for word in ['escalon']):
        if legs == 'R':
            primal_leg_index = detect_heel_strike(
                pos_Rheel_y_filtered, pos_Lheel_y_filtered, half_system=False
            )
        elif legs == 'L':
            primal_leg_index = detect_heel_strike(
                pos_Lheel_y_filtered, pos_Rheel_y_filtered, half_system=False
            )
    else:
        if legs == 'R':
            primal_leg_index = detect_heel_strike(
                pos_Rheel_y_filtered, pos_Lheel_y_filtered
            )
        elif legs == 'L':
            primal_leg_index = detect_heel_strike(
                pos_Lheel_y_filtered, pos_Rheel_y_filtered
            )
        
    # Extract heel X and Z coordinates
    pos_Rheel_x = pos_Rheel[:, 0]
    pos_Lheel_x = pos_Lheel[:, 0]
    pos_Rheel_z = pos_Rheel[:, 2]
    pos_Lheel_z = pos_Lheel[:, 2]
    
    # Prepare force signal for cross-correlation
    force_columns = get_columns([legs + '_ground_force_vy'], force_headers)
    forces_for_cross_corr = force_data[:, force_columns]

    # Downsample force data to match kinematics sampling rate
    framerate_forces = 1 / np.diff(force_data[:2, 0])[0]
    framerate_kinematics = 1 / np.diff(kinematics.time[:2])[0]

    time_forces_downsamp, forces_for_cross_corr_downsamp = ut.downsample(
        forces_for_cross_corr, force_data[:, 0],
        framerate_forces, framerate_kinematics
    )

    forces_for_cross_corr_downsamp = lowPassFilter(
        time_forces_downsamp,
        forces_for_cross_corr_downsamp,
        4, order=4
    )

    # Align force and COM signals by padding if necessary
    dif_lengths = len(forces_for_cross_corr_downsamp) - len(
        center_of_mass_acc_filtered['y'] * mass + mass * 9.8
    )

    if dif_lengths > 0:
        com_signal = np.pad(
            center_of_mass_acc_filtered['y'] * mass + mass * 9.8,
            (int(np.floor(dif_lengths / 2)), int(np.ceil(dif_lengths / 2))),
            'constant', constant_values=0
        )[:, np.newaxis]
        force_signal = forces_for_cross_corr_downsamp

    elif dif_lengths < 0:
        force_signal = np.pad(
            forces_for_cross_corr_downsamp,
            ((int(np.floor(np.abs(dif_lengths) / 2)),
              int(np.ceil(np.abs(dif_lengths) / 2))),
             (0, 0)),
            'constant', constant_values=0
        )
        com_signal = center_of_mass_acc_filtered['y'].values[:, np.newaxis] * mass + mass * 9.8
    else:
        force_signal = forces_for_cross_corr_downsamp

    # Normalize signals for synchronization
    force_signal = (force_signal - np.min(force_signal)) / (
        np.max(force_signal) - np.min(force_signal)
    )
    forces_for_cross_corr = (forces_for_cross_corr - np.min(forces_for_cross_corr)) / (
        np.max(forces_for_cross_corr) - np.min(forces_for_cross_corr)
    )

    # Detect initial force application and calculate lag time
    index_force = detect_upforce(forces_for_cross_corr)
    lag_time = force_data[index_force, 0] - time_heel[primal_leg_index]
    
    lag_times.append({
        'participante': participant_id,
        'movimiento': trial_name,
        'lag': lag_time
    })
    save_results(lag_times, 'lag_times.json')
    
    # Apply temporal synchronization
    force_data_new = np.copy(force_data)
    force_data_new[:, 0] = force_data[:, 0] - (
        force_data[index_force, 0] - time_heel[primal_leg_index]
    )
    
    # Synchronize heel coordinates with corrected force time
    time_force_corrected = force_data_new[:, 0]
    
    def robust_heel_interpolation(time_heel, heel_coords, time_force_sync, coord_name):
        """
        Robust interpolation with quality validation and error handling.
        
        Args:
            time_heel: Heel marker time array
            heel_coords: Heel coordinate array (X, Y, or Z)
            time_force_sync: Synchronized force plate time array
            coord_name: Coordinate name for error messages
        
        Returns:
            Interpolated heel coordinates at force plate time points
        """
        overlap_start = max(time_force_sync[0], time_heel[0])
        overlap_end = min(time_force_sync[-1], time_heel[-1])
        
        if overlap_end <= overlap_start:
            print(f"Warning: Limited temporal overlap for {coord_name}")
            method = 'linear'
        else:
            heel_points_in_overlap = np.sum(
                (time_heel >= overlap_start) & (time_heel <= overlap_end)
            )
            if heel_points_in_overlap > 10:
                method = 'cubic'
            else:
                method = 'linear'
        
        try:
            f_interp = interp1d(
                time_heel, heel_coords, kind=method,
                bounds_error=False, fill_value='extrapolate'
            )
            coords_interp = f_interp(time_force_sync)
            
            if np.any(np.isnan(coords_interp)) or np.any(np.isinf(coords_interp)):
                print(f"Warning: {method} interpolation failed for {coord_name}, using linear")
                f_interp = interp1d(
                    time_heel, heel_coords, kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
                coords_interp = f_interp(time_force_sync)
            
            return coords_interp
            
        except Exception as e:
            print(f"Error in interpolation for {coord_name}: {e}")
            return np.full_like(time_force_sync, heel_coords[0])
    
    # Interpolate heel coordinates to match force plate time
    pos_Rheel_x_sync = robust_heel_interpolation(
        time_heel, pos_Rheel_x, time_force_corrected, "RHeel_X"
    )
    pos_Lheel_x_sync = robust_heel_interpolation(
        time_heel, pos_Lheel_x, time_force_corrected, "LHeel_X"
    )
    pos_Rheel_z_sync = robust_heel_interpolation(
        time_heel, pos_Rheel_z, time_force_corrected, "RHeel_Z"
    )
    pos_Lheel_z_sync = robust_heel_interpolation(
        time_heel, pos_Lheel_z, time_force_corrected, "LHeel_Z"
    )
    
    # Assign synchronized coordinates to force data
    ground_R_leg_x = get_columns(['R' + '_ground_force_vx'], force_headers)
    ground_L_leg_x = get_columns(['L' + '_ground_force_vx'], force_headers)
    ground_R_leg_z = get_columns(['R' + '_ground_force_vz'], force_headers)
    ground_L_leg_z = get_columns(['L' + '_ground_force_vz'], force_headers)
    
    force_data_new[:, ground_R_leg_x] = pos_Rheel_x_sync.reshape(-1, 1)
    force_data_new[:, ground_L_leg_x] = pos_Lheel_x_sync.reshape(-1, 1)
    force_data_new[:, ground_R_leg_z] = pos_Rheel_z_sync.reshape(-1, 1)
    force_data_new[:, ground_L_leg_z] = pos_Lheel_z_sync.reshape(-1, 1)
    
    # Apply spatial correction to center of pressure
    cop_R_x_cols = get_columns(['R' + '_ground_force_px'], force_headers)
    cop_L_x_cols = get_columns(['L' + '_ground_force_px'], force_headers)
    cop_R_z_cols = get_columns(['R' + '_ground_force_pz'], force_headers)
    cop_L_z_cols = get_columns(['L' + '_ground_force_pz'], force_headers)
    
    if cop_R_x_cols and cop_L_x_cols:
        cop_R_x_original = force_data_new[:, cop_R_x_cols].flatten()
        cop_L_x_original = force_data_new[:, cop_L_x_cols].flatten()
        cop_R_z_original = force_data_new[:, cop_R_z_cols].flatten()
        cop_L_z_original = force_data_new[:, cop_L_z_cols].flatten()
        
        # Calculate average offset during contact
        force_threshold = 50  # N
        vgrf_R_cols = get_columns(['R_ground_force_vy'], force_headers)
        vgrf_L_cols = get_columns(['L_ground_force_vy'], force_headers)
        
        if vgrf_R_cols and vgrf_L_cols:
            vgrf_R = force_data_new[:, vgrf_R_cols].flatten()
            vgrf_L = force_data_new[:, vgrf_L_cols].flatten()
            
            contact_R = vgrf_R > force_threshold
            contact_L = vgrf_L > force_threshold
            
            if np.any(contact_R):
                offset_R_x = np.mean(pos_Rheel_x_sync[contact_R] - cop_R_x_original[contact_R])
                offset_R_z = np.mean(pos_Rheel_z_sync[contact_R] - cop_R_z_original[contact_R])
                
                force_data_new[:, cop_R_x_cols] = (cop_R_x_original + offset_R_x).reshape(-1, 1)
                force_data_new[:, cop_R_z_cols] = (cop_R_z_original + offset_R_z).reshape(-1, 1)
            
            if np.any(contact_L):
                offset_L_x = np.mean(pos_Lheel_x_sync[contact_L] - cop_L_x_original[contact_L])
                offset_L_z = np.mean(pos_Lheel_z_sync[contact_L] - cop_L_z_original[contact_L])
                
                force_data_new[:, cop_L_x_cols] = (cop_L_x_original + offset_L_x).reshape(-1, 1)
                force_data_new[:, cop_L_z_cols] = (cop_L_z_original + offset_L_z).reshape(-1, 1)

    # Visualization (if enabled)
    if visualize_synchronization:
        plt.figure()
        plt.plot(kinematics.time, com_signal, label='COM acceleration')
        plt.plot(force_data_new[:, 0], forces_for_cross_corr, label='vGRF')
        plt.legend()
        plt.grid()
        plt.show()
    
    # Save synchronization plot
    if save_plot:
        save_folder = os.path.join("graficas", participant_id)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{trial_name}_corte.png")
        
        plt.figure(figsize=(8, 5))
        plt.plot(time_heel, pos_Rheel_y_filtered, label='RHeel')
        plt.plot(time_heel, pos_Lheel_y_filtered, label='LHeel')
        plt.axvline(
            x=time_heel[primal_leg_index],
            color='r', linestyle='--',
            label='Heel strike'
        )
        plt.title(f'Trial: {trial_name} | Participant: {participant_id}')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend(loc='upper left', fontsize=8, frameon=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Save synchronized force data
    force_folder = os.path.join(data_folder, 'MeasuredForces', trial_name)
    os.makedirs(force_folder, exist_ok=True)
    force_output_path = os.path.join(force_folder, trial_name + '_syncd_forces.mot')
    ut.numpy_to_storage(force_headers, force_data_new, force_output_path, datatype=None)

    # Calculate time range for inverse dynamics
    time_range = {}
    time_range['start'] = np.max([force_data_new[0, 0], kinematics.time[0]])
    time_range['end'] = np.min([force_data_new[-1, 0], kinematics.time[-1]])

    # Setup and run inverse dynamics in OpenSim
    opensim_folder = os.path.join(data_folder, 'OpenSimData')
    id_folder = os.path.join(opensim_folder, 'InverseDynamics', trial_name)
    os.makedirs(id_folder, exist_ok=True)

    model_path = os.path.join(opensim_folder, 'Model', model_name + '.osim')
    ik_path = os.path.join(opensim_folder, 'Kinematics', trial_name + '.mot')
    el_path = os.path.join(id_folder, 'Setup_ExternalLoads.xml')
    id_path = os.path.join(id_folder, 'Setup_ID.xml')
    
    id_path_generic = os.path.join(script_folder, 'ID_setup', 'Setup_ID.xml')
    el_path_generic = os.path.join(script_folder, 'ID_setup', 'Setup_ExternalLoads.xml')

    if run_ID:
        opensim.Logger.setLevelString('error')
        ELTool = opensim.ExternalLoads(el_path_generic, True)
        ELTool.setDataFileName(force_output_path)
        ELTool.setName(trial_name)
        ELTool.printToXML(el_path)

        IDTool = opensim.InverseDynamicsTool(id_path_generic)
        IDTool.setModelFileName(model_path)
        IDTool.setName(trial_name)
        IDTool.setStartTime(time_range['start'])
        IDTool.setEndTime(time_range['end'])
        IDTool.setExternalLoadsFileName(el_path)
        IDTool.setCoordinatesFileName(ik_path)
        
        if not filter_kinematics_data:
            freq = -1
        else:
            freq = lowpass_filter_frequency
        
        IDTool.setLowpassCutoffFrequency(freq)
        IDTool.setResultsDir(id_folder)
        IDTool.setOutputGenForceFileName(trial_name + '.sto')
        IDTool.printToXML(id_path)
        print('Running inverse dynamics.')
        IDTool.run()

    # Load inverse dynamics results
    id_output_path = os.path.join(id_folder, trial_name + '.sto')
    id_dataframe = ut.load_storage(id_output_path, outputFormat='dataframe')

    force_dataframe = pd.DataFrame(force_data_new, columns=force_headers)
    
    # Define columns for plotting (if needed)
    sagittal_dofs = ['ankle_angle', 'knee_angle', 'hip_flexion']
    kinematics_columns_plot = [
        s + '_' + leg for s in sagittal_dofs for leg in ['r', 'l']
    ]
    moment_columns_plot = [s + '_moment' for s in kinematics_columns_plot]
    force_columns_plot = [
        leg + '_ground_force_v' + dir
        for leg in ['R', 'L'] for dir in ['x', 'y', 'z']
    ]

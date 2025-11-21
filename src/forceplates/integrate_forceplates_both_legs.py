"""
Force plate integration module for both-leg movements with unknown initial leg.

This module integrates force plate data with kinematic data from OpenCap for
movements that use both legs simultaneously. Unlike the single-leg version,
this module automatically detects which leg initiated contact first by comparing
heel strike events from both legs.

Key differences from single-leg integration:
- Automatically detects initial contact leg by comparing both legs
- Uses combined force signals from both legs for synchronization
- Suitable for movements like squats, jumps, or bilateral exercises

Author: ForcePlateIntegration Team
"""

import os
import sys
import json
import shutil
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
    Save data to a JSON file safely.
    
    Args:
        data: Data to save (list or dict)
        filename: JSON filename (str)
    """
    with open(filename, 'w', encoding='utf-8') as f:
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


def detect_heel_strike(signal, signal2, threshold_factor=0.2, half_system=True):
    """
    Detect the first heel contact with the ground after takeoff.
    
    Uses maximum fall rate (derivative) to identify heel strike.
    
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

    return min_idx


def load_lag_times():
    """
    Load existing lag_times from JSON file.
    
    If file doesn't exist, returns an empty list.
    
    Returns:
        List of dictionaries with loaded lag_times
    """
    lag_times_file = 'lag_times.json'
    if os.path.exists(lag_times_file):
        try:
            with open(lag_times_file, 'r', encoding='utf-8') as f:
                lag_times = json.load(f)
            return lag_times
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading {lag_times_file}: {e}")
            print("  Initializing empty lag_times list.")
            return []
    else:
        return []


def save_lag_times(lag_times):
    """
    Save lag_times to JSON file.
    
    Args:
        lag_times: List of dictionaries with lag_times to save
    """
    lag_times_file = 'lag_times.json'
    try:
        save_results(lag_times, lag_times_file)
    except IOError as e:
        print(f"Error saving {lag_times_file}: {e}")


def IntegrateForcepalte_vc0(session_id, trial_name, force_gdrive_url, participant_id,
                            use_local_forces=False, local_forces_folder=None):
    """
    Integrate force plate data with kinematic data for both-leg movements.
    
    This function automatically detects which leg initiated contact first by
    comparing heel strike events from both legs. It then synchronizes force
    plate and kinematic data temporally and spatially.
    
    Key features:
    - Automatically detects initial contact leg
    - Uses combined force signals from both legs
    - Supports loading force data from local files or Google Drive
    - Runs inverse dynamics analysis in OpenSim
    
    Args:
        session_id: OpenCap session ID (str)
        trial_name: Name of the trial to process (str)
        force_gdrive_url: Google Drive URL to download force plate data (str)
        participant_id: Participant identifier (str)
        use_local_forces: If True, attempts to load forces from local folder (bool)
        local_forces_folder: Path to folder with local force files (str, optional)
    
    Returns:
        None (saves results to disk)
    
    Note:
        Suitable for movements like squats, jumps, or bilateral exercises
        where both legs are used simultaneously and the initial contact leg
        is unknown or variable.
    """
    lag_times = load_lag_times()
    lag_time = 1000  # Default initial value (indicates not calculated)
    
    lag_times.append({
        'participante': participant_id,
        'movimiento': trial_name,
        'lag': lag_time
    })
    save_lag_times(lag_times)

    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', participant_id))
    force_dir = os.path.join(data_folder, 'MeasuredForces', trial_name)
    force_path = os.path.join(force_dir, f'{trial_name}_forces.mot')
    os.makedirs(force_dir, exist_ok=True)

    # Default local path
    if local_forces_folder is None:
        local_forces_folder = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\PLATAFORMAS_mod"

    # Try to load from local first
    if use_local_forces and local_forces_folder:
        local_file = None
        if os.path.exists(local_forces_folder):
            for file in os.listdir(local_forces_folder):
                if trial_name.lower() in file.lower() and file.endswith(('.mot', '.csv', '.xlsx')):
                    local_file = os.path.join(local_forces_folder, file)
                    break
        
        if local_file and os.path.exists(local_file):
            print(f"Loading forces from local: {local_file}")
            if local_file.endswith('.mot'):
                shutil.copy(local_file, force_path)
            else:
                print(f"Unsupported format: {local_file}")
                raise ValueError(f"Unsupported file format. Use .mot")
        else:
            print(f"Local file not found for: {trial_name}")
            print(f"  Downloading from Google Drive...")
            response = requests.get(force_gdrive_url)
            with open(force_path, 'wb') as f:
                f.write(response.content)
    else:
        print(f"Downloading forces from Google Drive...")
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

    # Detect heel strike for both legs and determine which leg initiated contact
    index_Rchange = detect_heel_strike(pos_Rheel_y_filtered, pos_Lheel_y_filtered)
    index_Lchange = detect_heel_strike(pos_Lheel_y_filtered, pos_Rheel_y_filtered)

    # Select the leg that made contact first (earlier time)
    primal_leg_index = index_Rchange if time_heel[index_Rchange] < time_heel[index_Lchange] else index_Lchange

    # Use combined force signals from both legs for synchronization
    force_columns = get_columns([leg + '_ground_force_vy' for leg in ['R', 'L']], force_headers)
    forces_for_cross_corr = np.sum(force_data[:, force_columns], axis=1, keepdims=True)

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
        com_signal = center_of_mass_acc_filtered['y'][:, np.newaxis] * mass + mass * 9.8
        force_signal = forces_for_cross_corr_downsamp

    # Normalize signals for synchronization
    com_signal = (com_signal - np.min(com_signal)) / (
        np.max(com_signal) - np.min(com_signal)
    )
    force_signal = (force_signal - np.min(force_signal)) / (
        np.max(force_signal) - np.min(force_signal)
    )
    forces_for_cross_corr = (forces_for_cross_corr - np.min(forces_for_cross_corr)) / (
        np.max(forces_for_cross_corr) - np.min(forces_for_cross_corr)
    )

    # Detect initial force application and calculate lag time
    index_force = detect_upforce(forces_for_cross_corr)
    lag_time = force_data[index_force, 0] - time_heel[primal_leg_index]
    
    # Save calculated lag time
    lag_times.append({
        'participante': participant_id,
        'movimiento': trial_name,
        'lag': lag_time
    })
    save_lag_times(lag_times)
    
    # Apply temporal synchronization
    force_data_new = np.copy(force_data)
    force_data_new[:, 0] = force_data[:, 0] - lag_time

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
            label='Direction change'
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


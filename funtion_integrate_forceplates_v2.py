'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_gait_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to integrate force data from forceplates for a jump.
    
    All data should be expressed in meters and Newtons.
    
    Input data:
    1) OpenCap session identifier and trial name
    2) Force data in a .mot file with 9 columns per leg: (see example data)
    (Fx, Fy, Fz, Tx, Ty, Tz, COPx, COPy, COPz). Column names should be 
    (R_ground_force_x,...R_ground_torque_x,...,R_ground_force_px,...
     L_ground_force_x,........)
    All data should be expressed in meters and Newtons. The 

'''

import os
import sys
sys.path.append("./../../")
script_folder,_ = os.path.split(os.path.abspath(__file__))
                
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import opensim
import requests

import utils as ut
from utilsProcessing import lowPassFilter
from utilsPlotting import plot_dataframe
import utilsKinematics
from scipy.interpolate import interp1d
from scipy.signal import correlate
from numpy.fft import fft, ifft

def resample_signal(time_source, signal_source, time_target):
    """ Interpola los datos de fuerza al mismo tiempo que la cinemática. """
    interp_func = interp1d(time_source, signal_source, kind='cubic', fill_value="extrapolate", axis=0)
    return interp_func(time_target)

def cross_corr_fft(signal1, signal2):
    """ Calcula la correlación cruzada usando FFT para mayor precisión. """
    f_signal1 = fft(signal1 - np.mean(signal1))
    f_signal2 = fft(signal2 - np.mean(signal2))
    correlation = ifft(f_signal1 * np.conj(f_signal2)).real
    lag = np.argmax(correlation) - (len(signal1) - 1)
    return lag

def compute_lag(signal1, signal2, window_size=1000, step_size=100):
    """ Aplica correlación cruzada en ventanas móviles para detectar el mejor lag. """
    max_corr = -np.inf
    best_lag = 0
    num_samples = min(len(signal1), len(signal2))

    for start in range(0, num_samples - window_size, step_size):
        end = start + window_size
        segment1 = signal1[start:end]
        segment2 = signal2[start:end]
        
        correlation = correlate(segment1 - np.mean(segment1), segment2 - np.mean(segment2), mode='full')
        lag = np.argmax(correlation) - (len(segment1) - 1)

        if np.max(correlation) > max_corr:
            max_corr = np.max(correlation)
            best_lag = lag

    return best_lag



def plot_synchronization(kinematics_time, com_signal, force_time, force_signal, lag=None):
    """ Grafica la aceleración del COM y las fuerzas verticales para verificar sincronización. """
    
    # Asegurar que ambas señales tengan la misma longitud
    force_signal = force_signal.squeeze()
    com_signal = com_signal.squeeze()

    # Interpolar la señal de fuerza para que coincida con la cinemática
    interp_force = interp1d(force_time, force_signal, kind='linear', fill_value="extrapolate")
    force_signal = interp_force(kinematics_time)
    force_time = np.copy(kinematics_time)  # Evita modificar la referencia original
    
    # Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(kinematics_time, com_signal, label="COM acceleration * mass + mg", linestyle='-', alpha=0.7)
    plt.plot(force_time, force_signal, label="Summed vGRF", linestyle='--', alpha=0.7)

    if lag is not None:
        plt.axvline(x=kinematics_time[0] + lag, color='r', linestyle=':', label=f"Lag: {lag:.2f} s")

    plt.xlabel("Time (s)")
    plt.ylabel("Force / Acceleration")
    plt.title("Synchronization of COM acceleration and GRF")
    plt.legend()
    plt.grid(True)
    
    plt.show()


def download_force_data(force_gdrive_url, force_path):
    """ Descarga datos de fuerza desde Google Drive. """
    os.makedirs(os.path.dirname(force_path), exist_ok=True)
    response = requests.get(force_gdrive_url)
    with open(force_path, 'wb') as f:
        f.write(response.content)

def load_kinematics_data(data_folder, session_id, trial_name):
    """ Carga datos de cinemática desde OpenCap. """
    if not os.path.exists(os.path.join(data_folder, 'MarkerData')):
        _, model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name, _ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))
    
    kinematics_obj = utilsKinematics.kinematics(data_folder, trial_name, modelName=model_name, lowpass_cutoff_frequency_for_coordinate_values=10)
    kinematics_obj.modelName = model_name  # Asegura que modelName existe en el objeto
    return kinematics_obj



def filter_force_data(force_data, force_headers, lowpass_filter_frequency):
    """ Aplica un filtro paso bajo a los datos de fuerza. """
    force_data[:, 1:] = lowPassFilter(force_data[:, 0], force_data[:, 1:], lowpass_filter_frequency, order=4)
    return force_data

def transform_forces_to_opencap_ref(force_data, force_headers):
    """ Aplica transformación de referencia de los datos de fuerza. """
    R_forcePlates_to_C = {'R': R.from_euler('y', -90, degrees=True),
                          'L': R.from_euler('y', -90, degrees=True)}

    quantity = ['ground_force_v', 'ground_torque_', 'ground_force_p']
    directions = ['x', 'y', 'z']

    for q in quantity:
        for leg in ['R', 'L']:
            force_columns = [i for i, name in enumerate(force_headers) if name.startswith(f"{leg}_{q}")]
            rot = R_forcePlates_to_C[leg]
            force_data[:, force_columns] = rot.inv().apply(force_data[:, force_columns])
    
    return force_data

def synchronize_forces_with_kinematics(force_data, force_headers, kinematics, mass):
    """ Sincroniza datos de fuerza con aceleraciones del centro de masa. """

    # Extraer señales
    center_of_mass_acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
    force_columns = [i for i, name in enumerate(force_headers) if '_ground_force_vy' in name]
    
    forces_for_cross_corr = np.sum(force_data[:, force_columns], axis=1, keepdims=True)

    # Interpolación cúbica para mejorar alineación
    forces_for_cross_corr = resample_signal(force_data[:, 0], forces_for_cross_corr, kinematics.time)

    # Aplicar filtro paso bajo
    forces_for_cross_corr = lowPassFilter(kinematics.time, forces_for_cross_corr, 4, order=4)

    # Calcular señal de referencia (COM * masa + gravedad)
    com_signal = center_of_mass_acc['y'].values * mass + mass * 9.8

    # Normalización para evitar sesgos en correlación
    forces_for_cross_corr = (forces_for_cross_corr - np.mean(forces_for_cross_corr)) / np.std(forces_for_cross_corr)
    com_signal = (com_signal - np.mean(com_signal)) / np.std(com_signal)

    # Cálculo del lag usando FFT + Ventana Móvil
    best_lag_fft = cross_corr_fft(com_signal, forces_for_cross_corr)
    best_lag_window = compute_lag(com_signal, forces_for_cross_corr, window_size=1000, step_size=100)

    # Usamos el promedio ponderado de ambas técnicas para mejorar la precisión
    best_lag = 0.7 * best_lag_window + 0.3 * best_lag_fft

    # Ajustar timestamps con el lag detectado
    force_data[:, 0] -= best_lag / (1 / np.diff(kinematics.time[:2])[0])

    # Graficar para validar sincronización
    # plot_synchronization(kinematics.time, com_signal, force_data[:, 0], forces_for_cross_corr, lag=best_lag)
    
    return force_data

def run_inverse_dynamics(data_folder, trial_name, model_name, force_output_path, ik_path, time_range):
    """ Ejecuta el análisis de dinámica inversa en OpenSim. """
    opensim_folder = os.path.join(data_folder, 'OpenSimData')
    id_folder = os.path.join(opensim_folder, 'InverseDynamics', trial_name)
    os.makedirs(id_folder, exist_ok=True)

    model_path = os.path.join(opensim_folder, 'Model', model_name + '.osim')
    el_path = os.path.join(id_folder, 'Setup_ExternalLoads.xml')
    id_path = os.path.join(id_folder, 'Setup_ID.xml')

    id_path_generic = os.path.join(script_folder, 'ID_setup', 'Setup_ID.xml')
    el_path_generic = os.path.join(script_folder, 'ID_setup', 'Setup_ExternalLoads.xml')

    # External loads
    ELTool = opensim.ExternalLoads(el_path_generic, True)
    ELTool.setDataFileName(force_output_path)
    ELTool.setName(trial_name)
    ELTool.printToXML(el_path)
    
    # ID tool
    IDTool = opensim.InverseDynamicsTool(id_path_generic)
    IDTool.setModelFileName(model_path)
    IDTool.setName(trial_name)
    IDTool.setStartTime(time_range['start'])
    IDTool.setEndTime(time_range['end'])      
    IDTool.setExternalLoadsFileName(el_path)
    IDTool.setCoordinatesFileName(ik_path)
    IDTool.setResultsDir(id_folder)
    IDTool.setOutputGenForceFileName(trial_name + '.sto')   
    IDTool.printToXML(id_path)   
    print('Running inverse dynamics.')
    IDTool.run()

def IntegrateForcepalte_custom(session_id, trial_name, force_gdrive_url):
    """ Función principal para integrar datos de fuerza con OpenCap. """
    
    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', session_id))
    force_dir = os.path.join(data_folder, 'MeasuredForces', trial_name)
    force_path = os.path.join(force_dir, f'{trial_name}_forces.mot')

    download_force_data(force_gdrive_url, force_path)
    
    kinematics = load_kinematics_data(data_folder, session_id, trial_name)
    
    forces_structure = ut.storage_to_numpy(force_path)
    force_data = forces_structure.view(np.float64).reshape(forces_structure.shape + (-1,))
    force_headers = forces_structure.dtype.names

    force_data = filter_force_data(force_data, force_headers, lowpass_filter_frequency=30)
    force_data = transform_forces_to_opencap_ref(force_data, force_headers)

    mass = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))['mass_kg']
    force_data = synchronize_forces_with_kinematics(force_data, force_headers, kinematics, mass)

    # Guardar fuerza procesada
    force_output_path = os.path.join(force_dir, f"{trial_name}_syncd_forces.mot")
    ut.numpy_to_storage(force_headers, force_data, force_output_path, datatype=None)

    # Rango de tiempo
    time_range = {
        'start': max(force_data[0, 0], kinematics.time[0]),
        'end': min(force_data[-1, 0], kinematics.time[-1])
    }

    run_inverse_dynamics(data_folder, trial_name, kinematics.modelName, force_output_path, 
                         os.path.join(data_folder, 'OpenSimData', 'Kinematics', f"{trial_name}.mot"), 
                         time_range)

    # Make plots
    # plt.close('all')
    # plot_dataframe([kinematics.get_coordinate_values()], xlabel='time', ylabel='angle [deg]',
    #                y = kinematics_columns_plot ,title='Kinematics')

    # plot_dataframe([id_dataframe], xlabel='time', ylabel='moment [Nm]',
    #                y = moment_columns_plot ,title='Moments')

    # plot_dataframe([force_dataframe], xlabel='time', ylabel='force [N]',
    #                y = force_columns_plot ,title='Ground Forces',xrange=list(time_range.values()))



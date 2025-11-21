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
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import medfilt, find_peaks
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, cutoff=3, fs=60, order=4):
    """
    Aplica un filtro pasabajas Butterworth a una señal.
    
    Parámetros:
    - signal: array de entrada.
    - cutoff: frecuencia de corte en Hz.
    - fs: frecuencia de muestreo en Hz.
    - order: orden del filtro.

    Retorna:
    - Señal filtrada.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def detect_upforce(signal, threshold_factor=0.1):
    """
    Detecta cundo hay el primer cambio de fuerza en la plataforma
    - signal: array, la señal de fuerza de las plataformas.
    - threshold_factor: float, factor del umbral basado en la desviación estándar.
    """
    baseline = np.mean(signal[:10])  # Nivel base del talón antes del movimiento
    
    # Encontrar el primer despegue (cuando el talón se eleva significativamente)
    
    for i in range(1, len(signal)):
        if signal[i] > baseline + np.std(signal) * threshold_factor:
            return i

def detect_heel_strike(signal, threshold_factor=2.5):
    """
    Detecta el primer contacto del talón con el suelo después de levantarlo.
    
    Parámetros:
    - signal: array, la señal del movimiento del talón.
    - threshold_factor: float, factor del umbral basado en la desviación estándar.

    Retorna:
    - Índice del impacto con el suelo después del primer despegue.
    """
    
    signal = medfilt(signal, kernel_size=5)  # Mediana para eliminar picos
    
    method = 'first_derivative'
    # --- Paso 1: Calcular derivada ---
    if method == 'first_derivative':
        derivative = np.diff(signal, prepend=signal[0])
    elif method == 'second_derivative':
        derivative = np.diff(np.diff(signal, prepend=signal[0]), prepend=0)
    else:
        raise ValueError("Método inválido. Usa 'first_derivative' o 'second_derivative'.")

    # --- Paso 2: Umbral basado en línea base inicial ---
    baseline_window = int(len(signal) / 5)
    baseline = np.mean(derivative[:baseline_window])
    std = np.std(derivative[:baseline_window])
    threshold = baseline + threshold_factor * std

    # --- Paso 3: Detectar despegue sostenido ---
    rising_window = 10
    despegue_idx = None
    for i in range(len(derivative) - rising_window):
        if np.all(derivative[i:i + rising_window] > threshold):
            despegue_idx = i
            break

    if despegue_idx is None:
        print("No se detectó despegue.")
        return None

    # --- Paso 4: Buscar mínimo local después del despegue ---
    landing_search_start = despegue_idx + 20
    landing_search_end = min(despegue_idx + 100, len(signal))
    local_min_idx = np.argmin(signal[landing_search_start:landing_search_end]) + landing_search_start

    # --- Visualización (opcional) ---

    plt.figure(figsize=(10, 5))
    plt.plot(signal, label="Señal original")
    plt.plot(derivative, '--', label=f"Derivada ({method})")
    plt.axhline(y=threshold, color='r', linestyle='--', label='Umbral derivada')
    plt.axvline(x=despegue_idx, color='g', linestyle='--', label='Despegue detectado')
    plt.axvline(x=local_min_idx, color='purple', linestyle='--', label='Caída del talón')
    plt.xlabel("Muestras")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return local_min_idx

def IntegrateForcepalte_v0(session_id, trial_name, force_gdrive_url,participant_id):
    
    data_folder = os.path.abspath(os.path.join(script_folder,'Data', session_id))

# Path and filename for force data. Should be a *.mot file of forces applied
# to right and left foot.
    force_dir = os.path.join(data_folder,'MeasuredForces',trial_name)
    force_path = os.path.join(force_dir,f'{trial_name}_forces.mot')
    # force_path = os.path.join(force_dir,f'{trial_name}.mot')

    # Download force data from gdrive: this is only for this example. This section
    # can be deleted if your force_path points to a local file.
    os.makedirs(force_dir,exist_ok=True)

    # if not os.path.exists(force_path):
    #     raise FileNotFoundError(f"El archivo de fuerzas {force_path} no se encontró. Verifica la ruta.")


    # force_gdrive_url = 'https://drive.google.com/uc?id=1Uppzkskn7OiJ5GncNT2sl35deX_U3FTN&export=download'
    # --------usar si es web------------------------
    # force_gdrive_url = 'https://drive.usercontent.google.com/u/2/uc?id=1-8bc4yZv0Ot8i0D6cczgXXQPBCyFrdCy&export=download'
    response = requests.get(force_gdrive_url)
    with open(force_path, 'wb') as f:
        f.write(response.content)  
    # ---------------------------------------------    

    # Lowpass filter ground forces and kinematics 
    lowpass_filter_frequency = 30
    filter_force_data = True
    filter_kinematics_data = True

    ## Transform from force reference frame to OpenCap reference frame.
    # We will use 4 reference frames G, C, R, and L:
    # C (checkerboard) and G (ground) are aligned and axes are defined by checkerboard. 
    # You should always have a black square in the top left corner, and the board 
    # should be on its side.
    # x out of board, y up, z away from checkers (left if looking at board). 
    # Origin (C0) is the top left black-to-black corner; 
    # Origin (G0) defined as opencap as the lowest point in the trial. There is a 
    # vertical offset value in the opencap data that defines r_C0_to_G0
    # Both kinematic and force data will be output in G.
    # R and L are the frames for the force plate data with origins R0 and L0.

    ## Position from checkerboard origin to force plate origin expressed in C.
    # You will need to measure this and position the checkerboard consistently for every
    # collection. 

    r_C0_to_forceOrigin_exp_C = {'R': [0,-.191,.083],
                                'L': [0,-.191,.083]}


    ## Rotation from force plates to checkerboard
    # You can represent rotations however you like, just make it a 
    # scipy.spatial.transform.Rotation object. 
    # docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    R_forcePlates_to_C = {'R':R.from_euler('y',-90,degrees=True),
                        'L':R.from_euler('y',-90,degrees=True)}

    # Flags
    visualize_synchronization = False#True # visualize kinematic/force synchronization
    save_plot = True
    run_ID = True

    # %% Functions

    def get_columns(list1,list2):
        inds = [i for i, item in enumerate(list2) if item in list1]
        return inds           

    #para verificar frecuencias de corte

    # %% Load and transform force data
    # We will transform the force data into the OpenCap reference frame. We will
    # then add a vertical offset to the COP, because the OpenCap data has a vertical
    # offset from the checkerboard origin.

    # Download kinematics data if folder doesn't exist. If incomplete data in folder,
    # delete the folder and re-run.
    if not os.path.exists(os.path.join(data_folder,'MarkerData')):
        _,model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name,_ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

    # Initiate kinematic analysis
    kinematics = utilsKinematics.kinematics(data_folder, trial_name, 
                                            modelName=model_name, 
                                            lowpass_cutoff_frequency_for_coordinate_values=10)

    # Get mass and vertical offset from opencap outputs
    opencap_settings = ut.get_main_settings(data_folder,trial_name)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder,'sessionMetadata.yaml'))
    mass = opencap_metadata['mass_kg']
    if 'verticalOffset' in opencap_settings.keys():
        vertical_offset = opencap_settings['verticalOffset']
    else:
        vertical_offset = 0

    forces_structure = ut.storage_to_numpy(force_path)
    force_data = forces_structure.view(np.float64).reshape(forces_structure.shape + (-1,))
    force_headers = forces_structure.dtype.names

    # %%

    # Filter force data
    # Note - it is not great to filter COP data directly. In the example GRF data
    # we filtered raw forces and moments before computing COP.

    if filter_force_data:
        force_data[:,1:] = lowPassFilter(force_data[:,0], force_data[:,1:],
                                    lowpass_filter_frequency, order=4)

    # Rotate the forces into C 
    quantity = ['ground_force_v','ground_torque_','ground_force_p']
    directions = ['x','y','z']
    for q in quantity:
        for leg in ['R','L']:
            force_columns= get_columns([leg + '_' + q + d for d in directions],force_headers)
            rot = R_forcePlates_to_C[leg]
            # we want r_expFP * R_FP_to_C, but rot.apply does R*r, so need to invert
            force_data[:,force_columns] = rot.inv().apply(force_data[:,force_columns])                                      
        
    ## Transform COP from force plates to G
    r_G0_to_C0_expC = np.array((0,-vertical_offset,0))

    for leg in ['R','L']:
        force_columns = get_columns([leg + '_ground_force_p' + d for d in directions],force_headers)
        r_forceOrigin_to_COP_exp_C = force_data[:,force_columns]
        r_G0_to_COP_exp_G = ( r_G0_to_C0_expC + 
                            r_C0_to_forceOrigin_exp_C[leg] + 
                            r_forceOrigin_to_COP_exp_C )
        force_data[:,force_columns] = r_G0_to_COP_exp_G
    # t_inicio = 4.55  

# Filtrar la señal de aceleración del centro de masa (COM) a partir del tiempo especificado
    

    ## Time synchronize
    # Here we will use cross correlation of the summed vertical GRFs vs. COM acceleration
    center_of_mass_acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
    # center_of_mass_pos = kinematics.get_center_of_mass_values(lowpass_cutoff_frequency=4) # other option
    # mask_cinematica = kinematics.time >= t_inicio
    # center_of_mass_acc_filtrada = center_of_mass_acc[mask_cinematica].copy()
    # time_cinematica_filtrada = kinematics.time[mask_cinematica].copy()  # Filtrar el tiempo 
    
    center_of_mass_acc_filtrada = center_of_mass_acc
    time_cinematica_filtrada = kinematics.time# Filtrar el tiempo 
    
    
    time_heel = kinematics.get_marker_dict(session_dir=f'Data\{session_id}', trial_name=trial_name, 
                        lowpass_cutoff_frequency=4)['time']
    pos_Rheel = kinematics.get_marker_dict(session_dir=f'Data\{session_id}', trial_name=trial_name, 
                        lowpass_cutoff_frequency=4)['markers']['RHeel']
    pos_Lheel = kinematics.get_marker_dict(session_dir=f'Data\{session_id}', trial_name=trial_name, 
                        lowpass_cutoff_frequency=4)['markers']['LHeel']
    
    pos_Rheel_y = (pos_Rheel[:,1] - np.min(pos_Rheel[:,1])) / (np.max(pos_Rheel[:,1]) - np.min(pos_Rheel[:,1]))
    pos_Lheel_y = (pos_Lheel[:,1] - np.min(pos_Lheel[:,1])) / (np.max(pos_Lheel[:,1]) - np.min(pos_Lheel[:,1]))

    pos_Rheel_y_filtered=lowPassFilter(time_heel, pos_Rheel_y, lowpass_cutoff_frequency = 3, order=4)
    pos_Lheel_y_filtered=lowPassFilter(time_heel, pos_Lheel_y,lowpass_cutoff_frequency= 3, order=4)
    pos_Rheel_y_filtered = lowpass_filter(pos_Rheel_y_filtered)
    pos_Lheel_y_filtered = lowpass_filter(pos_Lheel_y_filtered)


    index_Rchange = detect_heel_strike(pos_Rheel_y_filtered)
    index_Lchange = detect_heel_strike(pos_Lheel_y_filtered)
    # sampling_rate = 1/np.diff(time_heel[:2])[0]  # Ajusta según la frecuencia de muestreo de tus datos
    
    # plot_fft(pos_Lheel_y, sampling_rate, title="FFT - Señal del Talón")
    # plot_fft(pos_Rheel_y, sampling_rate, title="FFT - Centro de Masa")
    
    if time_heel[index_Rchange] < time_heel[index_Lchange]:
        primal_leg_index = index_Rchange
    elif time_heel[index_Lchange] < time_heel[index_Rchange]:
        primal_leg_index = index_Lchange

    #RBigToe

    # Ajustar el tiempo para que inicie en 0
    # time_cinematica_filtrada -= t_inicio
    # Ajustar el tiempo de la señal cinemática para que comience desde 0 después del corte
    
    # print(f"Dimensiones de time_cinematica_filtrada: {time_cinematica_filtrada.shape}")
    # print(f"Dimensiones de com_signal: {center_of_mass_acc_filtrada.shape}")
    force_columns = get_columns([leg + '_ground_force_vy' for leg in ['R','L']],force_headers)
    forces_for_cross_corr = np.sum(force_data[:,force_columns],axis=1,keepdims=True)

    framerate_forces = 1/np.diff(force_data[:2,0])[0]

    framerate_kinematics = 1/np.diff(kinematics.time[:2])[0]
    #print(center_of_mass_acc[center_of_mass_acc['time']>= 4.55])
    # print(f"Framerate de las fuerzas: {framerate_forces}")
    # print(f"Framerate de la cinemática: {framerate_kinematics}")
    # print('forces_for_cross_corr: ', len(forces_for_cross_corr))
    # print('forces_data-shape: ', len(force_data.shape))
    
    time_forces_downsamp, forces_for_cross_corr_downsamp = ut.downsample(forces_for_cross_corr,
                                                                        force_data[:,0],
                                                                        framerate_forces,
                                                                        framerate_kinematics)
    # print('time_forces_downsamp',time_forces_downsamp)
    forces_for_cross_corr_downsamp = lowPassFilter(time_forces_downsamp,
                                                forces_for_cross_corr_downsamp,
                                                4, order=4)
    # print(f'shape of forces_for_cross_corr_downsamp {forces_for_cross_corr_downsamp.shape}')
    # get body mass from metadata
    #----------------------modificado--------------------------------------------
    # Calculamos la diferencia de longitudes entre las señales
    # dif_lengths = len(forces_for_cross_corr_downsamp) - len(center_of_mass_pos['y'])
    dif_lengths = len(forces_for_cross_corr_downsamp) - len(center_of_mass_acc_filtrada['y'] * mass + mass * 9.8)

    if dif_lengths > 0:
        # forces_for_cross_corr_downsamp es más largo, entonces debemos ajustar com_signal
        com_signal = np.pad(center_of_mass_acc_filtrada['y'] * mass + mass * 9.8, 
                            (int(np.floor(dif_lengths / 2)), 
                            int(np.ceil(dif_lengths / 2))), 
                            'constant', constant_values=0)[:, np.newaxis]
        # com_signal = np.pad(center_of_mass_pos['y'] , 
        #                     (int(np.floor(dif_lengths / 2)), 
        #                     int(np.ceil(dif_lengths / 2))), 
        #                     'constant', constant_values=0)[:, np.newaxis]
        
        # force_signal = forces_for_cross_corr_downsamp
        kinematics_pad_length = int(np.floor(dif_lengths / 2))
        # min_length = min(len(center_of_mass_acc_filtrada['y']), len(forces_for_cross_corr_downsamp))
        # force_signal = forces_for_cross_corr_downsamp[:min_length]
        # com_signal = (center_of_mass_acc_filtrada['y'].values[:min_length, np.newaxis] * mass) + (mass * 9.8)
        
        
        


    elif dif_lengths < 0:
        # center_of_mass_acc['y'] es más largo, entonces debemos ajustar forces_for_cross_corr_downsamp
        # print(f"Shape of forces_for_cross_corr_downsamp before padding: {forces_for_cross_corr_downsamp.shape}")
        force_signal =forces_for_cross_corr_downsamp
        # print(force_signal.shape)
        # force_signal = np.pad(np.squeeze(forces_for_cross_corr_downsamp), 
        #                     (int(np.floor(np.abs(dif_lengths) / 2)), 
        #                     int(np.ceil(np.abs(dif_lengths) / 2))), 
        #                     'constant', constant_values=0)
        force_signal = np.pad(forces_for_cross_corr_downsamp, 
                        ((int(np.floor(np.abs(dif_lengths) / 2)), 
                            int(np.ceil(np.abs(dif_lengths) / 2))), 
                        (0, 0)),  # No modificar las columnas
                        'constant', constant_values=0)

        # print(f"Shape of force_signal after padding: {force_signal.shape}")
        
        com_signal = center_of_mass_acc_filtrada['y'].values[:, np.newaxis] * mass + mass * 9.8
        # com_signal = center_of_mass_pos['y'].values[:, np.newaxis] 
        
        
        kinematics_pad_length = 0
        # force_signal = force_signal[:, 0] #linea nueva
        # print('center_of_mass_acc es más largo que forces_for_cross_corr_downsamp')
        # print("Shape of com_signal:", com_signal.shape)
        # print('dif_lengths < 0')

    else:
        # Las longitudes ya son iguales, no es necesario hacer padding
        com_signal = center_of_mass_acc_filtrada['y'][:, np.newaxis] * mass + mass * 9.8
        
        # com_signal = center_of_mass_pos['y'][:, np.newaxis] 
        force_signal = forces_for_cross_corr_downsamp
        kinematics_pad_length = 0
        
        # print('Las señales ya tienen la misma longitud')
        # print("Shape of com_signal:", com_signal.shape)
        # print("Shape of force_signal:", force_signal.shape)
    #----------------------------------------------------------------------------------------

    # print(f"Dimensiones de time_cinematica_filtrada: {time_cinematica_filtrada.shape}")
    # print(f"Dimensiones de com_signal: {com_signal.shape}")
    com_signal = (com_signal - np.min(com_signal)) / (np.max(com_signal) - np.min(com_signal))
    force_signal = (force_signal - np.min(force_signal)) / (np.max(force_signal) - np.min(force_signal))
    forces_for_cross_corr = (forces_for_cross_corr - np.min(forces_for_cross_corr)) / (np.max(forces_for_cross_corr) - np.min(forces_for_cross_corr))
    
    index_force = detect_upforce(forces_for_cross_corr)
    max_corr,lag = ut.cross_corr(np.squeeze(com_signal),np.squeeze(force_signal),  #linea original donde el max_corr era _ y es force_signal
                        visualize=visualize_synchronization)
    
    # # Ahora, usar force_signal_interp en el cálculo de lag
    # max_corr, lag = ut.cross_corr(np.squeeze(com_signal), np.squeeze(force_signal_interp),
    #                             visualize=visualize_synchronization)
    

    # print(f"Calculated lag: {lag}")
    # print(f"Max correlation: {max_corr}")
    # lag = lag*0.1
    # _,lag = ut.cross_corr(np.squeeze(com_signal),np.squeeze(force_signal),
    #                                 multCorrGaussianStd=50, window_size=1000, use_fft=True, 
    #                                 #visualize=True
    #                                 )

    # print(f"Calculated lag: {lag}")
    

    # Desplazar la señal de fuerzas en función del lag
    print('lag time: ',force_data[index_force,0]-time_heel[primal_leg_index])
    force_data_new = np.copy(force_data)
    # force_data_new[:,0] = force_data[:,0] - (-lag+kinematics_pad_length)/framerate_kinematics#-lag/framerate_kinematics original
    force_data_new[:,0] = force_data[:,0] - (force_data[index_force,0]-time_heel[primal_leg_index])

    desplazamiento_derecha = 0#1.7  # Ajusta este valor según el desplazamiento deseado

    # Aplica el desplazamiento al tiempo de los datos de fuerza
    # force_data_new[:,0] = force_data_new[:, 0] + desplazamiento_derecha
    # Plot vertical force and (COM acceleration*m +mg)
    
    

    if visualize_synchronization:
        print("vizulize")
        plt.figure()
        plt.plot(kinematics.time,center_of_mass_acc['y']*mass + mass*9.8,label='COM acceleration')
        # plt.plot(kinematics.time,center_of_mass_pos['y'],label='COM acceleration')
        plt.plot(force_data_new[:,0],forces_for_cross_corr, label = 'vGRF')
        plt.legend()
        plt.grid()
        plt.show()
    if save_plot:
        save_folder = os.path.join("graficas", participant_id)
        os.makedirs(save_folder, exist_ok=True)  # Crea la carpeta si no existe

        # Ruta completa para guardar la imagen
        save_path = os.path.join(save_folder, f"{trial_name}_corte.png")

        # Crear la figura y graficar los datos
        plt.figure(figsize=(8, 5))
        # plt.plot(time_cinematica_filtrada, com_signal , label='COM')
        # plt.plot(force_data_new[:, 0], forces_for_cross_corr, label='vGRF')
        plt.plot(time_heel, pos_Rheel_y_filtered, label='RHeel' )
        plt.plot(time_heel, pos_Lheel_y_filtered, label='LHeel')   
        
        plt.axvline(x=time_heel[primal_leg_index], color='r', linestyle='--', label='Cambio de dirección')
        # plt.axvline(x=force_data_new[index_force,0], color='g', linestyle='--', label='Inicio de la fuerza')
        
        plt.title('Señales sincronizadas', fontsize=12)
        plt.title(f'Trial: {trial_name} | Participante: {participant_id}')
        
        plt.xlabel('Time (s)')        
        plt.grid()        
        plt.legend(loc='upper left', fontsize=8, frameon=True)

        # Guardar la imagen sin mostrarla
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Force data directories
    force_folder = os.path.join(data_folder,'MeasuredForces',trial_name)
    os.makedirs(force_folder,exist_ok=True)

    # Save force data
    _,force_file_name = os.path.split(force_path)
    root,ext = os.path.splitext(force_file_name)
    force_output_path = os.path.join(force_folder,trial_name + '_syncd_forces.mot')
    ut.numpy_to_storage(force_headers, force_data_new, force_output_path, datatype=None)

    # Identify time range for ID
    time_range = {}
    time_range['start'] = np.max([force_data_new[0,0], kinematics.time[0]])
    time_range['end'] = np.min([force_data_new[-1,0], kinematics.time[-1]])

    # print('time start of force',force_data_new[0,0])
    # print('time start of kinematics',kinematics.time[0])

    # print('time end of force',force_data_new[-1,0])
    # print('time end of kinematics',kinematics.time[-1])

    # print('time_range star',time_range['start'])
    # print('time_range end',time_range['end'])

    # %% Run Inverse Dynamics

    # Make folders
    opensim_folder = os.path.join(data_folder,'OpenSimData')

    id_folder = os.path.join(opensim_folder,'InverseDynamics',trial_name)
    os.makedirs(id_folder,exist_ok=True)

    model_path = os.path.join(opensim_folder,'Model',model_name + '.osim')
    ik_path = os.path.join(opensim_folder,'Kinematics', trial_name + '.mot')
    el_path = os.path.join(id_folder,'Setup_ExternalLoads.xml')
    id_path = os.path.join(id_folder,'Setup_ID.xml')

    # Generic setups
    id_path_generic = os.path.join(script_folder,'ID_setup','Setup_ID.xml')
    el_path_generic = os.path.join(script_folder,'ID_setup','Setup_ExternalLoads.xml')

    if run_ID:
        # External loads
        opensim.Logger.setLevelString('error')
        ELTool = opensim.ExternalLoads(el_path_generic, True)
        ELTool.setDataFileName(force_output_path)
        ELTool.setName(trial_name)
        ELTool.printToXML(el_path)
        
        # ID    
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

    # %% Load and plot kinematics, forces, inverse dynamics results

    # Create force and ID dataframesID results
    id_output_path = os.path.join(id_folder,trial_name + '.sto')
    id_dataframe = ut.load_storage(id_output_path,outputFormat='dataframe')

    force_dataframe = pd.DataFrame(force_data_new, columns=force_headers)
    # print(force_dataframe)
    # Select columns to plot
    sagittal_dofs = ['ankle_angle','knee_angle','hip_flexion']
    kinematics_columns_plot = [s + '_' + leg for s in sagittal_dofs for leg in ['r','l']]
    moment_columns_plot = [s + '_moment' for s in kinematics_columns_plot]
    force_columns_plot = [leg + '_ground_force_v' + dir for leg in ['R','L'] 
                                                    for dir in ['x','y','z']]

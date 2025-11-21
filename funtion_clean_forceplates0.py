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
import json

import utils as ut
from utilsProcessing import lowPassFilter
import utilsKinematics
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks
from scipy.signal import butter, filtfilt

def guardar_resultados(data, filename):
    """
    Guarda datos en formato JSON de manera segura.
    
    Parámetros:
    - data: datos a guardar (lista o diccionario)
    - filename: nombre del archivo JSON
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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

def detect_interception(signal1, signal2, threshold=0.028, method='first', filter_kernel=3, visualize=False):
    """
    Detecta el punto de intersección (o convergencia) entre dos señales del mismo tamaño.
    
    Parámetros:
    - signal1, signal2: arrays de señales (mismo tamaño).
    - threshold: valor máximo de diferencia para considerar que hay "intersección".
    - method: 'first', 'closest', o 'centered' (modo de selección del punto).
    - filter_kernel: si >1, aplica filtro de mediana para suavizar.
    - visualize: si True, grafica las señales y el punto encontrado.

    Retorna:
    - Índice del punto de intersección o None si no se encuentra.
    """

    

    # Diferencia absoluta
    diff = np.abs(signal1 - signal2)

    # Encuentra todos los puntos donde se acercan
    close_idxs = np.where(diff < threshold)[0]

    if len(close_idxs) == 0:
        print("No se encontró intersección bajo el umbral.")
        return None

    # Selección del índice
    if method == 'first':
        index = close_idxs[0]
    elif method == 'closest':
        index = np.argmin(diff)
    elif method == 'centered':
        mid = len(signal1) // 2
        index = close_idxs[np.argmin(np.abs(close_idxs - mid))]
    else:
        raise ValueError("Método no válido. Usa 'first', 'closest' o 'centered'.")

    # Visualización
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(signal1,'o', label="Señal 1")
    # plt.plot(signal2,'o', label="Señal 2")
    # plt.axvline(index, color='purple', linestyle='--', label='Intersección')
    # plt.title("Detección de intersección entre señales")
    # plt.xlabel("Muestras")
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    if index:
        return index 
    else: return None

def detect_heel_strike(signal,signal2, threshold_factor=0.2, half_sistem=True):
    """
    Detecta el primer contacto del talón con el suelo después del despegue,
    usando la caída máxima (segunda derivada).
    
    Parámetros:
    - signal: array, señal vertical del talón normalizada.
    - threshold_factor: float, para ajustar la sensibilidad del despegue.

    Retorna:
    - Índice del impacto con el suelo.
    """

    # plt.figure(figsize=(10, 5))
    # plt.plot(signal, 'b', label="Señal (principal)")
    # plt.plot(signal2, 'g', label="Señal (ref)")  
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    if half_sistem:
        half = int(len(signal) / 2)
        signal = medfilt(signal[:half], kernel_size=5)
        signal2 = signal2[:half]
    else:
        signal = medfilt(signal[:], kernel_size=5)
        signal2 = signal2[:]
        

    # --- Paso 1: Detectar despegue basado en umbral simple ---
    baseline = max(signal[50:]) - min(signal[:])
    threshold = baseline * threshold_factor + min(signal[:])
    rising_window = 10
    despegue_idx = None
    for i in range(30,len(signal) - rising_window):
        future = signal[i+1] - signal[i] 
        if np.all(signal[i:i + rising_window] > threshold) and future > 0:
            despegue_idx = i
            break

    # plt.figure(figsize=(10, 5))
    # plt.plot(signal, 'b', label="Señal (principal)")
    # plt.plot(signal2, 'g', label="Señal (ref)")  
    # plt.axhline(y=threshold, color='gray', linestyle='--', label='Umbral despegue')
    # plt.axhline(y=max(signal[50:]), color='green',linestyle='--', label='Maximo')
    # plt.axhline(y=min(signal[:]), color='green',linestyle='--', label='Minimo')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    if despegue_idx is None:
        print("No se detectó despegue.")
        return None

    # --- Paso 2: Detectar caída rápida después del despegue ---
    search_start = despegue_idx + 20
    search_end = min(despegue_idx + 150, len(signal))

    deriv = np.gradient(signal)
    second_deriv = np.gradient(deriv)

    fall_zone = deriv[search_start:search_end]
    if len(fall_zone) == 0:
        print("Zona de caída vacía.")
        return None

    impact_idx = np.argmin(fall_zone) + search_start
    # impact_idx = detect_flat_segment(fall_zone, search_start)
    

    # plt.figure(figsize=(10, 5))
    # plt.plot(signal, 'b', label="Señal (filtrada)")
    # plt.plot(signal2, 'g', label="Señal (filtrada)")
    
    # plt.axvline(x=despegue_idx, color='orange', linestyle='--', label='Despegue')
    # plt.axvline(x=impact_idx, color='purple', linestyle='--', label='Impacto del talón')
    # plt.axhline(y=threshold, color='gray', linestyle='--', label='Umbral despegue')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    if half_sistem:
        
        landing_search_start = despegue_idx + 10
        indx_interception = detect_interception(signal[landing_search_start:],signal2[landing_search_start:])
        if indx_interception is None:
            return impact_idx
        local_min_idx = indx_interception + landing_search_start

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
            print("❌ No se pudo determinar el impacto.")
            return None

    
    
    # --- Visualización ---
    # plt.figure(figsize=(10, 5))
    # plt.plot(signal, 'b', label="Señal principal (filtrada)")
    # plt.plot(signal2, 'g', label="Señal (filtrada)")
    # plt.plot(deriv, 'g--', label="Primera derivada")
    # plt.plot(second_deriv, 'r--', label="Segunda derivada")
    # plt.axvline(x=despegue_idx, color='orange', linestyle='--', label='Despegue')
    # plt.axvline(x=impact_idx, color='purple', linestyle='--', label='Impacto del talón')
    # plt.axvline(x=local_min_idx, color='red', linestyle='--', label='Intercepción')
    # plt.axvline(x=min_idx, color='black', linestyle='--', label='Índice mínimo')
    # plt.axhline(y=threshold, color='gray', linestyle='--', label='Umbral despegue')
    # # plt.axhline(y=max(signal[:]), color='green',linestyle='--', label='Maximo')
    # # plt.axhline(y=min(signal[:]), color='green',linestyle='--', label='Minimo')
    # plt.legend()
    # plt.grid(True)
    # plt.title("Detección de caída del talón")
    # plt.xlabel("Muestras")
    # plt.tight_layout()
    # plt.show()

    return min_idx

# ============================================================================
# GESTIÓN DE LAG_TIMES
# ============================================================================
def cargar_lag_times():
    """
    Carga los lag_times existentes desde el archivo JSON.
    Si el archivo no existe, retorna una lista vacía.
    
    Retorna:
    - Lista de diccionarios con los lag_times cargados
    """
    lag_times_file = 'lag_times.json'
    if os.path.exists(lag_times_file):
        try:
            with open(lag_times_file, 'r', encoding='utf-8') as f:
                lag_times = json.load(f)
            return lag_times
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Advertencia: Error al cargar {lag_times_file}: {e}")
            print("  Inicializando lista vacía de lag_times.")
            return []
    else:
        return []

def guardar_lag_times(lag_times):
    """
    Guarda los lag_times en el archivo JSON.
    
    Parámetros:
    - lag_times: lista de diccionarios con los lag_times a guardar
    """
    lag_times_file = 'lag_times.json'
    try:
        guardar_resultados(lag_times, lag_times_file)
    except IOError as e:
        print(f"❌ Error al guardar {lag_times_file}: {e}")

def IntegrateForcepalte_vc0(session_id, trial_name, force_gdrive_url, participant_id, use_local_forces=False, local_forces_folder=None):
    """
    Integra datos de plataformas de fuerza con datos cinemáticos de OpenCap.
    
    Parámetros:
    - session_id: ID de la sesión de OpenCap
    - trial_name: nombre del trial
    - force_gdrive_url: URL de Google Drive para descargar datos de fuerza
    - participant_id: ID del participante
    - use_local_forces: si True, intenta cargar fuerzas desde carpeta local
    - local_forces_folder: ruta a la carpeta con fuerzas locales
    """
    
    # ========================================================================
    # INICIALIZACIÓN DE LAG_TIMES
    # ========================================================================
    lag_times = cargar_lag_times()
    lag_time = 1000  # Valor inicial por defecto (indica no calculado)
    
    # Guardar registro inicial con valor por defecto
    lag_times.append({
        'participante': participant_id,
        'movimiento': trial_name,
        'lag': lag_time
    })
    guardar_lag_times(lag_times)

    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', participant_id))
    force_dir = os.path.join(data_folder, 'MeasuredForces', trial_name)
    force_path = os.path.join(force_dir, f'{trial_name}_forces.mot')
    os.makedirs(force_dir, exist_ok=True)

    # Definir ruta local por defecto
    if local_forces_folder is None:
        local_forces_folder = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\PLATAFORMAS_mod"

    # Intenta cargar desde local primero
    if use_local_forces and local_forces_folder:
        local_file = None
        if os.path.exists(local_forces_folder):
            # Buscar archivo que coincida con el trial_name
            for file in os.listdir(local_forces_folder):
                if trial_name.lower() in file.lower() and file.endswith(('.mot', '.csv', '.xlsx')):
                    local_file = os.path.join(local_forces_folder, file)
                    break
        
        if local_file and os.path.exists(local_file):
            print(f"✓ Cargando fuerzas desde local: {local_file}")
            if local_file.endswith('.mot'):
                import shutil
                shutil.copy(local_file, force_path)
            else:
                print(f"⚠ Formato no soportado: {local_file}")
                raise ValueError(f"Formato de archivo no soportado. Use .mot")
        else:
            print(f"⚠ No se encontró archivo local para: {trial_name}")
            print(f"  Descargando desde Google Drive...")
            response = requests.get(force_gdrive_url)
            with open(force_path, 'wb') as f:
                f.write(response.content)
    else:
        print(f"Descargando fuerzas desde Google Drive...")
        response = requests.get(force_gdrive_url)
        with open(force_path, 'wb') as f:
            f.write(response.content)

    lowpass_filter_frequency = 30
    filter_force_data = True
    filter_kinematics_data = True

    r_C0_to_forceOrigin_exp_C = {'R': [0, -.191, .083],
                                'L': [0, -.191, .083]}
    R_forcePlates_to_C = {'R': R.from_euler('y', -90, degrees=True),
                        'L': R.from_euler('y', -90, degrees=True)}

    visualize_synchronization = False
    save_plot = True
    run_ID = True

    def get_columns(list1, list2):
        return [i for i, item in enumerate(list2) if item in list1]

    if not os.path.exists(os.path.join(data_folder, 'MarkerData')):
        _, model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name, _ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

    kinematics = utilsKinematics.kinematics(data_folder, trial_name,
                                            modelName=model_name,
                                            lowpass_cutoff_frequency_for_coordinate_values=10)

    opencap_settings = ut.get_main_settings(data_folder, trial_name)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))
    mass = opencap_metadata['mass_kg']
    if 'verticalOffset' in opencap_settings.keys():
        vertical_offset = opencap_settings['verticalOffset']
    else:
        vertical_offset = 0

    forces_structure = ut.storage_to_numpy(force_path)
    force_data = forces_structure.view(np.float64).reshape(forces_structure.shape + (-1,))
    force_headers = forces_structure.dtype.names

    # %%
    if filter_force_data:
        force_data[:, 1:] = lowPassFilter(force_data[:, 0], force_data[:, 1:],
                                        lowpass_filter_frequency, order=4)

    quantity = ['ground_force_v', 'ground_torque_', 'ground_force_p']
    directions = ['x', 'y', 'z']
    for q in quantity:
        for leg in ['R', 'L']:
            force_columns = get_columns([leg + '_' + q + d for d in directions], force_headers)
            rot = R_forcePlates_to_C[leg]
            force_data[:, force_columns] = rot.inv().apply(force_data[:, force_columns])

    r_G0_to_C0_expC = np.array((0, -vertical_offset, 0))

    for leg in ['R', 'L']:
        force_columns = get_columns([leg + '_ground_force_p' + d for d in directions], force_headers)
        r_forceOrigin_to_COP_exp_C = force_data[:, force_columns]
        r_G0_to_COP_exp_G = ( r_G0_to_C0_expC + 
                            r_C0_to_forceOrigin_exp_C[leg]+
                            r_forceOrigin_to_COP_exp_C )
        force_data[:, force_columns] = r_G0_to_COP_exp_G

    center_of_mass_acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
    center_of_mass_acc_filtrada = center_of_mass_acc
    time_cinematica_filtrada = kinematics.time

    marker_data = kinematics.get_marker_dict(session_dir=f'Data\{participant_id}', trial_name=trial_name, lowpass_cutoff_frequency=4)
    time_heel = marker_data['time']
    pos_Rheel = marker_data['markers']['RHeel']
    pos_Lheel = marker_data['markers']['LHeel']

    pos_Rheel_y = (pos_Rheel[:, 1] - np.min(pos_Rheel[:, 1])) / (np.max(pos_Rheel[:, 1]) - np.min(pos_Rheel[:, 1]))
    pos_Lheel_y = (pos_Lheel[:, 1] - np.min(pos_Lheel[:, 1])) / (np.max(pos_Lheel[:, 1]) - np.min(pos_Lheel[:, 1]))

    pos_Rheel_y_filtered=lowPassFilter(time_heel, pos_Rheel_y, lowpass_cutoff_frequency = 3, order=4)
    pos_Lheel_y_filtered=lowPassFilter(time_heel, pos_Lheel_y,lowpass_cutoff_frequency= 3, order=4)
    pos_Rheel_y_filtered = lowpass_filter(pos_Rheel_y_filtered)
    pos_Lheel_y_filtered = lowpass_filter(pos_Lheel_y_filtered)

    index_Rchange = detect_heel_strike(pos_Rheel_y_filtered,pos_Lheel_y_filtered)
    index_Lchange = detect_heel_strike(pos_Lheel_y_filtered,pos_Rheel_y_filtered)

    primal_leg_index = index_Rchange if time_heel[index_Rchange] < time_heel[index_Lchange] else index_Lchange

    force_columns = get_columns([leg + '_ground_force_vy' for leg in ['R', 'L']], force_headers)
    forces_for_cross_corr = np.sum(force_data[:, force_columns], axis=1, keepdims=True)

    framerate_forces = 1 / np.diff(force_data[:2, 0])[0]
    framerate_kinematics = 1 / np.diff(kinematics.time[:2])[0]

    time_forces_downsamp, forces_for_cross_corr_downsamp = ut.downsample(
        forces_for_cross_corr, force_data[:, 0], framerate_forces, framerate_kinematics)

    forces_for_cross_corr_downsamp = lowPassFilter(time_forces_downsamp,
                                                    forces_for_cross_corr_downsamp,
                                                    4, order=4)

    dif_lengths = len(forces_for_cross_corr_downsamp) - len(center_of_mass_acc_filtrada['y'] * mass + mass * 9.8)

    if dif_lengths > 0:
        com_signal = np.pad(center_of_mass_acc_filtrada['y'] * mass + mass * 9.8,
                            (int(np.floor(dif_lengths / 2)), int(np.ceil(dif_lengths / 2))),
                            'constant', constant_values=0)[:, np.newaxis]
        # kinematics_pad_length = int(np.floor(dif_lengths / 2))
        force_signal = forces_for_cross_corr_downsamp

    elif dif_lengths < 0:
        force_signal = np.pad(forces_for_cross_corr_downsamp,
                            ((int(np.floor(np.abs(dif_lengths) / 2)),
                                int(np.ceil(np.abs(dif_lengths) / 2))),
                            (0, 0)), 'constant', constant_values=0)
        com_signal = center_of_mass_acc_filtrada['y'].values[:, np.newaxis] * mass + mass * 9.8
    else:
        com_signal = center_of_mass_acc_filtrada['y'][:, np.newaxis] * mass + mass * 9.8
        force_signal = forces_for_cross_corr_downsamp

    com_signal = (com_signal - np.min(com_signal)) / (np.max(com_signal) - np.min(com_signal))
    force_signal = (force_signal - np.min(force_signal)) / (np.max(force_signal) - np.min(force_signal))
    forces_for_cross_corr = (forces_for_cross_corr - np.min(forces_for_cross_corr)) / (np.max(forces_for_cross_corr) - np.min(forces_for_cross_corr))

    index_force = detect_upforce(forces_for_cross_corr)

    # ========================================================================
    # CÁLCULO Y ALMACENAMIENTO DEL LAG_TIME REAL
    # ========================================================================
    lag_time = force_data[index_force, 0] - time_heel[primal_leg_index]
    
    # Guardar el lag_time calculado
    lag_times.append({
        'participante': participant_id,
        'movimiento': trial_name,
        'lag': lag_time
    })
    guardar_lag_times(lag_times)
    
    # Aplicar corrección temporal a los datos de fuerza
    force_data_new = np.copy(force_data)
    force_data_new[:, 0] = force_data[:, 0] - lag_time

    if save_plot:
        save_folder = os.path.join("graficas", participant_id)
        os.makedirs(save_folder, exist_ok=True)
        
        save_path = os.path.join(save_folder, f"{trial_name}_corte.png")
        
        plt.figure(figsize=(8, 5))
        # plt.plot(time_cinematica_filtrada, com_signal , label='COM')
        # plt.plot(force_data_new[:, 0], forces_for_cross_corr, label='vGRF')
        plt.plot(time_heel, pos_Rheel_y_filtered, label='RHeel')
        plt.plot(time_heel, pos_Lheel_y_filtered, label='LHeel')
        plt.axvline(x=time_heel[primal_leg_index], color='r', linestyle='--', label='Cambio de dirección')
        plt.title(f'Trial: {trial_name} | Participante: {participant_id}')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend(loc='upper left', fontsize=8, frameon=True)
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

    time_range = {}
    time_range['start'] = np.max([force_data_new[0,0], kinematics.time[0]])
    time_range['end'] = np.min([force_data_new[-1,0], kinematics.time[-1]])


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
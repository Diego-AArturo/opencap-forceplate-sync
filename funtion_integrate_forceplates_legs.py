import os
import sys
sys.path.append("./../../")
script_folder,_ = os.path.split(os.path.abspath(__file__))
                
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import opensim
import requests
import json
import utils as ut
from utilsProcessing import lowPassFilter
from utilsPlotting import plot_dataframe
import utilsKinematics
from scipy.signal import medfilt, find_peaks
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def guardar_resultados(data, filename):
    with open(filename, 'w') as f:
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


def detect_flat_segment(fall_zone, search_start, min_duration=5, tolerance=2e-3):
    
    """
    Detecta el inicio de una región donde la derivada se mantiene aproximadamente constante (≈ 0).

    Parámetros:
    - signal: array, la señal original.
    - search_start, search_end: índices para limitar la búsqueda.
    - min_duration: número mínimo de muestras con derivada ≈ 0.
    - tolerance: tolerancia para considerar la derivada como 0.

    Retorna:
    - Índice del inicio de la zona plana o None si no se encuentra.
    """

    flat = np.isclose(fall_zone, 0, atol=tolerance).astype(int)

    count = 0
    for i in range(len(flat)):
        if flat[i] == 1:
            count += 1
            if count >= min_duration:
                return search_start + i - min_duration + 1  # índice relativo a la señal original
        else:
            count = 0

    print("No se encontró zona plana suficientemente larga.")
    return None

def calculate_spatial_calibration(heel_pos, cop_pos, force_signal, method='weighted_average'):
    """
    Calcula la calibración espacial entre la posición del talón y el centro de presión.
    
    Parámetros:
    - heel_pos: posiciones del talón [N x 3]
    - cop_pos: posiciones del centro de presión [N x 3] 
    - force_signal: señal de fuerza vertical para ponderar
    - method: método de calibración ('weighted_average', 'contact_only', 'dynamic')
    
    Retorna:
    - offset: vector de corrección [x, y, z]
    """
    
    if method == 'weighted_average':
        # Usar toda la señal ponderada por la fuerza
        weights = np.abs(force_signal) / np.max(np.abs(force_signal))
        weights = weights.reshape(-1, 1)  # Para broadcasting
        
        offset = np.average(heel_pos - cop_pos, axis=0, weights=weights.flatten())
        
    elif method == 'contact_only':
        # Solo usar períodos de contacto significativo
        threshold = 0.1 * np.max(np.abs(force_signal))  # 10% de la fuerza máxima
        contact_mask = np.abs(force_signal) > threshold
        
        if np.any(contact_mask):
            offset = np.mean(heel_pos[contact_mask] - cop_pos[contact_mask], axis=0)
        else:
            offset = np.zeros(3)
            
    elif method == 'dynamic':
        # Calibración dinámica que varía con el tiempo
        window_size = min(50, len(force_signal) // 10)  # Ventana adaptativa
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
        raise ValueError(f"Método no reconocido: {method}")
    
    return offset

def detect_heel_strike(signal,signal2, threshold_factor=0.6, half_sistem=True):
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

    else:
        #=============solo para escalon=================
        min_idx = detect_flat_segment(fall_zone, search_start)
        #================================================
    
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

    return impact_idx



# %% User-defined variables.
if os.path.exists('lag_times.json'):
        with open('lag_times.json', 'r') as f:
            lag_times = json.load(f)
else:
    lag_times = []
def IntegrateForcepalte_legs(session_id, trial_name, force_gdrive_url, participant_id, legs):
    '''legs = leg / R or L'''
    # OpenCap session information from url.
    # View example at https://app.opencap.ai/session/9eea5bf0-a550-4fa5-bc69-f5f072765848

    
    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', participant_id))
    force_dir = os.path.join(data_folder, 'MeasuredForces', trial_name)
    force_path = os.path.join(force_dir, f'{trial_name}_forces.mot')
    os.makedirs(force_dir, exist_ok=True)

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
    lag_time = 1000
    
    lag_times.append({'participante':participant_id, 'movimiento':trial_name, 'lag':lag_time})

    guardar_resultados(lag_times, 'lag_times.json')
    
    pos_Rheel_y = (pos_Rheel[:, 1] - np.min(pos_Rheel[:, 1])) / (np.max(pos_Rheel[:, 1]) - np.min(pos_Rheel[:, 1]))
    pos_Lheel_y = (pos_Lheel[:, 1] - np.min(pos_Lheel[:, 1])) / (np.max(pos_Lheel[:, 1]) - np.min(pos_Lheel[:, 1]))

    

    pos_Rheel_y_filtered=lowPassFilter(time_heel, pos_Rheel_y, lowpass_cutoff_frequency = 3, order=4)
    pos_Lheel_y_filtered=lowPassFilter(time_heel, pos_Lheel_y,lowpass_cutoff_frequency= 3, order=4)
    pos_Rheel_y_filtered = lowpass_filter(pos_Rheel_y_filtered)
    pos_Lheel_y_filtered = lowpass_filter(pos_Lheel_y_filtered)
    
    if any(word in trial_name for word in ['escalon']):
        if legs == 'R': 
            primal_leg_index = detect_heel_strike(pos_Rheel_y_filtered,pos_Lheel_y_filtered,half_sistem=False)
        elif legs == 'L':
            primal_leg_index = detect_heel_strike(pos_Lheel_y_filtered,pos_Rheel_y_filtered,half_sistem=False)
    else:
        if legs == 'R': 
            primal_leg_index = detect_heel_strike(pos_Rheel_y_filtered,pos_Lheel_y_filtered)
        elif legs == 'L':
            primal_leg_index = detect_heel_strike(pos_Lheel_y_filtered,pos_Rheel_y_filtered)
        
    pos_Rheel_x = pos_Rheel[:, 0]
    pos_Lheel_x = pos_Lheel[:, 0]
    pos_Rheel_z = pos_Rheel[:, 2]
    pos_Lheel_z = pos_Lheel[:, 2]
    
    # NOTA: Las coordenadas del talón se asignarán DESPUÉS de calcular el lag_time
    # para garantizar sincronización perfecta

    force_columns = get_columns([legs + '_ground_force_vy'], force_headers)
    forces_for_cross_corr = force_data[:, force_columns]

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
#        com_signal = center_of_mass_acc_filtrada['y'][:, np.newaxis] * mass + mass * 9.8
        force_signal = forces_for_cross_corr_downsamp

#    com_signal = (com_signal - np.min(com_signal)) / (np.max(com_signal) - np.min(com_signal))
    force_signal = (force_signal - np.min(force_signal)) / (np.max(force_signal) - np.min(force_signal))
    forces_for_cross_corr = (forces_for_cross_corr - np.min(forces_for_cross_corr)) / (np.max(forces_for_cross_corr) - np.min(forces_for_cross_corr))

    index_force = detect_upforce(forces_for_cross_corr)
    
    
    lag_time = force_data[index_force, 0] - time_heel[primal_leg_index]
    
    lag_times.append({'participante':participant_id, 'movimiento':trial_name, 'lag':lag_time})

    guardar_resultados(lag_times, 'lag_times.json')
    # print('lag time: ', force_data[index_force, 0] - time_heel[primal_leg_index])
    force_data_new = np.copy(force_data)
    force_data_new[:, 0] = force_data[:, 0] - (force_data[index_force, 0] - time_heel[primal_leg_index])
    
    # ===== SINCRONIZACIÓN DE COORDENADAS DEL TALÓN CON TIEMPO CORREGIDO =====
    # Ahora que tenemos el tiempo de fuerza corregido, sincronizamos las coordenadas X y Z
    time_force_corrected = force_data_new[:, 0]
    
    def robust_heel_interpolation(time_heel, heel_coords, time_force_sync, coord_name):
        """
        Interpolación robusta con validación de calidad y manejo de errores
        """
        # Verificar rangos temporales
        overlap_start = max(time_force_sync[0], time_heel[0])
        overlap_end = min(time_force_sync[-1], time_heel[-1])
        
        if overlap_end <= overlap_start:
            print(f"⚠️ Advertencia: Superposición temporal limitada para {coord_name}")
            method = 'linear'
        else:
            # Verificar densidad de puntos para elegir método de interpolación
            heel_points_in_overlap = np.sum((time_heel >= overlap_start) & (time_heel <= overlap_end))
            if heel_points_in_overlap > 10:
                method = 'cubic'  # Más suave para datos densos
            else:
                method = 'linear'  # Más estable para pocos puntos
        
        try:
            f_interp = interp1d(time_heel, heel_coords, kind=method, 
                               bounds_error=False, fill_value='extrapolate')
            coords_interp = f_interp(time_force_sync)
            
            # Validación de calidad
            if np.any(np.isnan(coords_interp)) or np.any(np.isinf(coords_interp)):
                print(f"⚠️ Interpolación {method} falló para {coord_name}, usando lineal")
                f_interp = interp1d(time_heel, heel_coords, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                coords_interp = f_interp(time_force_sync)
            
            return coords_interp
            
        except Exception as e:
            print(f"❌ Error en interpolación para {coord_name}: {e}")
            # Fallback: repetir el valor más cercano
            return np.full_like(time_force_sync, heel_coords[0])
    
    # Interpolar coordenadas con el tiempo de fuerza ya sincronizado
    pos_Rheel_x_sync = robust_heel_interpolation(time_heel, pos_Rheel_x, time_force_corrected, "RHeel_X")
    pos_Lheel_x_sync = robust_heel_interpolation(time_heel, pos_Lheel_x, time_force_corrected, "LHeel_X")
    pos_Rheel_z_sync = robust_heel_interpolation(time_heel, pos_Rheel_z, time_force_corrected, "RHeel_Z")
    pos_Lheel_z_sync = robust_heel_interpolation(time_heel, pos_Lheel_z, time_force_corrected, "LHeel_Z")
    
    # Asignar coordenadas perfectamente sincronizadas a los datos de fuerza
    gorund_R_leg_x = get_columns(['R' + '_ground_force_vx'], force_headers)
    gorund_L_leg_x = get_columns(['L' + '_ground_force_vx'], force_headers)
    gorund_R_leg_z = get_columns(['R' + '_ground_force_vz'], force_headers)
    gorund_L_leg_z = get_columns(['L' + '_ground_force_vz'], force_headers)
    
    # Asignar arrays completos sincronizados
    force_data_new[:, gorund_R_leg_x] = pos_Rheel_x_sync.reshape(-1, 1)
    force_data_new[:, gorund_L_leg_x] = pos_Lheel_x_sync.reshape(-1, 1)
    force_data_new[:, gorund_R_leg_z] = pos_Rheel_z_sync.reshape(-1, 1)
    force_data_new[:, gorund_L_leg_z] = pos_Lheel_z_sync.reshape(-1, 1)
    
    # ===== CORRECCIÓN ESPACIAL ADICIONAL =====
    # El problema reportado es que las coordenadas X,Z del vector de fuerza 
    # no coinciden espacialmente con el talón, aunque estén sincronizadas temporalmente
    
    # CONFIGURACIÓN: Método de alineación espacial
    # True = Fuerza el vector a aparecer en el talón (visual)
    # False = Calcula offset real entre COP y talón (más preciso físicamente)
    use_heel_coordinates_for_cop = False  # Cambiado a False para probar el método de offset
    
    
        # OPCIÓN B: Calcular y aplicar offset espacial entre COP real y talón
        
        
    # Obtener coordenadas originales del COP (antes de sobrescribir)
    cop_R_x_cols = get_columns(['R' + '_ground_force_px'], force_headers)
    cop_L_x_cols = get_columns(['L' + '_ground_force_px'], force_headers)
    cop_R_z_cols = get_columns(['R' + '_ground_force_pz'], force_headers)
    cop_L_z_cols = get_columns(['L' + '_ground_force_pz'], force_headers)
    
    if cop_R_x_cols and cop_L_x_cols:  # Verificar que existen las columnas
        cop_R_x_original = force_data_new[:, cop_R_x_cols].flatten()
        cop_L_x_original = force_data_new[:, cop_L_x_cols].flatten()
        cop_R_z_original = force_data_new[:, cop_R_z_cols].flatten()
        cop_L_z_original = force_data_new[:, cop_L_z_cols].flatten()
        
        # Calcular offset promedio durante contacto
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
                
                # Aplicar corrección al COP original (mantener dimensionalidad correcta)
                force_data_new[:, cop_R_x_cols] = (cop_R_x_original + offset_R_x).reshape(-1, 1)
                force_data_new[:, cop_R_z_cols] = (cop_R_z_original + offset_R_z).reshape(-1, 1)
                
                # print(f"   - Offset pierna derecha: X={offset_R_x:.3f}m, Z={offset_R_z:.3f}m")
            
            if np.any(contact_L):
                offset_L_x = np.mean(pos_Lheel_x_sync[contact_L] - cop_L_x_original[contact_L])
                offset_L_z = np.mean(pos_Lheel_z_sync[contact_L] - cop_L_z_original[contact_L])
                
                # Aplicar corrección al COP original (mantener dimensionalidad correcta)
                force_data_new[:, cop_L_x_cols] = (cop_L_x_original + offset_L_x).reshape(-1, 1)
                force_data_new[:, cop_L_z_cols] = (cop_L_z_original + offset_L_z).reshape(-1, 1)
                
                # print(f"   - Offset pierna izquierda: X={offset_L_x:.3f}m, Z={offset_L_z:.3f}m")

    # Información de diagnóstico
    # print(f"✅ Sincronización espacial completada:")
    # print(f"   - Lag time aplicado: {lag_time:.4f}s")
    # print(f"   - Puntos de fuerza: {len(time_force_corrected)}")
    # print(f"   - Puntos de cinemática: {len(time_heel)}")
    # print(f"   - Rango temporal fuerza corregido: {time_force_corrected[0]:.3f} - {time_force_corrected[-1]:.3f}s")
    # print(f"   - Rango temporal talón: {time_heel[0]:.3f} - {time_heel[-1]:.3f}s")
    # ===================================================================

    if visualize_synchronization:
        # print("vizulize")
        plt.figure()
        plt.plot(kinematics.time,com_signal,label='COM acceleration')

        # plt.plot(kinematics.time,center_of_mass_pos['y'],label='COM acceleration')
        plt.plot(force_data_new[:, 0], forces_for_cross_corr, label='vGRF')
        plt.legend()
        plt.grid()
        plt.show()
    
    if save_plot:
        save_folder = os.path.join("graficas", participant_id)
        os.makedirs(save_folder, exist_ok=True)
        
        save_path = os.path.join(save_folder, f"{trial_name}_corte.png")
        
        plt.figure(figsize=(8, 5))
        # plt.plot(time_cinematica_filtrada, com_signal , label='COM')
        # plt.plot(force_data_new[:, 0], forces_for_cross_corr, label='vGRF')
        plt.plot(time_heel, pos_Rheel_y_filtered, label='RHeel')
        plt.plot(time_heel, pos_Lheel_y_filtered, label='LHeel')
        plt.axvline(x=time_heel[primal_leg_index], color='r', linestyle='--', label='Caida del talón')
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
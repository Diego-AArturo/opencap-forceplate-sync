import os
import sys
sys.path.append("./../../")
script_folder,_ = os.path.split(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import opensim
# import requests
import matplotlib.pyplot as plt
import src.utils.utils as ut
from src.utils import utilsKinematics
import re
from src.utils.utils import storage_to_numpy
from openpyxl import load_workbook
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import seaborn as sns

# Expresión regular para eliminar el número final del nombre del trial
def extract_movement_name(trial_name):
    return re.sub(r'_\d+$', '', trial_name)

def safe_correlation(x, y):
    """Calcula correlación de forma segura evitando warnings por valores constantes."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    # Verificar si hay variabilidad suficiente
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    try:
        corr = np.corrcoef(x, y)[0,1]
        return 0.0 if np.isnan(corr) else corr
    except:
        return 0.0

def safe_polyfit_slope(x, y):
    """Calcula la pendiente usando polyfit de forma segura."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    # Verificar si hay variabilidad suficiente en y
    if np.std(y) < 1e-10:
        return 0.0
    
    try:
        slope = np.polyfit(x, y, 1)[0]
        return 0.0 if np.isnan(slope) else slope
    except:
        return 0.0

def identify_main_peak_improved(signal, min_value, baseline):
    """Identifica el pico principal usando múltiples criterios."""
    from scipy.signal import find_peaks
    
    # Buscar picos significativos con criterios más estrictos
    peak_height_threshold = min_value + baseline * 0.3  # Más alto que antes
    peaks, properties = find_peaks(signal, height=peak_height_threshold, distance=100)  # Mayor distancia
    
    if len(peaks) == 0:
        return np.argmax(signal)
    
    # Evaluar cada pico por múltiples criterios
    peak_scores = []
    for peak_idx in peaks:
        height = signal[peak_idx]
        
        # Criterio 1: Altura absoluta (40%)
        height_score = (height - min_value) / baseline
        
        # Criterio 2: Prominencia (30%)
        left_min = np.min(signal[max(0, peak_idx-200):peak_idx]) if peak_idx > 200 else min_value
        right_min = np.min(signal[peak_idx:min(len(signal), peak_idx+200)]) if peak_idx < len(signal)-200 else min_value
        prominence = height - max(left_min, right_min)
        prominence_score = prominence / baseline if baseline > 0 else 0
        
        # Criterio 3: Posición temporal (30%) - preferir picos en el medio-final
        position_score = peak_idx / len(signal)  # 0 a 1
        if position_score < 0.3:  # Penalizar picos muy tempranos
            position_score *= 0.5
        
        total_score = height_score * 0.4 + prominence_score * 0.3 + position_score * 0.3
        peak_scores.append(total_score)
    
    # Seleccionar el pico con mayor puntuación
    best_peak_idx = np.argmax(peak_scores)
    return peaks[best_peak_idx]

def adaptive_detection_criteria(signal, baseline):
    """Determina criterios adaptativos según las características de la señal."""
    # Analizar la "complejidad" de la señal
    signal_complexity = np.std(np.diff(signal)) / baseline if baseline > 0 else 0
    
    if signal_complexity > 0.1:  # Señal muy ruidosa/compleja
        # Criterios más estrictos
        return {
            'min_slope': 0.1,
            'min_consistency': 0.6,
            'min_peak_reach': 0.8,
            'search_range': 300,
            'future_trend_threshold': 0.03
        }
    else:  # Señal más limpia
        # Criterios más permisivos
        return {
            'min_slope': 0.05,
            'min_consistency': 0.4,
            'min_peak_reach': 0.6,
            'search_range': 400,
            'future_trend_threshold': 0.02
        }

def validate_detected_phase(signal, inicio_idx, max_peak_idx, baseline, min_value):
    """Valida que la fase detectada sea razonable."""
    
    # Verificar que no esté demasiado lejos del pico principal
    time_to_peak = max_peak_idx - inicio_idx
    if time_to_peak < 50:  # Muy cerca del pico
        return False, "Demasiado cerca del pico"
    
    if time_to_peak > 600:  # Muy lejos del pico
        return False, "Demasiado lejos del pico"
    
    # Verificar que haya una tendencia ascendente clara
    path_to_peak = signal[inicio_idx:max_peak_idx]
    if len(path_to_peak) > 10:
        overall_trend = safe_polyfit_slope(range(len(path_to_peak)), path_to_peak)
        if overall_trend < 0.02:  # Tendencia muy débil
            return False, "Tendencia ascendente insuficiente"
    
    # Verificar que el valor inicial no sea demasiado alto
    max_reasonable_start = min_value + baseline * 0.2
    if signal[inicio_idx] > max_reasonable_start:
        return False, "Valor inicial demasiado alto"
    
    return True, "Válido"

def conservative_detection(signal, max_peak_idx, criteria, baseline_threshold, max_start_threshold, min_value, baseline):
    """Detección conservadora con criterios estrictos."""
    from scipy.signal import savgol_filter
    smoothed_signal = savgol_filter(signal, window_length=15, polyorder=3)
    
    for i in range(max_peak_idx, max(0, max_peak_idx - criteria['search_range']), -1):
        if signal[i] <= baseline_threshold and signal[i] <= max_start_threshold:
            if i < len(signal) - 50:
                future_window = min(100, len(signal) - i - 1)
                if future_window > 30:
                    future_values = signal[i:i+future_window]
                    trend_slope = safe_polyfit_slope(range(len(future_values)), future_values)
                    trend_consistency = safe_correlation(range(len(future_values)), future_values)
                    
                    max_future_value = np.max(future_values)
                    
                    if (trend_slope > criteria['min_slope'] and 
                        trend_consistency > criteria['min_consistency'] and 
                        max_future_value >= signal[max_peak_idx] * criteria['min_peak_reach']):
                        
                        path_to_peak = signal[i:max_peak_idx]
                        if len(path_to_peak) > 0:
                            max_in_path = np.max(path_to_peak)
                            if max_in_path >= signal[max_peak_idx] * 0.8:
                                return i
    return None

def moderate_detection(signal, max_peak_idx, criteria, baseline_threshold, max_start_threshold, min_value, baseline):
    """Detección moderada con criterios medios."""
    from scipy.signal import savgol_filter
    smoothed_signal = savgol_filter(signal, window_length=15, polyorder=3)
    derivative = np.gradient(smoothed_signal)
    
    for i in range(max_peak_idx, max(0, max_peak_idx - criteria['search_range']), -1):
        if i >= 15 and i < len(derivative) - 15:
            past_trend = np.mean(derivative[i-15:i])
            future_trend = np.mean(derivative[i:i+15])
            
            if past_trend <= 0.005 and future_trend > criteria['future_trend_threshold']:
                if (signal[i] < min_value + baseline * 0.15 and 
                    signal[i] <= max_start_threshold):
                    
                    future_window = min(80, len(signal) - i - 1)
                    if future_window > 20:
                        max_future = np.max(signal[i:i+future_window])
                        if max_future >= signal[max_peak_idx] * (criteria['min_peak_reach'] - 0.1):
                            return i
    return None

def permissive_detection(signal, max_peak_idx, criteria, baseline_threshold, max_start_threshold, min_value, baseline):
    """Detección permisiva con criterios relajados."""
    
    # Buscar cualquier punto que tenga pendiente ascendente hacia el pico
    for i in range(max_peak_idx, max(0, max_peak_idx - criteria['search_range']), -1):
        if i < len(signal) - 15:
            local_window = min(15, len(signal) - i - 1)
            if local_window > 8:
                future_values = signal[i:i+local_window]
                local_slope = safe_polyfit_slope(range(len(future_values)), future_values)
                
                if local_slope > (criteria['min_slope'] * 0.5):  # Criterio más permisivo
                    max_in_future = np.max(signal[i:min(i+150, len(signal))])
                    if max_in_future >= signal[max_peak_idx] * (criteria['min_peak_reach'] - 0.2):
                        if signal[i] <= max_start_threshold * 1.2:  # Más permisivo
                            return i
    return None

def detect_max_slope_onset(signal, window_length=10, polyorder=3, slope_threshold=0.15):
    """
    Detecta el inicio de la pendiente más pronunciada en una señal.
    
    Parámetros:
    - signal: array de la señal original
    - window_length: tamaño de ventana para suavizado (debe ser impar)
    - polyorder: orden del polinomio para Savitzky-Golay
    - slope_threshold: umbral para considerar que empieza una subida sostenida
    
    Retorna:
    - Índice de inicio de la pendiente más pronunciada
    """
    
    # Suavizar la señal
    smoothed = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
    
    slope = np.gradient(smoothed)

    baseline = max(signal) - min(signal)
    min_value = min(signal)
    # Threshold más bajo y adaptativo
    threshold = baseline * 0.15 + min_value
    
    # Buscar la máxima pendiente en toda la señal, no solo después del índice 50
    max_slope_idx = np.argmax(slope)
    onset_idx = max_slope_idx
    
    # Threshold adaptativo para la pendiente basado en la máxima pendiente encontrada
    max_slope_value = slope[max_slope_idx]
    adaptive_slope_threshold = max(slope_threshold, max_slope_value * 0.1)

    # Buscar hacia atrás desde la máxima pendiente
    for i in range(max_slope_idx, 0, -1):
        if slope[i] < adaptive_slope_threshold and smoothed[i] < threshold:
            onset_idx = i
            break
    
    # Refinamiento mejorado: detectar el verdadero punto de transición
    transition_onset = onset_idx
    
    # Buscar el punto donde la tendencia cambia de constante/negativa a positiva
    for i in range(onset_idx, max(0, onset_idx - 80), -1):
        if i >= 8 and i < len(slope) - 8:
            # Analizar la tendencia de la derivada antes y después
            past_slope_trend = np.mean(slope[i-8:i])
            future_slope_trend = np.mean(slope[i:i+8])
            
            # Detectar transición clara en la pendiente
            if past_slope_trend <= 0.01 and future_slope_trend > 0.03:
                transition_onset = i
                break
            
            # También detectar cambios significativos en la pendiente
            if future_slope_trend - past_slope_trend > adaptive_slope_threshold * 0.5:
                transition_onset = i
                break
    
    onset_idx = transition_onset
        

    


    # plt.figure(figsize=(8, 4))
    # plt.plot(signal, label='Original')
    # plt.plot(slope, label='derivada')
    # plt.plot(smoothed, label='Suavizada', linewidth=2)
    # plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    # plt.axvline(onset_idx, color='green', linestyle='--', label='Inicio de pendiente fuerte')
    # plt.axvline(max_slope_idx, color='red', linestyle='--', label='Máx. pendiente')
    # plt.legend()
    # plt.grid(True)
    # plt.title("Detección de pendiente más pronunciada")
    # plt.xlabel("Índice")
    # plt.tight_layout()
    # plt.show()

    return onset_idx


def detect_phases(signal, threshold_factor=0.3):
    """
    Detecta las fases de movimiento en una señal de ángulo articular.
    
    Parámetros:
    - signal: array, señal vertical del talón normalizada.
    - threshold_factor: float, para ajustar la sensibilidad del despegue (reducido de 0.8 a 0.3).
    
    Retorna:
    - Tupla (inicio_idx, medio_idx, fin_idx) o None si no se detecta movimiento
    """
    
    # Parámetros ajustados para mejor detección
    descend_window = 15
    stable_window = 8
    stable_tol = 2
    min_threshold = 0.06
    
    # Calcular estadísticas de la señal
    baseline = max(signal) - min(signal)
    min_value = min(signal)
    max_value = max(signal)
    
    # Encontrar el pico principal usando criterios mejorados
    max_peak_idx = identify_main_peak_improved(signal, min_value, baseline)
    
    # Thresholds adaptativos mucho más bajos para detectar el verdadero inicio
    main_threshold = baseline * threshold_factor + min_value  # Threshold principal más bajo
    min_threshold_value = baseline * min_threshold + min_value
    
    # Threshold ultra-sensible para detectar el verdadero inicio desde la línea base
    baseline_threshold = baseline * 0.02 + min_value  # Solo 2% por encima del mínimo
    
    # Threshold máximo absoluto para evitar inicios demasiado altos
    # El inicio no debería estar por encima del 15% del rango total desde el mínimo
    max_start_threshold = min_value + baseline * 0.15
    
    # Threshold permisivo para seguimiento de pendiente principal
    permissive_threshold = min_value + baseline * 0.25
    
    
    # Sistema flexible de detección con múltiples niveles
    inicio_idx = None
    
    # Obtener criterios adaptativos basados en la complejidad de la señal
    criteria = adaptive_detection_criteria(signal, baseline)
    
    
    print(f"Criterios adaptativos: complejidad={'alta' if criteria['min_slope'] > 0.05 else 'baja'}")
    
    # NIVEL 1: Detección conservadora (criterios estrictos)
    inicio_idx = conservative_detection(signal, max_peak_idx, criteria, baseline_threshold, max_start_threshold, min_value, baseline)
    
    if inicio_idx is not None:
        is_valid, reason = validate_detected_phase(signal, inicio_idx, max_peak_idx, baseline, min_value)
        if is_valid:
            print(f"Detección conservadora exitosa en índice {inicio_idx} - {reason}")
        else:
            print(f"Detección conservadora rechazada: {reason}")
            inicio_idx = None
    
    # NIVEL 2: Detección moderada (criterios medios)
    if inicio_idx is None:
        inicio_idx = moderate_detection(signal, max_peak_idx, criteria, baseline_threshold, max_start_threshold, min_value, baseline)
        
        if inicio_idx is not None:
            is_valid, reason = validate_detected_phase(signal, inicio_idx, max_peak_idx, baseline, min_value)
            if is_valid:
                print(f"Detección moderada exitosa en índice {inicio_idx} - {reason}")
            else:
                print(f"Detección moderada rechazada: {reason}")
                inicio_idx = None
    
    # NIVEL 3: Detección permisiva (criterios relajados)
    if inicio_idx is None:
        inicio_idx = permissive_detection(signal, max_peak_idx, criteria, baseline_threshold, max_start_threshold, min_value, baseline)
        
        if inicio_idx is not None:
            print(f"Detección permisiva exitosa en índice {inicio_idx}")
        else:
            print("No se pudo detectar inicio con ningún método")
    
    # NIVEL 4: Último recurso - método original como fallback
    if inicio_idx is None:
        search_start = max(0, max_peak_idx - 200)
        search_end = min(len(signal), max_peak_idx + 50)
        
        inicio_idx = detect_max_slope_onset(signal[search_start:search_end])
        if inicio_idx is not None:
            inicio_idx = inicio_idx + search_start
            print(f"Fallback: detección por pendiente en índice {inicio_idx}")
    
    if inicio_idx is None:
        print("FALLO: No se pudo detectar el inicio del movimiento")
        return None 

    # Buscar el final desde el pico hacia adelante
    fin_idx = None
    
    # Primero buscar cuando la señal baja significativamente
    for i in range(max_peak_idx, len(signal) - stable_window):
        if signal[i] < main_threshold:
            # Verificar estabilidad después de este punto
            window = signal[i:i + stable_window]
            if np.max(window) - np.min(window) < stable_tol:
                fin_idx = i
                break
    
    # Si no encuentra estabilidad, buscar simplemente cuando baja del threshold
    if fin_idx is None:
        for i in range(max_peak_idx, len(signal)):
            if signal[i] < min_threshold_value:
                fin_idx = i
                break
    
    if fin_idx is None:
        fin_idx = len(signal) - 1
    
    # Calcular punto medio
    med_idx = max((inicio_idx + fin_idx) // 2, max_peak_idx)
    
    # Visualización para debugging
    plt.axhline(y=main_threshold, color='r', linestyle='--', label=f'Threshold principal ({main_threshold:.2f})')
    plt.axhline(y=min_threshold_value, color='orange', linestyle='--', label=f'Threshold mínimo ({min_threshold_value:.2f})')
    plt.axhline(y=baseline_threshold, color='cyan', linestyle=':', label=f'Baseline threshold ({baseline_threshold:.2f})')
    plt.axhline(y=max_start_threshold, color='purple', linestyle='-.', label=f'Max start threshold ({max_start_threshold:.2f})')
    plt.axhline(y=permissive_threshold, color='brown', linestyle=':', alpha=0.7, label=f'Permissive threshold ({permissive_threshold:.2f})')
    
    
    
    return inicio_idx, med_idx, fin_idx

def extract_data_t(session_id, trial_name, leg='r', time_import=None):
    #'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38'
    #'escalon_derecho_1'
    # leg = 'l'
    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', session_id))
    gravity = 9.81
    if not os.path.exists(os.path.join(data_folder, 'MarkerData')):
        _, model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name, _ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

    # Iniciar análisis cinemático
    kinematics = utilsKinematics.kinematics(data_folder, trial_name, 
                                            modelName=model_name, 
                                            lowpass_cutoff_frequency_for_coordinate_values=10)

    opencap_settings = ut.get_main_settings(data_folder, trial_name)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))

    # Ruta del archivo .mot
    mot_file_path = os.path.join(data_folder, 'OpenSimData', 'Kinematics', f'{trial_name}.mot')

    # Cargar el archivo .mot utilizando OpenSim
    mot_table = opensim.TimeSeriesTable(mot_file_path)
    
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))
    mass = opencap_metadata['mass_kg']

    mot_df = pd.DataFrame(mot_table.getMatrix().to_numpy())
    mot_df.columns = mot_table.getColumnLabels()
    mot_df['time'] = mot_table.getIndependentColumn()

    try:
        force_path = os.path.join(data_folder, "MeasuredForces", trial_name, f"{trial_name}_syncd_forces.mot")
        
        forces = storage_to_numpy(force_path)
        force_data = forces.view(np.float64).reshape(forces.shape + (-1,))
        headers = forces.dtype.names

        time_real = force_data[:, 0]  # Tiempo de la fuerza
        # Extraer señal de fuerza vertical (R + L)
        vyr_idx = [i for i, name in enumerate(headers) if "R_ground_force_vy" in name]
        vyl_idx = [i for i, name in enumerate(headers) if "L_ground_force_vy" in name]
        vy_idx = [i for i, name in enumerate(headers) if "ground_force_vy" in name]
        # Extraer señal de fuerza x (R + L)
        vxr_idx = [i for i, name in enumerate(headers) if "R_ground_force_vx" in name]
        vxl_idx = [i for i, name in enumerate(headers) if "L_ground_force_vx" in name]
        vx_idx = [i for i, name in enumerate(headers) if "ground_force_vx" in name]
        # Extraer señal de fuerza z (R + L)
        vzr_idx = [i for i, name in enumerate(headers) if "R_ground_force_vz" in name]
        vzl_idx = [i for i, name in enumerate(headers) if "L_ground_force_vz" in name]
        vz_idx = [i for i, name in enumerate(headers) if "ground_force_vz" in name]
        # Fuerzas en Y (vertical)
        real_vgrfy_n = np.sum(force_data[:, vy_idx], axis=1)  # N
        real_vgrfy_r_n = force_data[:, vyr_idx].squeeze()     # N
        # print(real_vgrfy_r_n)
        real_vgrfy_l_n = force_data[:, vyl_idx].squeeze()     # N

        # Fuerzas en X (medio-lateral)
        real_vgrfx_n = np.sum(force_data[:, vx_idx], axis=1)  # N
        real_vgrfx_r_n = force_data[:, vxr_idx].squeeze()     # N
        real_vgrfx_l_n = force_data[:, vxl_idx].squeeze()     # N

        # Fuerzas en Z (antero-posterior)
        real_vgrfz_n = np.sum(force_data[:, vz_idx], axis=1)  # N
        real_vgrfz_r_n = force_data[:, vzr_idx].squeeze()     # N
        real_vgrfz_l_n = force_data[:, vzl_idx].squeeze()     # N

        # Normalización a % del peso corporal
        bw = gravity * mass
        real_vgrfy_r = real_vgrfy_r_n / bw
        real_vgrfy_l = real_vgrfy_l_n / bw
        real_vgrfy_t = real_vgrfy_n / bw

        real_vgrfx_r = real_vgrfx_r_n / bw
        real_vgrfx_l = real_vgrfx_l_n / bw
        real_vgrfx_t = real_vgrfx_n / bw

        real_vgrfz_r = real_vgrfz_r_n / bw
        real_vgrfz_l = real_vgrfz_l_n / bw
        real_vgrfz_t = real_vgrfz_n / bw

        # Crear DataFrame completo
        df_dyn = pd.DataFrame({
            'time': time_real,

            # Fuerzas Y en N y %BW
            'vgrfy_r_n': real_vgrfy_r_n,
            'vgrfy_l_n': real_vgrfy_l_n,
            'vgrfy_t_n': real_vgrfy_n,
            'vgrfy_r_bw': real_vgrfy_r,
            'vgrfy_l_bw': real_vgrfy_l,
            'vgrfy_t_bw': real_vgrfy_t,

            # Fuerzas X en N y %BW
            'vgrfx_r_n': real_vgrfx_r_n,
            'vgrfx_l_n': real_vgrfx_l_n,
            'vgrfx_t_n': real_vgrfx_n,
            'vgrfx_r_bw': real_vgrfx_r,
            'vgrfx_l_bw': real_vgrfx_l,
            'vgrfx_t_bw': real_vgrfx_t,

            # Fuerzas Z en N y %BW
            'vgrfz_r_n': real_vgrfz_r_n,
            'vgrfz_l_n': real_vgrfz_l_n,
            'vgrfz_t_n': real_vgrfz_n,
            'vgrfz_r_bw': real_vgrfz_r,
            'vgrfz_l_bw': real_vgrfz_l,
            'vgrfz_t_bw': real_vgrfz_t,
        })

        df_dyn = df_dyn.round(3)
    except Exception as e:
        
        print('error: ',e)
        df_dyn = pd.DataFrame({
        'time': mot_df['time'],
        'real_vgrf_r':  np.zeros(len(mot_df['time'])),
        'real_vgrf_l': np.zeros(len(mot_df['time'])),
        'real_vgrf_t': np.zeros(len(mot_df['time'])),

        'vgrfy_r_n': np.zeros(len(mot_df['time'])),
        'vgrfy_l_n': np.zeros(len(mot_df['time'])),
        'vgrfy_t_n': np.zeros(len(mot_df['time'])),
        'vgrfy_r_bw': np.zeros(len(mot_df['time'])),
        'vgrfy_l_bw': np.zeros(len(mot_df['time'])),
        'vgrfy_t_bw': np.zeros(len(mot_df['time'])),

        # Fuerzas X en N y %BW
        'vgrfx_r_n': np.zeros(len(mot_df['time'])),
        'vgrfx_l_n': np.zeros(len(mot_df['time'])),
        'vgrfx_t_n': np.zeros(len(mot_df['time'])),
        'vgrfx_r_bw': np.zeros(len(mot_df['time'])),
        'vgrfx_l_bw': np.zeros(len(mot_df['time'])),
        'vgrfx_t_bw': np.zeros(len(mot_df['time'])),

        # Fuerzas Z en N y %BW
        'vgrfz_r_n': np.zeros(len(mot_df['time'])),
        'vgrfz_l_n': np.zeros(len(mot_df['time'])),
        'vgrfz_t_n': np.zeros(len(mot_df['time'])),
        'vgrfz_r_bw': np.zeros(len(mot_df['time'])),
        'vgrfz_l_bw': np.zeros(len(mot_df['time'])),
        'vgrfz_t_bw': np.zeros(len(mot_df['time'])),
    })
    df_dyn = df_dyn.round(3)
    # Convertir la tabla a un DataFrame de pandas
    # mot_df = pd.DataFrame(mot_table.getMatrix().to_numpy())
    # mot_df.columns = mot_table.getColumnLabels()
    # mot_df['time'] = mot_table.getIndependentColumn()

    # Reordenar las columnas para que 'time' sea la primera columna
    mot_df = mot_df[['time'] + [col for col in mot_df.columns if col != 'time']]
    df_mot = mot_df.round(3)
    if not time_import:
        # if session_id in ['P6','P13'] and leg == 'l':
        #     despegue_idx,med_idx,min_idx = detect_phases(df_mot[f'knee_angle_{leg}'][:-100].to_numpy())
        #     despegue_time,med_time,min_time = df_mot['time'][despegue_idx],df_mot['time'][med_idx],df_mot['time'][min_idx]
        # else:
        despegue_idx,med_idx,min_idx = detect_phases(df_mot[f'knee_angle_{leg}'].to_numpy())
        despegue_time,med_time,min_time = df_mot['time'][despegue_idx],df_mot['time'][med_idx],df_mot['time'][min_idx]
    else:
        # Convertir tiempos importados a índices
        despegue_time = time_import[0]
        med_time = time_import[1]
        min_time = time_import[2]
        
        # Encontrar los índices más cercanos a los tiempos especificados
        despegue_idx = (df_mot['time'] - despegue_time).abs().idxmin()
        med_idx = (df_mot['time'] - med_time).abs().idxmin()
        min_idx = (df_mot['time'] - min_time).abs().idxmin()
        print('index:',despegue_idx,med_idx,min_idx)
        print('time:',despegue_time,med_time,min_time)
    
    
    
    
    save_folder = os.path.join("graficas_rod_gfr", session_id)
    os.makedirs(save_folder, exist_ok=True)
        
    save_path = os.path.join(save_folder, f"{trial_name}.png")
    # Subplot 1: Rodilla derecha
    plt.plot(df_mot['time'], df_mot['knee_angle_r'],label='Rodilla derecha')
    plt.plot(df_mot['time'], df_mot['knee_angle_l'],label='Rodilla izquierda')

    plt.axvline(x=despegue_time, color='orange', linestyle='--', label='Fase 1')
    plt.axvline(x=med_time, color='green', linestyle='--', label='Fase 2')
    plt.axvline(x=min_time, color='red', linestyle='--', label='Fase 3')
    plt.title(f'Ángulo de la rodilla {session_id}')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=8, frameon=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

    #=======================================================
    desp_fidx = (df_dyn['time'] - despegue_time).abs().idxmin()
    med_fidx = (df_dyn['time'] - med_time).abs().idxmin()
    min_fidx = (df_dyn['time'] - min_time).abs().idxmin()
    #========================================================
    # save_folder = os.path.join("graficas_gfr", session_id)
    # os.makedirs(save_folder, exist_ok=True)
        
    # save_path = os.path.join(save_folder, f"{trial_name}.png")
    # # Subplot 1: Rodilla derecha
    # plt.plot(df_dyn['time'], df_dyn['real_vgrf_r'],label='Derecha')
    # plt.plot(df_dyn['time'], df_dyn['real_vgrf_l'],label='Izquierda')

    # plt.axvline(x=df_dyn['time'][desp_fidx], color='orange', linestyle='--', label='Fase 1')
    # plt.axvline(x=df_dyn['time'][med_fidx], color='green', linestyle='--', label='Fase 2')
    # plt.axvline(x=df_dyn['time'][min_fidx], color='red', linestyle='--', label='Fase 3')
    # plt.title(f'Ángulo de la rodilla {session_id}')
    # plt.grid(True)
    # plt.legend(loc='upper left', fontsize=8, frameon=True)

    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    #=========================================================
    
    global signal_part
    
    signal_part= {
    'time': df_dyn.loc[desp_fidx:min_fidx, 'time'],
    'vgrfy_r(N)': df_dyn.loc[desp_fidx:min_fidx, 'vgrfy_r_n'],
    'vgrfy_l(N)': df_dyn.loc[desp_fidx:min_fidx, 'vgrfy_l_n'],
    
    'vgrfx_r(N)': df_dyn.loc[desp_fidx:min_fidx, 'vgrfx_r_n'],
    'vgrfx_l(N)': df_dyn.loc[desp_fidx:min_fidx, 'vgrfx_l_n'],
    
    'vgrfz_r(N)': df_dyn.loc[desp_fidx:min_fidx, 'vgrfz_r_n'],
    'vgrfz_l(N)': df_dyn.loc[desp_fidx:min_fidx, 'vgrfz_l_n'],
    }

    #=========================================================
    save_folder = os.path.join("graficas_gfr_all", session_id)
    os.makedirs(save_folder, exist_ok=True)

    # Ruta de guardado del gráfico
    save_path = os.path.join(save_folder, f"{trial_name}.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ejes = ['Y', 'X', 'Z']
    colores = ['b', 'g', 'm']

    # Datos por eje
    ejes_datos = {
        'Y': ('vgrfy_r_n', 'vgrfy_l_n'), #vertical
        'X': ('vgrfx_r_n', 'vgrfx_l_n'), #antero-posterior
        'Z': ('vgrfz_r_n', 'vgrfz_l_n') #medio-lateral
    }

    for i, eje in enumerate(ejes):
        col_r, col_l = ejes_datos[eje]
        ax = axes[i]
        ax.plot(df_dyn['time'], df_dyn[col_r], label='Derecha', color='steelblue')
        ax.plot(df_dyn['time'], df_dyn[col_l], label='Izquierda', color='darkorange')

        # Marcar fases
        ax.axvline(x=df_dyn['time'][desp_fidx], color='orange', linestyle='--', label='Fase 1' if i == 0 else "")
        ax.axvline(x=df_dyn['time'][med_fidx], color='green', linestyle='--', label='Fase 2' if i == 0 else "")
        ax.axvline(x=df_dyn['time'][min_fidx], color='red', linestyle='--', label='Fase 3' if i == 0 else "")

        ax.set_ylabel(f'Fuerza {eje} (%BW)')
        ax.grid(True)

    # Títulos y leyenda
    axes[0].set_title(f'Fuerzas de Reacción - {session_id} | {trial_name}')
    axes[-1].set_xlabel('Tiempo (s)')
    axes[0].legend(loc='upper right', fontsize=8, frameon=True)

    # Guardar figura
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
#=====================================================
    # Interpolación para igualar longitudes
    # interp_real = interp1d(df_dyn['time'], df_dyn['vgrfy_r_bw'], kind='linear', bounds_error=False, fill_value="extrapolate")
    # real_resampled_r = interp_real(df_mot['time'])

    #========================================================
    # save_folder = os.path.join("graficas_gfr_vs_rod", session_id)
    # os.makedirs(save_folder, exist_ok=True)
        
    # save_path = os.path.join(save_folder, f"{trial_name}.png")
    # # Subplot 1: Rodilla derecha
    # plt.plot(df_mot['knee_angle_r'][despegue_idx:min_idx],real_resampled_r[despegue_idx:min_idx],'o')
    # plt.plot(real_resampled_r[despegue_idx:min_idx])
    # plt.plot(df_mot['knee_angle_r'][despegue_idx:min_idx])


    # plt.axvline(x=[despegue_idx], color='orange', linestyle='--', label='Fase 1')
    # plt.axvline(x=[med_idx], color='green', linestyle='--', label='Fase 2')
    # plt.axvline(x=[min_idx], color='red', linestyle='--', label='Fase 3')
    # plt.title(f'GFR vz ROD {session_id}')
    # plt.grid(True)
    # plt.legend(loc='upper left', fontsize=8, frameon=True)

    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    #=============================================================================================0
    result_row = {
        'time1': despegue_time,
        'knee_angle_r1': df_mot.loc[despegue_idx, 'knee_angle_r'],
        'knee_angle_l1': df_mot.loc[despegue_idx, 'knee_angle_l'],
        'vgrfy_r1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfy_r_bw'],
        'vgrfy_l1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfy_l_bw'],
        'vgrfy_t1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfy_t_bw'],
        'vgrfy_r1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfy_r_n'],
        'vgrfy_l1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfy_l_n'],
        'vgrfy_t1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfy_t_n'],

        'vgrfx_r1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfx_r_bw'],
        'vgrfx_l1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfx_l_bw'],
        'vgrfx_t1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfx_t_bw'],
        'vgrfx_r1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfx_r_n'],
        'vgrfx_l1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfx_l_n'],
        'vgrfx_t1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfx_t_n'],

        'vgrfz_r1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfz_r_bw'],
        'vgrfz_l1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfz_l_bw'],
        'vgrfz_t1 (%BW)': df_dyn.loc[desp_fidx, 'vgrfz_t_bw'],
        'vgrfz_r1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfz_r_n'],
        'vgrfz_l1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfz_l_n'],
        'vgrfz_t1 (NEWTON)': df_dyn.loc[desp_fidx, 'vgrfz_t_n'],

        'time2': med_time,
        'knee_angle_r2': df_mot.loc[med_idx, 'knee_angle_r'],
        'knee_angle_l2': df_mot.loc[med_idx, 'knee_angle_l'],
        'vgrfy_r2 (%BW)': df_dyn.loc[med_fidx, 'vgrfy_r_bw'],
        'vgrfy_l2 (%BW)': df_dyn.loc[med_fidx, 'vgrfy_l_bw'],
        'vgrfy_t2 (%BW)': df_dyn.loc[med_fidx, 'vgrfy_t_bw'],
        'vgrfy_r2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfy_r_n'],
        'vgrfy_l2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfy_l_n'],
        'vgrfy_t2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfy_t_n'],

        'vgrfx_r2 (%BW)': df_dyn.loc[med_fidx, 'vgrfx_r_bw'],
        'vgrfx_l2 (%BW)': df_dyn.loc[med_fidx, 'vgrfx_l_bw'],
        'vgrfx_t2 (%BW)': df_dyn.loc[med_fidx, 'vgrfx_t_bw'],
        'vgrfx_r2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfx_r_n'],
        'vgrfx_l2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfx_l_n'],
        'vgrfx_t2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfx_t_n'],

        'vgrfz_r2 (%BW)': df_dyn.loc[med_fidx, 'vgrfz_r_bw'],
        'vgrfz_l2 (%BW)': df_dyn.loc[med_fidx, 'vgrfz_l_bw'],
        'vgrfz_t2 (%BW)': df_dyn.loc[med_fidx, 'vgrfz_t_bw'],
        'vgrfz_r2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfz_r_n'],
        'vgrfz_l2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfz_l_n'],
        'vgrfz_t2 (NEWTON)': df_dyn.loc[med_fidx, 'vgrfz_t_n'],

        'time3': min_time,
        'knee_angle_r3': df_mot.loc[min_idx, 'knee_angle_r'],
        'knee_angle_l3': df_mot.loc[min_idx, 'knee_angle_l'],
        'vgrfy_r3 (%BW)': df_dyn.loc[min_fidx, 'vgrfy_r_bw'],
        'vgrfy_l3 (%BW)': df_dyn.loc[min_fidx, 'vgrfy_l_bw'],
        'vgrfy_t3 (%BW)': df_dyn.loc[min_fidx, 'vgrfy_t_bw'],
        'vgrfy_r3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfy_r_n'],
        'vgrfy_l3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfy_l_n'],
        'vgrfy_t3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfy_t_n'],

        'vgrfx_r3 (%BW)': df_dyn.loc[min_fidx, 'vgrfx_r_bw'],
        'vgrfx_l3 (%BW)': df_dyn.loc[min_fidx, 'vgrfx_l_bw'],
        'vgrfx_t3 (%BW)': df_dyn.loc[min_fidx, 'vgrfx_t_bw'],
        'vgrfx_r3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfx_r_n'],
        'vgrfx_l3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfx_l_n'],
        'vgrfx_t3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfx_t_n'],

        'vgrfz_r3 (%BW)': df_dyn.loc[min_fidx, 'vgrfz_r_bw'],
        'vgrfz_l3 (%BW)': df_dyn.loc[min_fidx, 'vgrfz_l_bw'],
        'vgrfz_t3 (%BW)': df_dyn.loc[min_fidx, 'vgrfz_t_bw'],
        'vgrfz_r3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfz_r_n'],
        'vgrfz_l3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfz_l_n'],
        'vgrfz_t3 (NEWTON)': df_dyn.loc[min_fidx, 'vgrfz_t_n'],
        }

    # Mostrar las primeras filas del DataFrame
    #print(df_mot[df_mot['time']==time])
    return pd.DataFrame([result_row])


data = [
        {'session_id': 'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38', 'participante': 'P1',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_3',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_3',
            'estocada_derecha_1',
            'estocada_izquierda_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_2',
            'sentadilla_90_1'
            ],
        'times': [
            # [4.65, 6.5],  # escalón_derecho 4.65	6.5
            # [3.4, 5.4],  # escalón_izquierdo 3.4	5.4
            # [5.85, 7.817, 10.133],  # estocada_deslizamiento_lateral_derecho 5.85	7.817	10.133
            # [4.6, 5.6, 6.9],  # estocada_deslizamiento_lateral_izquierdo 4.6	5.6	6.9
            # [5.7, 7.3, 8.85],  # estocada_deslizamiento_posterior_derecho 5.7	7.3	8.85
            # [5.85, 7.883, 9.8],  # estocada_deslizamiento_posterior_izquierdo 5.85	7.883	9.8
            # [5.05, 7.6, 9.1],  # estocada_derecha 5.05	7.6	9.1
            # [3.55, 6.467, 9.05],  # estocada_izquierda 3.55	6.467	9.05
            # [6.467, 9.417, 13.333],  # estocada_lateral_derecha 6.467	9.417	13.333
            # [5.933, 9.467, 12.317],  # estocada_lateral_izquierda 5.933	9.467	12.317
            # [6.95, 12.217, 16.1],  # sentadilla_60 6.95	12.217	16.1
            [5.85, 10.933, 14.283]   # sentadilla_90  5.85	10.92	14.283
        ]
        },
        {'session_id': 'a3724192-e2b6-4636-b176-3b3028d66230', 'participante': 'P2',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_1',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_1',
            'sentadilla_90_1'
            ],
        'times': [
            [1.23, 2.34],  # escalón_derecho
            [1.12, 2.22],  # escalón_izquierdo
            [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
            [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
            [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
            [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
            [1.05, 2.15, 3.25],  # estocada_izquierda
            [1.50, 2.60, 3.70],  # estocada_derecha
            [1.50, 2.60, 3.70],  # estocada_lateral_derecha
            [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
            [1.75, 2.85, 3.95],  # sentadilla_60
            [1.30, 2.40, 3.50]   # sentadilla_90
        ]},
        {'session_id': '0199dfe7-2595-471e-8f49-d5a887434357', 'participante': 'P3',
        'trial_names': [
            'escalón_derecho_2',
            'escalón_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_1',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_1',
            'sentadilla_90_1'
            ],
        'times': [
            [1.23, 2.34],  # escalón_derecho
            [1.12, 2.22],  # escalón_izquierdo
            [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
            [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
            [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
            [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
            [1.05, 2.15, 3.25],  # estocada_izquierda
            [1.50, 2.60, 3.70],  # estocada_lateral_derecha
            [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
            [1.75, 2.85, 3.95],  # sentadilla_60
            [1.30, 2.40, 3.50]   # sentadilla_90
        ]},
        {'session_id': 'dd83a45d-85b0-4f5e-bd26-93d22c413ed9', 'participante': 'P4',
        'trial_names': [
            'escalón_derecho_2',
            'escalón_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_1'
            ],
        'times': [
            [1.23, 2.34],  # escalón_derecho
            [1.12, 2.22],  # escalón_izquierdo
            [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
            [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
            [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
            [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
            [1.05, 2.15, 3.25],  # estocada_izquierda
            [1.50, 2.60, 3.70],  # estocada_lateral_derecha
            [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
            [1.75, 2.85, 3.95],  # sentadilla_60
            [1.30, 2.40, 3.50]   # sentadilla_90
        ]},
        {'session_id': '34b7f090-1cbd-43ce-86a2-a40c50feec3f', 'participante': 'P5',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_1',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_1',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4.85, 6],  # escalón_derecho
            [4.65, 5.35],  # escalón_izquierdo
            [4.95, 7.3, 8.65],  # estocada_deslizamiento_lateral_derecho
            [4.7, 7.55, 9.05],  # estocada_deslizamiento_lateral_izquierdo
            [4.05, 6.85, 9.35],  # estocada_deslizamiento_posterior_derecho
            [3.9, 5.95, 8.35],  # estocada_deslizamiento_posterior_izquierdo
            [4.4, 7.95, 11.65],  # estocada_izquierda
            [4.5, 9.05, 11.85],  # estocada_derecha
            [6.2, 9.85, 12.2],  # estocada_lateral_derecha
            [5.65, 10.45, 13.95],  # estocada_lateral_izquierda
            [4.9, 7.5, 11.35],  # sentadilla_60
            [8.95, 13.05, 16.15]   # sentadilla_90
        ]},
        {'session_id': 'bfd679e8-b5e3-4185-b2f2-011554045077', 'participante': 'P6',
        'trial_names': [
            'escalon_derecho_3',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_1',
            'estocada_derecha_1',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_1',
            'sentadilla_90_2'],
        'times': [
            [3.85, 4.55],  # escalón_derecho
            [3.55, 4.3],  # escalón_izquierdo
            [6.1, 8.3, 10.35],  # estocada_deslizamiento_lateral_derecho
            [5.6, 7.1, 8.75],  # estocada_deslizamiento_lateral_izquierdo
            [5.3, 7.55, 9.9],  # estocada_deslizamiento_posterior_derecho
            [4.8, 8.15, 10.75],  # estocada_deslizamiento_posterior_izquierdo
            [4.35, 8.8, 12],  # estocada_izquierda
            [10, 14.6, 17.35],  # estocada_derecha
            [5.25, 8.25, 11.05],  # estocada_lateral_derecha
            [5.9, 9.05, 11.9],  # estocada_lateral_izquierda
            [5.45, 10.1, 13.9],  # sentadilla_60
            [5.9, 10.65, 14.35]   # sentadilla_90
        ]},
        {'session_id': '37ad96c7-d786-4338-81ea-0d58104e9bb5', 'participante': 'P7',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_1',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_1',
            'sentadilla_90_2'],
        'times': [
            [4.45, 5.3],  # escalón_derecho
            [4.6, 5.55],  # escalón_izquierdo
            [3.75, 5.5, 7.15],  # estocada_deslizamiento_lateral_derecho
            [3.75, 5.25, 6.85],  # estocada_deslizamiento_lateral_izquierdo
            [3.75, 6.55, 8.3],  # estocada_deslizamiento_posterior_derecho
            [3.55, 5.4, 7.9],  # estocada_deslizamiento_posterior_izquierdo
            [3.8, 8.05, 10.6],  # estocada_izquierda
            [4.35, 7.85, 10.45],  # estocada_derecha
            [3.95, 8.3, 10.2],  # estocada_lateral_derecha
            [5.7, 9.2, 10.65],  # estocada_lateral_izquierda
            [4.9, 8.6, 12.55],  # sentadilla_60
            [4.45, 9.55, 12.75]   # sentadilla_90
        ]},
        {'session_id': '1a934006-0010-471a-9c38-fbd7edb6ffbc', 'participante': 'P8',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_2',
            'estocada_derecha_1',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_1',
            'sentadilla_90_1'],
        'times': [
            [4.3, 5.05],  # escalón_derecho
            [4.35, 5.05],  # escalón_izquierdo
            [4.6, 6.7, 8.15],  # estocada_deslizamiento_lateral_derecho
            [4.75, 6.6, 8.85],  # estocada_deslizamiento_lateral_izquierdo
            [4.7, 7.35, 10.55],  # estocada_deslizamiento_posterior_derecho
            [5.5, 7.25, 10],  # estocada_deslizamiento_posterior_izquierdo
            [4.65, 8.05, 10.1],  # estocada_izquierda
            [4.4, 7.15, 9.45],  # estocada_derecha
            [4.4, 6.65, 9.25],  # estocada_lateral_derecha
            [5.55, 7.25, 9.7],  # estocada_lateral_izquierda
            [4.55, 7.35, 11.1],  # sentadilla_60
            [4.3, 8.35, 12.1]   # sentadilla_90
        ]},
        {'session_id': '03142c4c-922a-4d9e-8cf3-a2eb229baa14', 'participante': 'P9',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4.35, 5.25],  # escalón_derecho
            [4.85, 5.95],  # escalón_izquierdo
            [4.6, 5.95, 7.35],  # estocada_deslizamiento_lateral_derecho
            [4.65, 5.95, 7],  # estocada_deslizamiento_lateral_izquierdo
            [4.2, 5.65, 7.55],  # estocada_deslizamiento_posterior_derecho
            [3.6, 5.3, 6.5],  # estocada_deslizamiento_posterior_izquierdo
            [4.15, 7.45, 8.75],  # estocada_izquierda
            [4.15, 7.7, 9.1],  # estocada_derecha
            [4, 6.6, 8.55],  # estocada_lateral_derecha
            [4.8, 7.85, 9.6],  # estocada_lateral_izquierda
            [4.8, 8.6, 10.8],  # sentadilla_60
            [5, 8.75, 11.75]   # sentadilla_90
        ]},
        {'session_id': 'cb608973-5e67-4d57-8b64-2e73fbbc6361', 'participante': 'P10',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_2',
            'estocada_derecha_1',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4.7, 5.45],  # escalón_derecho
            [4.6, 5.25],  # escalón_izquierdo
            [4.9, 6.45, 7.8],  # estocada_deslizamiento_lateral_derecho
            [4.25, 5.75, 7.15],  # estocada_deslizamiento_lateral_izquierdo
            [4.35, 5.7, 7.55],  # estocada_deslizamiento_posterior_derecho
            [4.85, 6.55, 8],  # estocada_deslizamiento_posterior_izquierdo
            [3.85, 7.75, 9.6],  # estocada_izquierda
            [4.15, 7.65, 9.65],  # estocada_derecha
            [5.15, 9.25, 10.75],  # estocada_lateral_derecha
            [5.05, 9.45, 11.15],  # estocada_lateral_izquierda
            [4.9, 8.9, 11.7],  # sentadilla_60
            [5.9, 9.35, 12.4]   # sentadilla_90
        ]},
        {'session_id': '39cb5ebb-153d-4aa0-ada7-a28c849eec67', 'participante': 'P11',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [5.15, 6.4],  # escalón_derecho
            [4, 4.8],  # escalón_izquierdo
            [5.65, 7.5, 8.9],  # estocada_deslizamiento_lateral_derecho
            [5.15, 6.75, 8.15],  # estocada_deslizamiento_lateral_izquierdo
            [4.65, 6, 7.3],  # estocada_deslizamiento_posterior_derecho
            [4.55, 6.4, 8.45],  # estocada_deslizamiento_posterior_izquierdo
            [4.25, 6.85, 9.35],  # estocada_izquierda
            [5, 7.45, 9.6],  # estocada_derecha
            [5.85, 9.2, 10.5],  # estocada_lateral_derecha
            [4.4, 7.25, 9],  # estocada_lateral_izquierda
            [4.05, 6.55, 9.05],  # sentadilla_60
            [5, 8.15, 10.75]   # sentadilla_90
        ]},
        {'session_id': '3eee7aa7-cece-4ead-a822-554b95b05613', 'participante': 'P12',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_1',           
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_1',
            'estocada_derecha_1',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [3.95, 5.15],  # escalón_derecho
            [6.15, 7.4],  # escalón_izquierdo
            [6.15, 7.65, 9.65],  # estocada_deslizamiento_lateral_derecho
            [5.9, 8.15, 9.85],  # estocada_deslizamiento_lateral_izquierdo
            [6.4, 8.6, 10.2],  # estocada_deslizamiento_posterior_derecho
            [6.65, 8.7, 10.2],  # estocada_deslizamiento_posterior_izquierdo
            [4.45, 9.15, 11.3],  # estocada_izquierda
            [3.95, 7.85, 10.25],  # estocada_derecha
            [4.35, 9.05, 11.65],  # estocada_lateral_derecha
            [6.35, 9.05, 12.25],  # estocada_lateral_izquierda
            [4.65, 9.75, 12.25],  # sentadilla_60
            [4.3, 8.45, 13.65]   # sentadilla_90
        ]},
        {'session_id': '0199dfe7-2595-471e-8f49-d5a887434357', 'participante': 'P13',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_1',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4.25, 4.9],  # escalón_derecho
            [3.75, 4.5],  # escalón_izquierdo
            [4.25, 6.05, 7.55],  # estocada_deslizamiento_lateral_derecho
            [6.1, 8.15, 10.15],  # estocada_deslizamiento_lateral_izquierdo
            [5.15, 9.05, 11.45],  # estocada_deslizamiento_posterior_derecho
            [4.85, 7.95, 10.25],  # estocada_deslizamiento_posterior_izquierdo
            [4.75, 6.85, 8.85],  # estocada_izquierda
            [5.15, 7.75, 11.25],  # estocada_derecha
            [5.35, 8.5, 11.1],  # estocada_lateral_derecha
            [6.1, 8.95, 12.15],  # estocada_lateral_izquierda
            [4.75, 7.15, 10.15],  # sentadilla_60
            [6.35, 9.65, 13.25]   # sentadilla_90
        ]},
        {'session_id': 'ad28c4e1-9f27-4554-9e19-1063888ab302', 'participante': 'P14',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_1',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [3.85, 4.85],  # escalón_derecho
            [4, 4.8],  # escalón_izquierdo
            [4.9, 8.35, 9.9],  # estocada_deslizamiento_lateral_derecho
            [4.5, 5.75, 7.45],  # estocada_deslizamiento_lateral_izquierdo
            [4.05, 6, 7.65],  # estocada_deslizamiento_posterior_derecho
            [4.15, 6, 7.75],  # estocada_deslizamiento_posterior_izquierdo
            [3.45, 5.65, 8.1],  # estocada_izquierda
            [3.95, 7.05, 9.6],  # estocada_derecha
            [4.75, 7.55, 10.1],  # estocada_lateral_derecha
            [3.85, 6.85, 9.55],  # estocada_lateral_izquierda
            [4.45, 7.55, 10],  # sentadilla_60
            [4.65, 7.55, 10.95]   # sentadilla_90
        ]},
        {'session_id': '8cff7224-37bf-44d5-94d0-f7dfdea5bc36', 'participante': 'P15',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_derecha_1_1',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_1',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [3.45, 3.95],  # escalón_derecho
            [3.75, 4.2],  # escalón_izquierdo
            [6.15, 9.95, 11.35],  # estocada_deslizamiento_lateral_derecho
            [5.45, 6.9, 7.95],  # estocada_deslizamiento_lateral_izquierdo
            [6, 6.9, 8.35],  # estocada_deslizamiento_posterior_derecho
            [3.5, 5.65, 6.75],  # estocada_deslizamiento_posterior_izquierdo
            [4.15, 6.6, 9],  # estocada_izquierda
            [5, 7.4, 9.45],  # estocada_derecha
            [6.5, 10.1, 12.85],  # estocada_lateral_derecha
            [4.9, 7.25, 9.85],  # estocada_lateral_izquierda
            [4.35, 6.25, 9.15],  # sentadilla_60
            [4.75, 8.05, 9.85]   # sentadilla_90
        ]},
        {'session_id': 'fe7e294e-b199-4c16-b9ca-c8c7841c42ba', 'participante': 'P16',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [5.3, 5.8],  # escalón_derecho
            [4.25, 4.75],  # escalón_izquierdo
            [4.4, 6.35, 7.8],  # estocada_deslizamiento_lateral_derecho
            [6.3, 7.7, 8.95],  # estocada_deslizamiento_lateral_izquierdo
            [5.25, 7.1, 8.65],  # estocada_deslizamiento_posterior_derecho
            [4.5, 5.9, 7.15],  # estocada_deslizamiento_posterior_izquierdo
            [8.6, 11.8, 13],  # estocada_izquierda
            [4.6, 7, 10.25],  # estocada_derecha
            [4.7, 7.6, 9.55],  # estocada_lateral_derecha
            [4.7, 7.5, 9.5],  # estocada_lateral_izquierda
            [5.9, 8, 11.35],  # sentadilla_60
            [4.95, 8.15, 10.25]   # sentadilla_90
        ]},
        {'session_id': '634a9945-6598-4478-aa6f-3e88bdb2ab07', 'participante': 'P17',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_1',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_1',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [3.75, 4.55],  # escalón_derecho
            [3.8, 4.55],  # escalón_izquierdo
            [4.4, 6.45, 7.6],  # estocada_deslizamiento_lateral_derecho
            [4.4, 6.9, 8.2],  # estocada_deslizamiento_lateral_izquierdo
            [4.5, 5.85, 7.05],  # estocada_deslizamiento_posterior_derecho
            [3.35, 5.4, 7.3],  # estocada_deslizamiento_posterior_izquierdo
            [3.35, 6.65, 8.65],  # estocada_izquierda
            [3.85, 7.55, 9.75],  # estocada_derecha
            [4.9, 7.75, 10.25],  # estocada_lateral_derecha
            [4.1, 6.95, 8.95],  # estocada_lateral_izquierda
            [3.95, 6.95, 9.6],  # sentadilla_60
            [4.45, 7.75, 9.8]   # sentadilla_90
        ]},
        {'session_id': '126c9b36-d8f4-4d25-83e3-cd3afbc04148', 'participante': 'P18',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4.25, 4.85],  # escalón_derecho
            [4, 4.55],  # escalón_izquierdo
            [4.8, 6.25, 7.7],  # estocada_deslizamiento_lateral_derecho
            [4.65, 5.75, 7.15],  # estocada_deslizamiento_lateral_izquierdo
            [3.4, 5.2, 6.45],  # estocada_deslizamiento_posterior_derecho
            [3.35, 5.3, 6.2],  # estocada_deslizamiento_posterior_izquierdo
            [4.65, 7.15, 9.15],  # estocada_izquierda
            [3.85, 6.4, 8.15],  # estocada_derecha
            [6.35, 9.35, 11.85],  # estocada_lateral_derecha
            [4.35, 7.65, 9.35],  # estocada_lateral_izquierda
            [4.4, 7.15, 11.15],  # sentadilla_60
            [4.25, 8.65, 11.95]   # sentadilla_90
        ]},
        {'session_id': '67a1f5a5-1723-4c3c-86a5-54a06a0eb878', 'participante': 'P19',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [5, 6],  # escalón_derecho
            [4.5, 5.15],  # escalón_izquierdo
            [5.65, 6.95, 9.35],  # estocada_deslizamiento_lateral_derecho
            [7.65, 8.9, 10.35],  # estocada_deslizamiento_lateral_izquierdo
            [5.5, 8.35, 9.9],  # estocada_deslizamiento_posterior_derecho
            [4, 6.3, 7.6],  # estocada_deslizamiento_posterior_izquierdo
            [3.75, 6.95, 9.9],  # estocada_izquierda
            [5, 7.45, 9.5],  # estocada_derecha
            [5.75, 9.15, 12.55],  # estocada_lateral_derecha
            [5.25, 8, 9.7],  # estocada_lateral_izquierda
            [4.65, 7.95, 10.2],  # sentadilla_60
            [5.85, 9.75, 12.95]   # sentadilla_90
        ]},
        {'session_id': 'a3ba3d30-d5fd-4727-b460-b4310339a852', 'participante': 'P20',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4.45, 4.95],  # escalon_derecho
            [4.4, 4.9],  # escalon_izquierdo
            [5.15, 6.5, 7.8],  # estocada_deslizamiento_lateral_derecho
            [4.9, 6.3, 7.75],  # estocada_deslizamiento_lateral_izquierdo
            [4.5, 6.8, 8.55],  # estocada_deslizamiento_posterior_derecho
            [4, 5.15, 6.8],  # estocada_deslizamiento_posterior_izquierdo
            [4.75, 7.25, 9],  # estocada_izquierda
            [4.6, 6.95, 8.75],  # estocada_derecha
            [5, 7.15, 9],  # estocada_lateral_derecha
            [4.8, 6.75, 8.8],  # estocada_lateral_izquierda
            [5.25, 8.2, 10.65],  # sentadilla_60
            [4.45, 7.95, 10.25]   # sentadilla_90
        ]},
        {'session_id': 'f8ab78be-81ee-451f-affa-80a56880f741', 'participante': 'P21',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [3.4, 4],  # escalon_derecho
            [4.65, 5.3],  # escalon_izquierdo
            [4.35, 5.8, 6.95],  # estocada_deslizamiento_lateral_derecho
            [4, 5.15, 6.15],  # estocada_deslizamiento_lateral_izquierdo
            [4, 6.15, 7.25],  # estocada_deslizamiento_posterior_derecho
            [3.85, 5.2, 6],  # estocada_deslizamiento_posterior_izquierdo
            [4.65, 6.55, 9.25],  # estocada_izquierda
            [4.1, 6.45, 8.25],  # estocada_derecha
            [4.65, 7.35, 8.8],  # estocada_lateral_derecha
            [5.2, 7.65, 9.55],  # estocada_lateral_izquierda
            [4.1, 7.8, 10.8],  # sentadilla_60
            [6, 8.5, 11.85]   # sentadilla_90
        ]},
        {'session_id': 'c71d092b-93ba-4dfb-afc3-57fb46ef0736', 'participante': 'P22',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_2',
            'estocada_derecha_1',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [5.15, 5.75],  # escalon_derecho
            [6.05, 6.45],  # escalon_izquierdo
            [4.35, 5.65, 7.1],  # estocada_deslizamiento_lateral_derecho
            [4.35, 5.6, 7.25],  # estocada_deslizamiento_lateral_izquierdo
            [4.25, 5.4, 6.5],  # estocada_deslizamiento_posterior_derecho
            [4.2, 5.45, 6.95],  # estocada_deslizamiento_posterior_izquierdo
            [4.65, 7.25, 9.6],  # estocada_izquierda
            [3.95, 7.2, 9.3],  # estocada_derecha
            [4.85, 7.05, 10.15],  # estocada_lateral_derecha
            [4.8, 7.65, 10.25],  # estocada_lateral_izquierda
            [4.9, 9, 11.95],  # sentadilla_60
            [8.5, 11.05, 13.95]   # sentadilla_90
        ]},
        {'session_id': '85faebd8-e10e-41a4-a0cd-4abea37e1948', 'participante': 'P23',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_2',
           # 'estocada_izquierda',
            'estocada_derecha_1',
            'estocada_lateral_derecha_1',
           # 'estocada_lateral_izquierda',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [4, 5.1],  # escalon_derecho
            [4.6, 5.25],  # escalon_izquierdo
            [8.6, 9.45, 10.65],  # estocada_deslizamiento_lateral_derecho
            [6.8, 8.6, 10.15],  # estocada_deslizamiento_lateral_izquierdo
            [6.95, 9.6, 10.9],  # estocada_deslizamiento_posterior_derecho
            [5.15, 6.5, 8.25],  # estocada_deslizamiento_posterior_izquierdo
           # [1.05, 2.15, 3.25],  # estocada_izquierda
            [5.55, 8.5, 10.45],  # estocada_derecha
            [8.85, 11.25, 14.75],  # estocada_lateral_derecha
           # [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
            [7.15, 9.25, 12.9],  # sentadilla_60
            [7.15, 10.35, 12.95]   # sentadilla_90
        ]},
        {'session_id': '136cdc8a-cdf8-4745-8d12-a4499bfffe30', 'participante': 'P24',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_1',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_1',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_2',
            'sentadilla_90_2'],
        'times': [
            [3.8, 4.25],  # escalon_derecho
            [3.55, 4.1],  # escalon_izquierdo
            [4.5, 5.15, 6.15],  # estocada_deslizamiento_lateral_derecho
            [4.15, 4.9, 5.75],  # estocada_deslizamiento_lateral_izquierdo
            [3.65, 4.55, 5.85],  # estocada_deslizamiento_posterior_derecho
            [2.8, 3.9, 5.25],  # estocada_deslizamiento_posterior_izquierdo
            [4, 7.15, 8.8],  # estocada_izquierda
            [4.6, 7.6, 9.25],  # estocada_derecha
            [3.5, 5.5, 7.65],  # estocada_lateral_derecha
            [3.35, 5.45, 7.8],  # estocada_lateral_izquierda
            [4.3, 7, 9.65],  # sentadilla_60
            [4.35, 7.55, 8.95]   # sentadilla_90
        ]},
        {'session_id': '8d1c6dc2-552c-401f-af3d-c11b6dcebe3d', 'participante': 'P25',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_2',
            'estocada_derecha_2',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_1',
            'sentadilla_90_2'],
        'times': [
            [4, 6.5],  # escalon_derecho
            [4, 5.95],  # escalon_izquierdo
            [6.5, 8.4, 10.15],  # estocada_deslizamiento_lateral_derecho
            [6.5, 8.05, 9.65],  # estocada_deslizamiento_lateral_izquierdo
            [6.25, 8.05, 9.95],  # estocada_deslizamiento_posterior_derecho
            [5.65, 8.25, 10.65],  # estocada_deslizamiento_posterior_izquierdo
            [5.45, 9.15, 11.55],  # estocada_izquierda
            [6.3, 9.85, 11.6],  # estocada_derecha
            [6.75, 10.2, 13.1],  # estocada_lateral_derecha
            [6.8, 10.45, 13.8],  # estocada_lateral_izquierda
            [6.75, 9.7, 12.4],  # sentadilla_60
            [4.85, 8.7, 13.25]   # sentadilla_90
        ]},
        {'session_id': 'eb09846b-b49e-46e9-be68-73c5e0eff109', 'participante': 'P26',
        'trial_names': [
            'escalon_derecho_1',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_2',
            'estocada_deslizamiento_posterior_izquierdo_2',
            'estocada_izquierda_1',
            'estocada_derecha_2',
            'estocada_lateral_derecha_2',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_1',
            'sentadilla_90_1'],
        'times': [
            [3.45, 5.45],  # escalon_derecho
            [4.15, 5.9],  # escalon_izquierdo
            [6.15, 10.05, 12.25],  # estocada_deslizamiento_lateral_derecho
            [4.8, 6.95, 8.55],  # estocada_deslizamiento_lateral_izquierdo
            [5.7, 7.65, 8.95],  # estocada_deslizamiento_posterior_derecho
            [4.6, 7.65, 9.2],  # estocada_deslizamiento_posterior_izquierdo
            [4.7, 7.85, 9.8],  # estocada_izquierda
            [3.95, 7.15, 8.5],  # estocada_derecha
            [5.25, 8, 13.2],  # estocada_lateral_derecha
            [5, 9.35, 11.95],  # estocada_lateral_izquierda
            [6.45, 9.95, 12.95],  # sentadilla_60
            [5.05, 8.35, 11.35]   # sentadilla_90
        ]},
        {'session_id': 'd8c338ad-3686-41d7-b70c-186aeacdc7cf', 'participante': 'P27',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_2',
            'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_1',
            'estocada_derecha_1',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_1',
            'sentadilla_90_1'],
        'times': [
            [4.45, 6.4],  # escalon_derecho
            [4.65, 6.75],  # escalon_izquierdo
            [5.55, 8.25, 9.5],  # estocada_deslizamiento_lateral_derecho
            [6.15, 7.85, 9.9],  # estocada_deslizamiento_lateral_izquierdo
            [4.95, 7.5, 9.35],  # estocada_deslizamiento_posterior_derecho
            [5.2, 8.35, 10.85],  # estocada_deslizamiento_posterior_izquierdo
            [6.15, 10.25, 13.95],  # estocada_izquierda
            [8, 11, 15],  # estocada_derecha
            [5.7, 12.75, 16],  # estocada_lateral_derecha
            [6, 9.65, 13.7],  # estocada_lateral_izquierda
            [9.6, 12.3, 16.95],  # sentadilla_60
            [6.15, 10.05, 15.1]   # sentadilla_90
        ]},
        {'session_id': '5ebde49f-8966-4258-bc17-5228a3e41b2d', 'participante': 'P28',
        'trial_names': [
            'escalon_derecho_2',
            'escalon_izquierdo_2',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_1',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_1',
            'estocada_derecha_1',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_2',
            'sentadilla_60_1',
            'sentadilla_90_1'],
        'times': [
            [3.4, 4.6],  # escalon_derecho
            [4.2, 4.95],  # escalon_izquierdo
            [6.5, 9.15, 11],  # estocada_deslizamiento_lateral_derecho
            [5.75, 6.95, 8.6],  # estocada_deslizamiento_lateral_izquierdo
            [4.25, 5.45, 6.85],  # estocada_deslizamiento_posterior_derecho
            [5.35, 7, 7.85],  # estocada_deslizamiento_posterior_izquierdo
            [5.2, 8.8, 13.25],  # estocada_izquierda
            [6.4, 9.55, 14.7],  # estocada_derecha
            [7.2, 11.45, 14.95],  # estocada_lateral_derecha
            [5.15, 7.55, 11.95],  # estocada_lateral_izquierda
            [7.45, 10.25, 15.2],  # sentadilla_60
            [6.15, 10, 15.15]   # sentadilla_90
        ]},

        
        {'session_id': '4862ff70-cf69-4ec8-9402-8a8d88b881ef', 'participante': 'P29',
        'trial_names': [
            'escalon_derecho_2', #2   3.55	 4.85
            'escalon_izquierdo_2', #2	3.95	5
            'estocada_deslizamiento_lateral_derecho_1',# 1	6.4	7.8	8.95
            'estocada_deslizamiento_lateral_izquierdo_2', #2	4.3	5.95	6.95
            'estocada_deslizamiento_posterior_derecho_2',# 2	5.05	7.25	8.25
            'estocada_deslizamiento_posterior_izquierdo_1',# 1	4.85	6.25	7.55
            'estocada_derecha_2',# 2	  3.35	  6.1	 9.6
            'estocada_izquierda_1', #1	4.45	6.65	8.95
            'estocada_lateral_derecha_1',# 1	5.05	8.25	11.65
            'estocada_lateral_izquierda_2', #2	5.05	10.1	14.15
            'sentadilla_60_2',# 2	4.95	7.6	    10.35
            'sentadilla_90_2'],# 2	3.65	7	10.35
        'times': [
            [3.55, 4.85],  # escalón_derecho
            [3.95, 5],  # escalón_izquierdo
            [6.4, 7.8, 8.95],  # estocada_deslizamiento_lateral_derecho
            [4.3, 5.95, 6.95],  # estocada_deslizamiento_lateral_izquierdo
            [5.05, 7.25, 8.25],  # estocada_deslizamiento_posterior_derecho
            [4.85, 6.25, 7.55],  # estocada_deslizamiento_posterior_izquierdo
            [3.35, 6.1,  9.6],  # estocada_derecha
            [4.45,	6.65,	8.95],  # estocada_izquierda
            [5.05,	8.25,	11.65],  # estocada_lateral_derecha
            [5.05,	10.1,	14.1],  # estocada_lateral_izquierda
            [4.95,	7.6,	    10.35],  # sentadilla_60
            [3.65,	7,	10.35]   # sentadilla_90
        ]},
        {'session_id': '86713dcf-995b-40e1-8df2-5da494cb170b', 'participante': 'P30',
        'trial_names': [
            'escalon_derecho_1',# 1	3.85	5.05
            'escalon_izquierdo_2',# 2	3.85	5.05
            'estocada_deslizamiento_lateral_derecho_1', #1	5.85	7.25	8.55
            'estocada_deslizamiento_lateral_izquierdo_2',# 2 	4.55	6.35	7.4
            'estocada_deslizamiento_posterior_derecho_1',# 1   4.3	 6.2	  7.75
            'estocada_deslizamiento_posterior_izquierdo_1',# 1	4.85	7.1	  8.2
            'estocada_derecha_1',# 1	  4.65	 8.7 	12.45
            'estocada_izquierda_1',# 1	3.95	6.4 	8.45
            'estocada_lateral_derecha_2',# 2	 5.6	11.2	14.45
            'estocada_lateral_izquierda_1',# 1	5.65	8.15	10.45
            'sentadilla_60_1',# 1	4.9	 8.25	10.85
            'sentadilla_90_1'],# 1	4.55	9.2	 12.35
        'times': [
            [3.85,	5.05],  # escalón_derecho
            [3.85,	5.05],  # escalón_izquierdo
            [5.85,	7.25,	8.55],  # estocada_deslizamiento_lateral_derecho
            [4.55,	6.35,	7.4],  # estocada_deslizamiento_lateral_izquierdo
            [4.3,	 6.2,	  7.75],  # estocada_deslizamiento_posterior_derecho
            [4.85,	7.1,	  8.2],  # estocada_deslizamiento_posterior_izquierdo
            [4.65,	 8.7, 	12.45],  # estocada_derecha
            [3.95,	6.4, 	8.45],  # estocada_izquierda
            [5.6,	11.2,	14.45],  # estocada_lateral_derecha
            [5.65,	8.15,	10.45],  # estocada_lateral_izquierda
            [4.9,	 8.25,	10.85],  # sentadilla_60
        #     [4.55,	9.2,	 12.35]   # sentadilla_90

        ]},
        
]

#=====================para extraer los datos de los participantes=========================
movement_dfs = {extract_movement_name(trial): [] for trial in data[0]['trial_names']}
# workbook = load_workbook('S60_GFR.xlsx') # En esta linea se abre un excel anteriormente existente
# hoja_activa = workbook.active
# Procesar los datos
data_all = []
for participant in data[:]:
    session_id = participant['session_id']
    print(f'participant: {participant["participante"]}')
    for trial_name in participant['trial_names']: 
        if trial_name in [
                        # 'estocada_derecha_1','estocada_derecha_2',
                        # 'estocada_izquierda_1','estocada_izquierda_2',
                        # 'estocada_lateral_derecha_1','estocada_lateral_derecha_2',
                        # 'estocada_lateral_izquierda_1','estocada_lateral_izquierda_2',
                        # 'estocada_deslizamiento_posterior_derecho_1','estocada_deslizamiento_posterior_derecho_2',
                        # 'estocada_deslizamiento_posterior_izquierdo_1','estocada_deslizamiento_posterior_izquierdo_2',
                        # 'estocada_deslizamiento_lateral_derecho_1','estocada_deslizamiento_lateral_derecho_2',
                        # 'estocada_deslizamiento_lateral_izquierdo_1','estocada_deslizamiento_lateral_izquierdo_2',
                        'escalon_derecho_1','escalon_derecho_2',
                        'escalon_izquierdo_1','escalon_izquierdo_2',
                        ]:
            # print(trial_name)
            base_trial_name = extract_movement_name(trial_name)  # Obtener el nombre base sin el número
            
            try:
                if any(word in trial_name for word in ['izquierda', 'izquierdo']):
                    datf = extract_data_t(session_id=participant['participante'], trial_name=trial_name,leg='l')
                    
                else:
                    datf = extract_data_t(session_id=participant['participante'], trial_name=trial_name,leg='r')
                    
                data_all.append(signal_part)
                row = datf.iloc[0]

                data_for_frame = {
                    'participante': participant['participante'],

                    'F1TIME': row['time1'],
                    'F1-RODSAGITAL D': row['knee_angle_r1'],
                    'F1-RODSAGITAL I': row['knee_angle_l1'],
                    'F1-VGRFY D (%BW)': row['vgrfy_r1 (%BW)'],
                    'F1-VGRFY I (%BW)': row['vgrfy_l1 (%BW)'],
                    'F1-VGRFY T (%BW)': row['vgrfy_t1 (%BW)'],
                    'F1-VGRFY D (NEWTON)': row['vgrfy_r1 (NEWTON)'],
                    'F1-VGRFY I (NEWTON)': row['vgrfy_l1 (NEWTON)'],
                    'F1-VGRFY T (NEWTON)': row['vgrfy_t1 (NEWTON)'],

                    'F1-VGRFX D (%BW)': row['vgrfx_r1 (%BW)'],
                    'F1-VGRFX I (%BW)': row['vgrfx_l1 (%BW)'],
                    'F1-VGRFX T (%BW)': row['vgrfx_t1 (%BW)'],
                    'F1-VGRFX D (NEWTON)': row['vgrfx_r1 (NEWTON)'],
                    'F1-VGRFX I (NEWTON)': row['vgrfx_l1 (NEWTON)'],
                    'F1-VGRFX T (NEWTON)': row['vgrfx_t1 (NEWTON)'],

                    'F1-VGRFZ D (%BW)': row['vgrfz_r1 (%BW)'],
                    'F1-VGRFZ I (%BW)': row['vgrfz_l1 (%BW)'],
                    'F1-VGRFZ T (%BW)': row['vgrfz_t1 (%BW)'],
                    'F1-VGRFZ D (NEWTON)': row['vgrfz_r1 (NEWTON)'],
                    'F1-VGRFZ I (NEWTON)': row['vgrfz_l1 (NEWTON)'],
                    'F1-VGRFZ T (NEWTON)': row['vgrfz_t1 (NEWTON)'],

                    'F2TIME': row['time2'],
                    'F2-RODSAGITAL D': row['knee_angle_r2'],
                    'F2-RODSAGITAL I': row['knee_angle_l2'],
                    'F2-VGRFY D (%BW)': row['vgrfy_r2 (%BW)'],
                    'F2-VGRFY I (%BW)': row['vgrfy_l2 (%BW)'],
                    'F2-VGRFY T (%BW)': row['vgrfy_t2 (%BW)'],
                    'F2-VGRFY D (NEWTON)': row['vgrfy_r2 (NEWTON)'],
                    'F2-VGRFY I (NEWTON)': row['vgrfy_l2 (NEWTON)'],
                    'F2-VGRFY T (NEWTON)': row['vgrfy_t2 (NEWTON)'],

                    'F2-VGRFX D (%BW)': row['vgrfx_r2 (%BW)'],
                    'F2-VGRFX I (%BW)': row['vgrfx_l2 (%BW)'],
                    'F2-VGRFX T (%BW)': row['vgrfx_t2 (%BW)'],
                    'F2-VGRFX D (NEWTON)': row['vgrfx_r2 (NEWTON)'],
                    'F2-VGRFX I (NEWTON)': row['vgrfx_l2 (NEWTON)'],
                    'F2-VGRFX T (NEWTON)': row['vgrfx_t2 (NEWTON)'],

                    'F2-VGRFZ D (%BW)': row['vgrfz_r2 (%BW)'],
                    'F2-VGRFZ I (%BW)': row['vgrfz_l2 (%BW)'],
                    'F2-VGRFZ T (%BW)': row['vgrfz_t2 (%BW)'],
                    'F2-VGRFZ D (NEWTON)': row['vgrfz_r2 (NEWTON)'],
                    'F2-VGRFZ I (NEWTON)': row['vgrfz_l2 (NEWTON)'],
                    'F2-VGRFZ T (NEWTON)': row['vgrfz_t2 (NEWTON)'],

                    'F3TIME': row['time3'],
                    'F3-RODSAGITAL D': row['knee_angle_r3'],
                    'F3-RODSAGITAL I': row['knee_angle_l3'],
                    'F3-VGRFY D (%BW)': row['vgrfy_r3 (%BW)'],
                    'F3-VGRFY I (%BW)': row['vgrfy_l3 (%BW)'],
                    'F3-VGRFY T (%BW)': row['vgrfy_t3 (%BW)'],
                    'F3-VGRFY D (NEWTON)': row['vgrfy_r3 (NEWTON)'],
                    'F3-VGRFY I (NEWTON)': row['vgrfy_l3 (NEWTON)'],
                    'F3-VGRFY T (NEWTON)': row['vgrfy_t3 (NEWTON)'],

                    'F3-VGRFX D (%BW)': row['vgrfx_r3 (%BW)'],
                    'F3-VGRFX I (%BW)': row['vgrfx_l3 (%BW)'],
                    'F3-VGRFX T (%BW)': row['vgrfx_t3 (%BW)'],
                    'F3-VGRFX D (NEWTON)': row['vgrfx_r3 (NEWTON)'],
                    'F3-VGRFX I (NEWTON)': row['vgrfx_l3 (NEWTON)'],
                    'F3-VGRFX T (NEWTON)': row['vgrfx_t3 (NEWTON)'],

                    'F3-VGRFZ D (%BW)': row['vgrfz_r3 (%BW)'],
                    'F3-VGRFZ I (%BW)': row['vgrfz_l3 (%BW)'],
                    'F3-VGRFZ T (%BW)': row['vgrfz_t3 (%BW)'],
                    'F3-VGRFZ D (NEWTON)': row['vgrfz_r3 (NEWTON)'],
                    'F3-VGRFZ I (NEWTON)': row['vgrfz_l3 (NEWTON)'],
                    'F3-VGRFZ T (NEWTON)': row['vgrfz_t3 (NEWTON)'],
                }
                movement_dfs[base_trial_name].append(data_for_frame)
            except Exception as e:
                print(f"Error processing {trial_name} for participant {participant['participante']}: {e}")

abreviaciones = {
    "escalon_derecho": "ESCALON D",
    "escalon_izquierdo": "ESCALON I",
    "estocada_deslizamiento_lateral_derecho": "EST.DESl.LATERAL D",
    "estocada_deslizamiento_lateral_izquierdo": "EST.DESl.LATERAL I",
    "estocada_deslizamiento_posterior_derecho": "EST.DESL.POSTERIOR D",
    "estocada_deslizamiento_posterior_izquierdo": "EST.DESL.POSTERIOR I",
    "estocada_derecha": "EST.POSTERIOR D",
    "estocada_izquierda": "EST.POSTERIOR I",
    "estocada_lateral_derecha": "EST.LATERAL D",
    "estocada_lateral_izquierda": "EST.LATERAL I",
    "sentadilla_60": "SENT 60",
    "sentadilla_90": "SENT 90"
}
# Guardar en un archivo Excel con múltiples hojas

with pd.ExcelWriter(f"{' '.join(base_trial_name.split('_')[:-1])}.xlsx") as writer:
    # print(movement_dfs.items())
    
    for trial_name, data_list in movement_dfs.items():
        df = pd.DataFrame(data_list)
        sheet_name = abreviaciones.get(trial_name, trial_name)

        # Asegurar que el nombre de la hoja tenga máximo 31 caracteres
        sheet_name = sheet_name[:31]
        df.to_excel(writer, sheet_name=sheet_name, float_format="%.3f",index=False)

signals_interp = []

for part_dict in data_all:
    t_orig = part_dict['time'].to_numpy()
    t_norm = np.linspace(0, 100, 101)

    for eje in ['vgrfy', 'vgrfx', 'vgrfz']:
        for pierna, etiqueta in zip(['r', 'l'], ['Derecha', 'Izquierda']):
            col_name = f"{eje}_{pierna}(N)"
            y = part_dict[col_name].to_numpy()

            f_interp = interp1d(t_orig, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_interp = f_interp(np.linspace(t_orig[0], t_orig[-1], 101))
            y_interp = savgol_filter(y_interp, window_length=15, polyorder=3)

            df_temp = pd.DataFrame({
                'tiempo_norm': t_norm,
                'valor': y_interp,
                'eje': eje[-1].upper(),  # 'Y', 'X', 'Z'
                'pierna': etiqueta
            })
            signals_interp.append(df_temp)

# Unir todo en un solo DataFrame
df_signals = pd.concat(signals_interp, ignore_index=True)
save_folder = os.path.join("all_forces")
if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)
        
save_path = os.path.join(save_folder, f"{base_trial_name}_forces.png")
# print(save_path)
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

ejes = ['Y', 'X', 'Z']
titulos = ['Vertical', 'Anteroposterior', 'Medio-lateral']
piernas = ['Derecha', 'Izquierda']

for i, eje in enumerate(ejes):
    for j, pierna in enumerate(piernas):
        ax = axes[i, j]
        df_sub = df_signals[(df_signals['eje'] == eje) & (df_signals['pierna'] == pierna)]

        grouped = df_sub.groupby('tiempo_norm')['valor'].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(grouped['tiempo_norm'], grouped['mean'], label='Media', color='red' if pierna == 'Derecha' else 'blue')
        ax.fill_between(grouped['tiempo_norm'], grouped['min'], grouped['max'],
                        color='red' if pierna == 'Derecha' else 'blue', alpha=0.2)

        ax.set_title(f'{titulos[i]} - {pierna}', fontsize=10)
        if i == 2:
            ax.set_xlabel('Tiempo normalizado [%]')
        if j == 0:
            ax.set_ylabel('Fuerza (N)')
        ax.grid(True)



plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

# extract_data_t(session_id='e2eae0d5-db45-4c68-9ccc-1bad0c84a08d',trial_name='SENTADILLA_60_7')
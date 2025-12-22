"""
Script para generar gráficas agregadas de validación:
1. Curva promedio con banda de variabilidad para sentadilla_60 y sentadilla_90
2. Diagramas de dispersión con línea de identidad para correlación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import pearsonr, gaussian_kde
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AggregatePlotGenerator:
    def __init__(self, validation_results_folder="validation_results"):
        self.validation_results_folder = Path(validation_results_folder)
        self.participant_id = "P31T"
        
        # Rutas base
        self.gold_forces_folder = Path(r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\PLATAFORMAS")
        self.gold_markers_folder = Path(r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\MOTIVE")
        self.opencap_forces_folder = Path(f"Data\\{self.participant_id}\\MeasuredForces")
        self.opencap_markers_folder = Path(f"Data\\{self.participant_id}\\MarkerData")
    
    def _read_mot_file(self, file_path):
        """Lee un archivo .mot y devuelve un DataFrame."""
        with open(file_path) as f:
            for i, line in enumerate(f):
                if line.strip().lower() == 'endheader':
                    skip = i + 1
                    break
        
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            skiprows=skip,
            engine='python'
        )
        return df

    def _read_trc_file(self, file_path):
        """Lee un archivo .trc y devuelve un DataFrame y diccionario de marcadores."""
        with open(file_path) as f:
            lines = f.readlines()
        
        # Extraer nombres de marcadores del header
        header_line = lines[3].strip().split('\t')
        marker_names = header_line[2::3]
        
        # Leer datos
        df = pd.read_csv(file_path, sep='\t', skiprows=5, header=None)
        
        # Asignar nombres de columnas
        column_names = ['Frame', 'time']
        for marker_name in marker_names:
            column_names.extend([f'{marker_name}_x', f'{marker_name}_y', f'{marker_name}_z'])
        df.columns = column_names[:len(df.columns)]
        
        # Crear diccionario de marcadores
        markers_dict = {
            name: {
                'x': df[f'{name}_x'].values,
                'y': df[f'{name}_y'].values,
                'z': df[f'{name}_z'].values
            }
            for name in marker_names if f'{name}_x' in df.columns
        }
        
        return df, markers_dict, list(markers_dict.keys())

    def find_step_on_and_squat_boundaries(self, heel_y, time):
        """Detecta el inicio del movimiento (despegue del talón)."""
        heel_y_norm = (heel_y - np.min(heel_y)) / (np.max(heel_y) - np.min(heel_y))
        
        peaks, _ = find_peaks(heel_y_norm, height=0.5, prominence=0.3, distance=100)
        
        if len(peaks) == 0:
            return None
        
        step_on_peak_idx = peaks[0]
        search_window_before = 120
        search_start = max(0, step_on_peak_idx - search_window_before)
        takeoff_region = heel_y_norm[search_start:step_on_peak_idx]
        
        min_before_peak_idx = search_start + np.argmin(takeoff_region) if len(takeoff_region) > 0 else search_start
        return min_before_peak_idx

    def process_trial(self, trial_name):
        """Procesa un trial y retorna las señales normalizadas."""
        print(f"Procesando {trial_name}...")
        
        # Rutas de archivos
        gold_mot = self.gold_forces_folder / f"{trial_name}.mot"
        gold_trc = self.gold_markers_folder / f"{trial_name}.trc"
        opencap_mot = self.opencap_forces_folder / trial_name / f"{trial_name}_syncd_forces.mot"
        opencap_trc = self.opencap_markers_folder / f"{trial_name}.trc"
        
        if not all([gold_mot.exists(), gold_trc.exists(), opencap_mot.exists(), opencap_trc.exists()]):
            return None
        
        try:
            # Cargar datos
            gold_markers_df, gold_markers_dict, gold_marker_names = self._read_trc_file(gold_trc)
            opencap_markers_df, opencap_markers_dict, opencap_marker_names = self._read_trc_file(opencap_trc)
            
            # Buscar marcadores de talón
            opencap_heels = {name for name in opencap_marker_names if 'heel' in name.lower()}
            opencap_rheel = next((name for name in opencap_heels if 'r' in name.lower()), None)
            
            gold_heels = {name for name in gold_marker_names 
                         if any(x in name.lower() for x in ['unlabeled 1025', 'unlabeled1025', 'heel'])}
            gold_rheel = next((name for name in gold_heels if '1025' in name or 'r' in name.lower()), None)
            
            if not (opencap_rheel and gold_rheel):
                return None
            
            # Detectar límites
            opencap_heel_y = np.array(opencap_markers_dict[opencap_rheel]['y'])
            gold_heel_y = np.array(gold_markers_dict[gold_rheel]['y'])
            
            opencap_start_idx = self.find_step_on_and_squat_boundaries(
                opencap_heel_y, opencap_markers_df['time'].values
            )
            gold_start_idx = self.find_step_on_and_squat_boundaries(
                gold_heel_y, gold_markers_df['time'].values
            )
            
            if opencap_start_idx is None or gold_start_idx is None:
                return None
            
            # Cargar fuerzas
            gold_df = self._read_mot_file(gold_mot)
            opencap_df = self._read_mot_file(opencap_mot)
            
            # Renombrar columnas gold
            rename_map = {}
            for col in gold_df.columns:
                if 'r_gr_force' in col.lower():
                    rename_map[col] = col.replace('r_gr_force', 'R_ground_force')
                elif 'l_gr_force' in col.lower():
                    rename_map[col] = col.replace('l_gr_force', 'L_ground_force')
            if rename_map:
                gold_df = gold_df.rename(columns=rename_map)
            
            # Recortar datos
            gold_markers_df_trimmed = gold_markers_df.iloc[gold_start_idx:].copy()
            gold_df_trimmed = gold_df[(gold_df['time'] >= gold_markers_df_trimmed['time'].iloc[0]) & 
                                      (gold_df['time'] <= gold_markers_df_trimmed['time'].iloc[-1])].copy()
            
            opencap_markers_df_trimmed = opencap_markers_df.iloc[opencap_start_idx:].copy()
            opencap_df_trimmed = opencap_df[(opencap_df['time'] >= opencap_markers_df_trimmed['time'].iloc[0]) & 
                                            (opencap_df['time'] <= opencap_markers_df_trimmed['time'].iloc[-1])].copy()
            
            # Normalizar tiempo a porcentaje
            gold_df_trimmed['time_relative'] = gold_df_trimmed['time'] - gold_df_trimmed['time'].iloc[0]
            gold_duration = gold_df_trimmed['time'].iloc[-1] - gold_df_trimmed['time'].iloc[0]
            gold_df_trimmed['time_percent'] = (gold_df_trimmed['time_relative'] / gold_duration * 100) if gold_duration > 0 else 0
            
            opencap_df_trimmed['time_relative'] = opencap_df_trimmed['time'] - opencap_df_trimmed['time'].iloc[0]
            opencap_duration = opencap_df_trimmed['time'].iloc[-1] - opencap_df_trimmed['time'].iloc[0]
            opencap_df_trimmed['time_percent'] = (opencap_df_trimmed['time_relative'] / opencap_duration * 100) if opencap_duration > 0 else 0
            
            # Procesar fuerza vertical pierna derecha
            col_name = 'R_ground_force_vy'
            if col_name not in gold_df_trimmed.columns or col_name not in opencap_df_trimmed.columns:
                return None
            
            # Filtrar rango 0-100%
            gold_common = gold_df_trimmed[(gold_df_trimmed['time_percent'] >= 0.0) & 
                                          (gold_df_trimmed['time_percent'] <= 100.0)]
            opencap_common = opencap_df_trimmed[(opencap_df_trimmed['time_percent'] >= 0.0) & 
                                                (opencap_df_trimmed['time_percent'] <= 100.0)]
            
            # Interpolar OpenCap al tiempo de Gold
            opencap_force_interp = np.interp(
                gold_common['time_percent'].values,
                opencap_common['time_percent'].values,
                opencap_common[col_name].values
            )
            
            gold_force_raw = gold_common[col_name].values
            
            # Normalizar ambas señales a [0, 1] (para correlación)
            gold_min, gold_max = np.min(gold_force_raw), np.max(gold_force_raw)
            gold_force_norm = (gold_force_raw - gold_min) / (gold_max - gold_min + 1e-10)
            
            opencap_min, opencap_max = np.min(opencap_force_interp), np.max(opencap_force_interp)
            opencap_force_norm = (opencap_force_interp - opencap_min) / (opencap_max - opencap_min + 1e-10)
            
            # Procesar posición vertical del talón
            # Normalizar tiempo de marcadores también
            gold_markers_df_trimmed['time_relative'] = gold_markers_df_trimmed['time'] - gold_markers_df_trimmed['time'].iloc[0]
            gold_markers_duration = gold_markers_df_trimmed['time'].iloc[-1] - gold_markers_df_trimmed['time'].iloc[0]
            gold_markers_df_trimmed['time_percent'] = (gold_markers_df_trimmed['time_relative'] / gold_markers_duration * 100) if gold_markers_duration > 0 else 0
            
            opencap_markers_df_trimmed['time_relative'] = opencap_markers_df_trimmed['time'] - opencap_markers_df_trimmed['time'].iloc[0]
            opencap_markers_duration = opencap_markers_df_trimmed['time'].iloc[-1] - opencap_markers_df_trimmed['time'].iloc[0]
            opencap_markers_df_trimmed['time_percent'] = (opencap_markers_df_trimmed['time_relative'] / opencap_markers_duration * 100) if opencap_markers_duration > 0 else 0
            
            # Extraer posición Y del talón (ya recortada)
            gold_heel_y_trimmed = gold_heel_y[gold_start_idx:]
            opencap_heel_y_trimmed = opencap_heel_y[opencap_start_idx:]
            
            # Interpolar posición del talón de OpenCap al tiempo de Gold (ya están alineados por índice)
            # Gold ya está alineado, solo necesitamos interpolar OpenCap
            opencap_heel_y_interp = np.interp(
                gold_markers_df_trimmed['time_percent'].values,
                opencap_markers_df_trimmed['time_percent'].values,
                opencap_heel_y_trimmed
            )
            
            # Gold no necesita interpolación, ya está alineado
            gold_heel_y_interp = gold_heel_y_trimmed
            
            return {
                'time_percent': gold_common['time_percent'].values,
                'gold_force_norm': gold_force_norm,
                'opencap_force_norm': opencap_force_norm,
                'gold_force_raw': gold_force_raw,
                'opencap_force_raw': opencap_force_interp,
                # Datos de posición del talón
                'heel_time_percent': gold_markers_df_trimmed['time_percent'].values,
                'gold_heel_y': gold_heel_y_interp,
                'opencap_heel_y': opencap_heel_y_interp
            }
            
        except Exception as e:
            print(f"   Error procesando {trial_name}: {e}")
            return None

    def generate_combined_curves_plot(self, squat_type='60'):
        """Genera gráfica combinada con posición del talón y fuerza vertical en subplots."""
        print(f"\nGenerando gráficas combinadas para sentadilla_{squat_type}...")
        
        # Cargar todos los trials
        trial_names = [f"sentadilla_{squat_type}_{i}" for i in range(1, 11)]
        all_trials_data = []
        
        for trial_name in trial_names:
            data = self.process_trial(trial_name)
            if data is not None:
                all_trials_data.append(data)
        
        if len(all_trials_data) == 0:
            print(f"   No se encontraron datos válidos para sentadilla_{squat_type}")
            return
        
        print(f"   Procesados {len(all_trials_data)} trials válidos")
        
        # Interpolar todas las señales a una grilla común de tiempo (0-100%, 200 puntos)
        time_grid = np.linspace(0, 100, 200)
        
        # Procesar datos de posición del talón
        gold_heel_signals = []
        opencap_heel_signals = []
        
        # Procesar datos de fuerza
        gold_force_signals = []
        opencap_force_signals = []
        
        for trial_data in all_trials_data:
            # Posición del talón
            gold_heel_interp = np.interp(time_grid, trial_data['heel_time_percent'], trial_data['gold_heel_y'])
            opencap_heel_interp = np.interp(time_grid, trial_data['heel_time_percent'], trial_data['opencap_heel_y'])
            gold_heel_signals.append(gold_heel_interp)
            opencap_heel_signals.append(opencap_heel_interp)
            
            # Fuerza
            gold_force_interp = np.interp(time_grid, trial_data['time_percent'], trial_data['gold_force_raw'])
            opencap_force_interp = np.interp(time_grid, trial_data['time_percent'], trial_data['opencap_force_raw'])
            gold_force_signals.append(gold_force_interp)
            opencap_force_signals.append(opencap_force_interp)
        
        gold_heel_signals = np.array(gold_heel_signals)
        opencap_heel_signals = np.array(opencap_heel_signals)
        gold_force_signals = np.array(gold_force_signals)
        opencap_force_signals = np.array(opencap_force_signals)
        
        # Calcular estadísticas para posición del talón
        gold_heel_mean = np.mean(gold_heel_signals, axis=0)
        gold_heel_std = np.std(gold_heel_signals, axis=0)
        opencap_heel_mean = np.mean(opencap_heel_signals, axis=0)
        opencap_heel_std = np.std(opencap_heel_signals, axis=0)
        
        # Calcular estadísticas para fuerza
        gold_force_mean = np.mean(gold_force_signals, axis=0)
        gold_force_std = np.std(gold_force_signals, axis=0)
        opencap_force_mean = np.mean(opencap_force_signals, axis=0)
        opencap_force_std = np.std(opencap_force_signals, axis=0)
        
        # Crear figura con dos subplots verticales
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Synchronization Validation - {squat_type}° Squat (n={len(all_trials_data)} trials, Right Leg)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # ========== SUBPLOT 1: Vertical Heel Position ==========
        # Gold Standard
        ax1.plot(time_grid, gold_heel_mean, 'b-', linewidth=2.5, label='Gold Standard (Mean)', alpha=0.9)
        ax1.fill_between(time_grid, gold_heel_mean - gold_heel_std, gold_heel_mean + gold_heel_std, 
                        alpha=0.3, color='blue', label='Gold Standard (±1 SD)')
        
        # OpenCap
        ax1.plot(time_grid, opencap_heel_mean, 'r--', linewidth=2.5, label='OpenCap (Mean)', alpha=0.9)
        ax1.fill_between(time_grid, opencap_heel_mean - opencap_heel_std, opencap_heel_mean + opencap_heel_std, 
                        alpha=0.3, color='red', label='OpenCap (±1 SD)')
        
        ax1.set_xlabel('Movement Cycle (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Vertical Heel Position (mm)', fontsize=12, fontweight='bold')
        ax1.set_title('Vertical Heel Position', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # ========== SUBPLOT 2: Vertical Force Complete Data ==========
        # Gold Standard
        ax2.plot(time_grid, gold_force_mean, 'b-', linewidth=2.5, label='Gold Standard (Mean)', alpha=0.9)
        ax2.fill_between(time_grid, gold_force_mean - gold_force_std, gold_force_mean + gold_force_std, 
                        alpha=0.3, color='blue', label='Gold Standard (±1 SD)')
        
        # OpenCap
        ax2.plot(time_grid, opencap_force_mean, 'r--', linewidth=2.5, label='OpenCap (Mean)', alpha=0.9)
        ax2.fill_between(time_grid, opencap_force_mean - opencap_force_std, opencap_force_mean + opencap_force_std, 
                        alpha=0.3, color='red', label='OpenCap (±1 SD)')
        
        ax2.set_xlabel('Movement Cycle (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vertical Force (N)', fontsize=12, fontweight='bold')
        ax2.set_title('Vertical Force Complete Data', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        output_path = self.validation_results_folder / f'curvas_combinadas_sentadilla_{squat_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Gráfica combinada guardada: {output_path}")
        plt.close()

    def generate_heel_position_plot(self, squat_type='60'):
        """Genera gráfica de curva promedio de posición vertical del talón."""
        print(f"\nGenerando curva promedio de posición del talón para sentadilla_{squat_type}...")
        
        # Cargar todos los trials
        trial_names = [f"sentadilla_{squat_type}_{i}" for i in range(1, 11)]
        all_trials_data = []
        
        for trial_name in trial_names:
            data = self.process_trial(trial_name)
            if data is not None:
                all_trials_data.append(data)
        
        if len(all_trials_data) == 0:
            print(f"   No se encontraron datos válidos para sentadilla_{squat_type}")
            return
        
        print(f"   Procesados {len(all_trials_data)} trials válidos")
        
        # Interpolar todas las señales a una grilla común de tiempo (0-100%, 200 puntos)
        time_grid = np.linspace(0, 100, 200)
        gold_heel_signals = []
        opencap_heel_signals = []
        
        for trial_data in all_trials_data:
            gold_heel_interp = np.interp(time_grid, trial_data['heel_time_percent'], trial_data['gold_heel_y'])
            opencap_heel_interp = np.interp(time_grid, trial_data['heel_time_percent'], trial_data['opencap_heel_y'])
            gold_heel_signals.append(gold_heel_interp)
            opencap_heel_signals.append(opencap_heel_interp)
        
        gold_heel_signals = np.array(gold_heel_signals)
        opencap_heel_signals = np.array(opencap_heel_signals)
        
        # Calcular estadísticas
        gold_mean = np.mean(gold_heel_signals, axis=0)
        gold_std = np.std(gold_heel_signals, axis=0)
        opencap_mean = np.mean(opencap_heel_signals, axis=0)
        opencap_std = np.std(opencap_heel_signals, axis=0)
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Gold Standard
        ax.plot(time_grid, gold_mean, 'b-', linewidth=2.5, label='Gold Standard (Mean)', alpha=0.9)
        ax.fill_between(time_grid, gold_mean - gold_std, gold_mean + gold_std, 
                       alpha=0.3, color='blue', label='Gold Standard (±1 SD)')
        
        # OpenCap
        ax.plot(time_grid, opencap_mean, 'r--', linewidth=2.5, label='OpenCap (Mean)', alpha=0.9)
        ax.fill_between(time_grid, opencap_mean - opencap_std, opencap_mean + opencap_std, 
                       alpha=0.3, color='red', label='OpenCap (±1 SD)')
        
        ax.set_xlabel('Movement Cycle (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vertical Heel Position (mm)', fontsize=12, fontweight='bold')
        ax.set_title(f'Vertical Heel Position - {squat_type}° Squat\n'
                    f'(n={len(all_trials_data)} trials, Right Leg)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        output_path = self.validation_results_folder / f'curva_promedio_talon_sentadilla_{squat_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Gráfica guardada: {output_path}")
        plt.close()

    def generate_force_complete_plot(self, squat_type='60'):
        """Genera gráfica de curva promedio de fuerza vertical datos completos."""
        print(f"\nGenerando curva promedio de fuerza completa para sentadilla_{squat_type}...")
        
        # Cargar todos los trials
        trial_names = [f"sentadilla_{squat_type}_{i}" for i in range(1, 11)]
        all_trials_data = []
        
        for trial_name in trial_names:
            data = self.process_trial(trial_name)
            if data is not None:
                all_trials_data.append(data)
        
        if len(all_trials_data) == 0:
            print(f"   No se encontraron datos válidos para sentadilla_{squat_type}")
            return
        
        print(f"   Procesados {len(all_trials_data)} trials válidos")
        
        # Interpolar todas las señales a una grilla común de tiempo (0-100%, 200 puntos)
        time_grid = np.linspace(0, 100, 200)
        gold_force_signals = []
        opencap_force_signals = []
        
        for trial_data in all_trials_data:
            gold_force_interp = np.interp(time_grid, trial_data['time_percent'], trial_data['gold_force_raw'])
            opencap_force_interp = np.interp(time_grid, trial_data['time_percent'], trial_data['opencap_force_raw'])
            gold_force_signals.append(gold_force_interp)
            opencap_force_signals.append(opencap_force_interp)
        
        gold_force_signals = np.array(gold_force_signals)
        opencap_force_signals = np.array(opencap_force_signals)
        
        # Calcular estadísticas
        gold_mean = np.mean(gold_force_signals, axis=0)
        gold_std = np.std(gold_force_signals, axis=0)
        opencap_mean = np.mean(opencap_force_signals, axis=0)
        opencap_std = np.std(opencap_force_signals, axis=0)
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Gold Standard
        ax.plot(time_grid, gold_mean, 'b-', linewidth=2.5, label='Gold Standard (Mean)', alpha=0.9)
        ax.fill_between(time_grid, gold_mean - gold_std, gold_mean + gold_std, 
                       alpha=0.3, color='blue', label='Gold Standard (±1 SD)')
        
        # OpenCap
        ax.plot(time_grid, opencap_mean, 'r--', linewidth=2.5, label='OpenCap (Mean)', alpha=0.9)
        ax.fill_between(time_grid, opencap_mean - opencap_std, opencap_mean + opencap_std, 
                       alpha=0.3, color='red', label='OpenCap (±1 SD)')
        
        ax.set_xlabel('Movement Cycle (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vertical Force (N)', fontsize=12, fontweight='bold')
        ax.set_title(f'Vertical Force Complete Data - {squat_type}° Squat\n'
                    f'(n={len(all_trials_data)} trials, Right Leg)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        output_path = self.validation_results_folder / f'curva_promedio_fuerza_sentadilla_{squat_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Gráfica guardada: {output_path}")
        plt.close()

    def generate_correlation_scatter_plot(self, squat_type='60'):
        """Genera diagrama de dispersión con dos subplots: todos los datos y zoom en región de mayor densidad."""
        print(f"\nGenerando diagrama de dispersión para sentadilla_{squat_type}...")
        
        # Cargar todos los trials
        trial_names = [f"sentadilla_{squat_type}_{i}" for i in range(1, 11)]
        all_gold_values = []
        all_opencap_values = []
        correlations = []
        
        for trial_name in trial_names:
            data = self.process_trial(trial_name)
            if data is not None:
                # Usar valores normalizados para el scatter plot
                all_gold_values.extend(data['gold_force_norm'])
                all_opencap_values.extend(data['opencap_force_norm'])
                
                # Calcular correlación para este trial
                if len(data['gold_force_norm']) > 1:
                    corr, _ = pearsonr(data['gold_force_norm'], data['opencap_force_norm'])
                    correlations.append(corr)
        
        if len(all_gold_values) == 0:
            print(f"   No se encontraron datos válidos para sentadilla_{squat_type}")
            return
        
        all_gold_values = np.array(all_gold_values)
        all_opencap_values = np.array(all_opencap_values)
        
        # Guardar todos los datos completos para calcular correlación global
        all_gold_full = np.array(all_gold_values)
        all_opencap_full = np.array(all_opencap_values)
        
        # Calcular correlación global con todos los datos
        global_corr, global_p = pearsonr(all_gold_full, all_opencap_full)
        avg_corr = np.mean(correlations) if correlations else global_corr
        
        # Muestrear si hay demasiados puntos para mejorar la visualización
        max_points = 5000
        all_gold_plot = all_gold_values.copy()
        all_opencap_plot = all_opencap_values.copy()
        if len(all_gold_values) > max_points:
            step = len(all_gold_values) // max_points
            indices = np.arange(0, len(all_gold_values), step)
            all_gold_plot = all_gold_values[indices]
            all_opencap_plot = all_opencap_values[indices]
            print(f"   Muestreado a {len(all_gold_plot)} puntos para visualización")
        
        # Calcular regresión lineal con todos los datos (antes del muestreo)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_gold_values, all_opencap_values)
        
        # ========== Cálculo de densidad KDE para zoom agresivo ==========
        # Calcular densidad KDE bidimensional
        xy = np.vstack([all_gold_values, all_opencap_values])
        kde = gaussian_kde(xy)
        densities = kde(xy)
        
        # Seleccionar los puntos del top 80% de densidad (zoom agresivo)
        density_threshold = np.percentile(densities, 90)  # Top 80% = percentil 20
        mask_zoom = densities >= density_threshold
        
        gold_zoom = all_gold_values[mask_zoom]
        opencap_zoom = all_opencap_values[mask_zoom]
        
        # Crear figura con dos subplots horizontales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Correlation Gold Standard vs OpenCap - {squat_type}° Squat', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # ========== SUBPLOT 1: Todos los datos ==========
        # Scatter plot con todos los datos
        ax1.scatter(all_gold_plot, all_opencap_plot, 
                   alpha=0.4, s=8, c='steelblue', edgecolors='none', 
                   rasterized=True)
        
        # Líneas para el subplot completo
        min_val = min(np.min(all_gold_values), np.min(all_opencap_values))
        max_val = max(np.max(all_gold_values), np.max(all_opencap_values))
        line_x_full = np.linspace(min_val, max_val, 100)
        line_y_full = slope * line_x_full + intercept
        
        # Línea de identidad
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2.5, label='Identity Line (y=x)', alpha=0.8)
        
        # Línea de regresión
        ax1.plot(line_x_full, line_y_full, 'g-', linewidth=2.5, 
                label=f'Linear Regression (y={slope:.3f}x+{intercept:.3f})', alpha=0.8)
        
        # Ajustar límites
        margin = 0.05
        ax1.set_xlim(min_val - margin, max_val + margin)
        ax1.set_ylim(min_val - margin, max_val + margin)
        
        ax1.set_xlabel('Gold Standard - Normalized Force [0-1]', fontsize=13, fontweight='bold')
        ax1.set_ylabel('OpenCap - Normalized Force [0-1]', fontsize=13, fontweight='bold')
        ax1.set_title('All Data', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Estadísticas en el primer subplot
        textstr1 = f'r = {global_corr:.4f} (p < 0.001)\n'
        textstr1 += f'Mean r = {avg_corr:.4f}\n'
        textstr1 += f'n = {len(all_gold_full)} points\n'
        textstr1 += f'{len(correlations)} trials'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black')
        ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props, family='monospace')
        
        # ========== SUBPLOT 2: Zoom en región de mayor densidad (KDE) ==========
        
        # Muestrear si es necesario
        if len(gold_zoom) > max_points:
            step = len(gold_zoom) // max_points
            indices = np.arange(0, len(gold_zoom), step)
            gold_zoom = gold_zoom[indices]
            opencap_zoom = opencap_zoom[indices]
        
        # Scatter plot con zoom
        ax2.scatter(gold_zoom, opencap_zoom, 
                   alpha=0.5, s=10, c='steelblue', edgecolors='none', 
                   rasterized=True)
        
        # Calcular límites reales de los datos filtrados para el zoom
        zoom_min_x = np.min(gold_zoom)
        zoom_max_x = np.max(gold_zoom)
        zoom_min_y = np.min(opencap_zoom)
        zoom_max_y = np.max(opencap_zoom)
        
        # Líneas para el subplot de zoom (usar los límites reales de los datos)
        line_x_zoom = np.linspace(zoom_min_x, zoom_max_x, 100)
        line_y_zoom = slope * line_x_zoom + intercept
        
        # Línea de identidad (usar el rango común de ambos ejes)
        zoom_common_min = min(zoom_min_x, zoom_min_y)
        zoom_common_max = max(zoom_max_x, zoom_max_y)
        ax2.plot([zoom_common_min, zoom_common_max], [zoom_common_min, zoom_common_max], 
                'r--', linewidth=2.5, label='Identity Line (y=x)', alpha=0.8)
        
        # Línea de regresión
        ax2.plot(line_x_zoom, line_y_zoom, 'g-', linewidth=2.5, 
                label=f'Linear Regression (y={slope:.3f}x+{intercept:.3f})', alpha=0.8)
        
        # Ajustar límites con un pequeño margen porcentual basado en el rango de cada eje
        zoom_margin_x = (zoom_max_x - zoom_min_x) * 0.05
        zoom_margin_y = (zoom_max_y - zoom_min_y) * 0.05
        ax2.set_xlim(zoom_min_x - zoom_margin_x, zoom_max_x + zoom_margin_x)
        ax2.set_ylim(zoom_min_y - zoom_margin_y, zoom_max_y + zoom_margin_y)
        
        ax2.set_xlabel('Gold Standard - Normalized Force [0-1]', fontsize=13, fontweight='bold')
        ax2.set_ylabel('OpenCap - Normalized Force [0-1]', fontsize=13, fontweight='bold')
        ax2.set_title('Zoom: High Density Region (KDE Top 80%)', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='lower right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Estadísticas en el segundo subplot
        corr_zoom, _ = pearsonr(gold_zoom, opencap_zoom)
        textstr2 = f'r = {corr_zoom:.4f}\n'
        textstr2 += f'n = {len(gold_zoom)} points\n'
        textstr2 += f'Region: KDE Top 80%\ndensity'
        ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props, family='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = self.validation_results_folder / f'correlacion_scatter_sentadilla_{squat_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Gráfica guardada: {output_path}")
        plt.close()

    def run(self):
        """Ejecuta la generación de todas las gráficas."""
        print("="*60)
        print("GENERACIÓN DE GRÁFICAS AGREGADAS")
        print("="*60)
        
        # Gráficas combinadas (posición talón + fuerza)
        self.generate_combined_curves_plot('60')
        self.generate_combined_curves_plot('90')
        
        # Curvas promedio individuales (opcionales, para referencia)
        # self.generate_heel_position_plot('60')
        # self.generate_heel_position_plot('90')
        # self.generate_force_complete_plot('60')
        # self.generate_force_complete_plot('90')
        
        # Diagramas de dispersión
        self.generate_correlation_scatter_plot('60')
        self.generate_correlation_scatter_plot('90')
        
        print("\n" + "="*60)
        print("GENERACIÓN COMPLETADA")
        print("="*60)


if __name__ == "__main__":
    generator = AggregatePlotGenerator()
    generator.run()


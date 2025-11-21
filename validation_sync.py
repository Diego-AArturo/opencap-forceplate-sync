"""
Script de validación de sincronización OpenCap vs Gold Standard (Motive)

Compara la sincronización algorítmica de OpenCap con la sincronización por hardware 
del sistema gold standard para validar la precisión del algoritmo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate, find_peaks
from scipy.ndimage import uniform_filter1d
import json
from pathlib import Path

class SyncValidator:
    def __init__(self, gold_standard_mot_path, gold_standard_trc_path, 
                 opencap_syncd_mot_path, opencap_trc_path, output_folder="validation_results"):
        """
        Inicializa el validador de sincronización.
        
        Args:
            gold_standard_mot_path: Ruta al archivo MOT del gold standard (Plataformas)
            gold_standard_trc_path: Ruta al archivo TRC del gold standard (Marcadores Motive)
            opencap_syncd_mot_path: Ruta al archivo MOT sincronizado de OpenCap
            opencap_trc_path: Ruta al archivo TRC de OpenCap (Marcadores)
            output_folder: Carpeta para guardar resultados
        """
        self.gold_mot_path = Path(gold_standard_mot_path)
        self.gold_trc_path = Path(gold_standard_trc_path)
        self.opencap_mot_path = Path(opencap_syncd_mot_path)
        self.opencap_trc_path = Path(opencap_trc_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.trial_name = self.opencap_mot_path.stem.replace('_syncd_forces', '')
    
    def _read_mot_file(self, file_path):
        """Lee un archivo .mot y devuelve un DataFrame."""
        print(f"Leyendo archivo MOT: {file_path.name}")
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
        print(f"   Datos cargados: {df.shape[0]} frames, {df.shape[1]} columnas")
        return df

    def _read_trc_file(self, file_path):
        """Lee un archivo .trc y devuelve un DataFrame y diccionario de marcadores."""
        print(f"Leyendo archivo TRC: {file_path.name}")
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
        
        # Crear diccionario de marcadores con sus posiciones xyz
        markers_dict = {
            name: {
                'x': df[f'{name}_x'].values,
                'y': df[f'{name}_y'].values,
                'z': df[f'{name}_z'].values
            }
            for name in marker_names if f'{name}_x' in df.columns
        }
        
        print(f"   Marcadores cargados: {len(markers_dict)} marcadores, {len(df)} frames")
        return df, markers_dict, list(markers_dict.keys())

    def load_gold_standard_forces(self):
        """Carga fuerzas del archivo MOT del gold standard."""
        df = self._read_mot_file(self.gold_mot_path)
        
        # Renombrar columnas de fuerzas derecha (r_gr_force) e izquierda (l_gr_force)
        rename_map = {}
        for col in df.columns:
            if 'r_gr_force' in col.lower():
                rename_map[col] = col.replace('r_gr_force', 'R_ground_force')
            elif 'l_gr_force' in col.lower():
                rename_map[col] = col.replace('l_gr_force', 'L_ground_force')
        
        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"   Columnas renombradas para consistencia")
        
        return df
    
    def load_opencap_synced_forces(self):
        """Carga fuerzas sincronizadas de OpenCap."""
        return self._read_mot_file(self.opencap_mot_path)
    
    def load_gold_markers(self):
        """Carga marcadores del TRC gold standard (Motive)."""
        return self._read_trc_file(self.gold_trc_path)
    
    def load_opencap_markers(self):
        """Carga marcadores del archivo TRC de OpenCap."""
        return self._read_trc_file(self.opencap_trc_path)
    
    def find_step_on_and_squat_boundaries(self, heel_y, time, min_duration=2.0, max_duration=12.0):
        """
        Detecta el inicio del movimiento (despegue del talón).
        """
        heel_y_norm = (heel_y - np.min(heel_y)) / (np.max(heel_y) - np.min(heel_y))

        # 1. Encontrar el evento de "pisar" (el pico más prominente al inicio)
        peaks, _ = find_peaks(heel_y_norm, height=0.5, prominence=0.3, distance=100)
        
        if len(peaks) == 0:
            print("      ADVERTENCIA: No se encontró un pico prominente (paso inicial).")
            return None

        step_on_peak_idx = peaks[0]
        
        # 2. Encontrar el inicio del despegue (takeoff) buscando hacia atrás desde el pico
        search_window_before = 120  # frames
        search_start = max(0, step_on_peak_idx - search_window_before)
        takeoff_region = heel_y_norm[search_start:step_on_peak_idx]
        
        min_before_peak_idx = search_start + np.argmin(takeoff_region) if len(takeoff_region) > 0 else search_start
        start_idx = min_before_peak_idx

        print(f"      Análisis del paso:")
        print(f"         - Pico del paso: índice={step_on_peak_idx}, tiempo={time[step_on_peak_idx]:.3f}s")
        print(f"         - Inicio (despegue): índice={start_idx}, tiempo={time[start_idx]:.3f}s")

        return start_idx
        
    def align_signals_by_markers(self, gold_markers_df, opencap_markers_df, 
                                 gold_markers_dict, opencap_markers_dict, 
                                 gold_marker_names, opencap_marker_names):
        """
        Detecta los límites del movimiento basándose en marcadores de talón y retorna información de alineación.
        """
        print("\nDetectando límites del movimiento en marcadores...")
        
        # Buscar marcadores de talón en OpenCap (RHeel, LHeel)
        opencap_heels = {name for name in opencap_marker_names if 'heel' in name.lower()}
        opencap_rheel = next((name for name in opencap_heels if 'r' in name.lower()), None)
        opencap_lheel = next((name for name in opencap_heels if 'l' in name.lower()), None)

        # Buscar marcadores de talón en Gold Standard (Unlabeled 1025, Unlabeled 1008)
        gold_heels = {name for name in gold_marker_names 
                     if any(x in name.lower() for x in ['unlabeled 1025', 'unlabeled1025', 
                                                          'unlabeled 1008', 'unlabeled1008', 'heel'])}
        gold_rheel = next((name for name in gold_heels if '1025' in name or 'r' in name.lower()), None)
        gold_lheel = next((name for name in gold_heels if '1008' in name or 'l' in name.lower()), None)

        # Seleccionar un pie para alinear (preferir derecho)
        if opencap_rheel and gold_rheel:
            opencap_heel_name, gold_heel_name, heel_side = opencap_rheel, gold_rheel, "derecho"
        elif opencap_lheel and gold_lheel:
            opencap_heel_name, gold_heel_name, heel_side = opencap_lheel, gold_lheel, "izquierdo"
        else:
            print("   ADVERTENCIA: No se encontraron marcadores de talón coincidentes")
            return None
        
        print(f"   Usando marcadores de talón {heel_side}:")
        print(f"      OpenCap: '{opencap_heel_name}'")
        print(f"      Gold: '{gold_heel_name}'")
        
        # Detectar límites de movimiento
        opencap_heel_y = np.array(opencap_markers_dict[opencap_heel_name]['y'])
        gold_heel_y = np.array(gold_markers_dict[gold_heel_name]['y'])
        
        print(f"\n   Procesando OpenCap...")
        opencap_start_idx = self.find_step_on_and_squat_boundaries(
            opencap_heel_y, opencap_markers_df['time'].values
        )
        
        print(f"\n   Procesando Gold Standard...")
        gold_start_idx = self.find_step_on_and_squat_boundaries(
            gold_heel_y, gold_markers_df['time'].values
        )
        
        if opencap_start_idx is None or gold_start_idx is None:
            return None
        
        print(f"\n   Limites del movimiento detectados")
        
        return {
            'opencap_start_idx': opencap_start_idx,
            'opencap_end_idx': len(opencap_markers_df) - 1,
            'opencap_start_time': opencap_markers_df['time'].iloc[opencap_start_idx],
            'opencap_end_time': opencap_markers_df['time'].iloc[-1],
            'gold_start_idx': gold_start_idx,
            'gold_end_idx': len(gold_markers_df) - 1,
            'gold_start_time': gold_markers_df['time'].iloc[gold_start_idx],
            'gold_end_time': gold_markers_df['time'].iloc[-1],
            'heel_side': heel_side,
            'opencap_heel_name': opencap_heel_name,
            'gold_heel_name': gold_heel_name
        }
    
    def _normalize_time_percent(self, df):
        """Normaliza el tiempo a porcentaje del ciclo (0-100%)."""
        df['time_relative'] = df['time'] - df['time'].iloc[0]
        duration = df['time'].iloc[-1] - df['time'].iloc[0]
        df['time_percent'] = (df['time_relative'] / duration * 100) if duration > 0 else 0
        return df, duration
    
    def calculate_metrics(self, gold_df, opencap_df):
        """
        Calcula métricas de validación entre gold standard y OpenCap.
        Las señales de fuerza se normalizan a [0, 1] para validar sincronización y forma.
        """
        print("\nCalculando métricas de validación...")
        
        metrics = {}
        time_col = 'time_percent'  # Siempre usar porcentaje
        
        # Usar rango 0-100% (normalizado)
        time_start = 0.0
        time_end = 100.0
        
        gold_common = gold_df[(gold_df[time_col] >= time_start) & (gold_df[time_col] <= time_end)]
        opencap_common = opencap_df[(opencap_df[time_col] >= time_start) & (opencap_df[time_col] <= time_end)]
        
        print(f"   Rango temporal: {time_start:.1f}% - {time_end:.1f}% del ciclo")
        print(f"   Gold frames: {len(gold_common)}, OpenCap frames: {len(opencap_common)}")
        
        # Comparar fuerzas verticales para ambas piernas
        for leg in ['R', 'L']:
            col_name = f'{leg}_ground_force_vy'
            
            if col_name not in gold_common.columns or col_name not in opencap_common.columns:
                continue
            
            # Obtener señales originales
            gold_force_raw = gold_common[col_name].values
            opencap_force_raw = opencap_common[col_name].values
            
            # Interpolar OpenCap al porcentaje de gold
            opencap_force_raw = np.interp(
                gold_common[time_col].values,
                opencap_common[time_col].values,
                opencap_force_raw
            )
            
            # Normalizar ambas señales a [0, 1]
            gold_min, gold_max = np.min(gold_force_raw), np.max(gold_force_raw)
            gold_force = (gold_force_raw - gold_min) / (gold_max - gold_min + 1e-10)
            
            opencap_min, opencap_max = np.min(opencap_force_raw), np.max(opencap_force_raw)
            opencap_force = (opencap_force_raw - opencap_min) / (opencap_max - opencap_min + 1e-10)
            
            # Calcular métricas sobre señales normalizadas
            rmse = np.sqrt(np.mean((gold_force - opencap_force)**2))
            mae = np.mean(np.abs(gold_force - opencap_force))
            
            # Correlación (evitar errores si hay poca varianza)
            if len(gold_force) > 1 and np.std(gold_force) > 0 and np.std(opencap_force) > 0:
                correlation, p_value = pearsonr(gold_force, opencap_force)
            else:
                correlation, p_value = 0.0, 1.0
            
            metrics[leg] = {
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation,
                'p_value': p_value,
                'sync_error_ms': None
            }
            
            print(f"\n   Pierna {leg}:")
            print(f"      RMSE (normalizado): {rmse:.4f}")
            print(f"      MAE (normalizado): {mae:.4f}")
            print(f"      Correlación: {correlation:.4f} (p={p_value:.4e})")
        
        return metrics, gold_common, opencap_common
    
    def plot_comparison(self, gold_df, opencap_df, metrics, 
                       gold_markers_df=None, gold_markers_dict=None, gold_marker_names=None,
                       opencap_markers_df=None, opencap_markers_dict=None, opencap_marker_names=None,
                       alignment_info=None):
        """
        Genera gráficos de comparación entre gold standard y OpenCap.
        """
        print("\nGenerando gráficos...")
        
        time_col = 'time_percent'  # Siempre usar porcentaje
        time_label = 'Ciclo (%)'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Validacion de Sincronizacion: {self.trial_name}', fontsize=16, fontweight='bold')
        
        # Usar rango 0-100% del ciclo
        time_start = 0.0
        time_end = 100.0
        
        # Función auxiliar para plotear fuerza
        def plot_force_leg(ax, col, leg):
            if col not in gold_df.columns or col not in opencap_df.columns:
                return False
            
            gold_mask = (gold_df[time_col] >= time_start) & (gold_df[time_col] <= time_end)
            opencap_mask = (opencap_df[time_col] >= time_start) & (opencap_df[time_col] <= time_end)
            
            gold_data = gold_df[gold_mask]
            opencap_data = opencap_df[opencap_mask]
            
            # Ambas fuerzas están a 1000 Hz, graficar directamente sin interpolar
            # Normalizar señales a [0, 1] para enfocarse en sincronización y forma
            gold_force = gold_data[col].values
            opencap_force = opencap_data[col].values
            
            # Normalizar gold
            gold_min, gold_max = np.min(gold_force), np.max(gold_force)
            gold_norm = (gold_force - gold_min) / (gold_max - gold_min + 1e-10)
            
            # Normalizar opencap
            opencap_min, opencap_max = np.min(opencap_force), np.max(opencap_force)
            opencap_norm = (opencap_force - opencap_min) / (opencap_max - opencap_min + 1e-10)
            
            ax.plot(gold_data[time_col], gold_norm, 
                   label='Gold Standard', linewidth=2, alpha=0.7)
            ax.plot(opencap_data[time_col], opencap_norm, 
                   label='OpenCap Sync', linewidth=1.5, alpha=0.7, linestyle='--')
            
            title = f'Fuerza Vertical Pierna {leg} (Normalizada)'
            if leg in metrics:
                title += f"\nRMSE={metrics[leg]['rmse']:.4f}, r={metrics[leg]['correlation']:.3f}"
            
            ax.set_title(title)
            ax.set_xlabel(time_label)
            ax.set_ylabel('Fuerza (Normalizada [0-1])')
            ax.legend()
            ax.grid(True, alpha=0.3)
            return True
        
        # Plotear fuerzas
        plot_force_leg(axes[0, 0], 'R_ground_force_vy', 'Derecha')
        
        # Gráfica derecha: Datos COMPLETOS de Pierna Derecha sin recorte
        ax = axes[0, 1]
        col = 'R_ground_force_vy'
        if col in gold_df.columns and col in opencap_df.columns:
            # Graficar datos COMPLETOS sin recorte por porcentaje
            ax.plot(gold_df[time_col], gold_df[col], 
                   label='Gold Standard', linewidth=2, alpha=0.7)
            ax.plot(opencap_df[time_col], opencap_df[col], 
                   label='OpenCap Sync', linewidth=1.5, alpha=0.7, linestyle='--')
            
            title = f'Fuerza Vertical Pierna Derecha (Datos Completos)'
            if 'R' in metrics:
                title += f"\nRMSE={metrics['R']['rmse']:.2f}N, r={metrics['R']['correlation']:.3f}"
            
            ax.set_title(title)
            ax.set_xlabel(time_label)
            ax.set_ylabel('Fuerza (N)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plotear marcadores de talón si están disponibles
        if (gold_markers_df is not None and opencap_markers_df is not None and 
            gold_markers_dict is not None and opencap_markers_dict is not None and
            alignment_info is not None):
            
            ax = axes[1, 0]
            opencap_heel_name = alignment_info['opencap_heel_name']
            gold_heel_name = alignment_info['gold_heel_name']
            
            if opencap_heel_name and gold_heel_name:
                # Normalizar posiciones de talón
                gold_heel_y = np.array(gold_markers_dict[gold_heel_name]['y'])[alignment_info['gold_start_idx']:alignment_info['gold_end_idx'] + 1]
                gold_heel_y_norm = (gold_heel_y - np.min(gold_heel_y)) / (np.max(gold_heel_y) - np.min(gold_heel_y) + 1e-6)
                
                opencap_heel_y = np.array(opencap_markers_dict[opencap_heel_name]['y'])[alignment_info['opencap_start_idx']:alignment_info['opencap_end_idx'] + 1]
                opencap_heel_y_norm = (opencap_heel_y - np.min(opencap_heel_y)) / (np.max(opencap_heel_y) - np.min(opencap_heel_y) + 1e-6)

                ax.plot(gold_markers_df['time_percent'], gold_heel_y_norm, 
                       label=f'Gold Standard', linewidth=2, alpha=0.8)
                ax.plot(opencap_markers_df['time_percent'], opencap_heel_y_norm, 
                       label=f'OpenCap', linewidth=2, alpha=0.8, linestyle='--')
                
                ax.set_title('Posicion Vertical Talon (Normalizada)')
                ax.set_xlabel('Ciclo (%)')
                ax.set_ylabel('Posicion Y (norm)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Scatter plot - correlación pierna derecha
        ax = axes[1, 1]
        if 'R_ground_force_vy' in gold_df.columns and 'R_ground_force_vy' in opencap_df.columns and 'R' in metrics:
            gold_common = gold_df[(gold_df[time_col] >= time_start) & (gold_df[time_col] <= time_end)]
            opencap_common = opencap_df[(opencap_df[time_col] >= time_start) & (opencap_df[time_col] <= time_end)]
            
            opencap_interp = np.interp(
                gold_common[time_col].values,
                opencap_common[time_col].values,
                opencap_common['R_ground_force_vy'].values
            )
            
            ax.scatter(gold_common['R_ground_force_vy'].values, opencap_interp, alpha=0.5, s=10)
            
            # Línea de identidad
            min_val = min(gold_common['R_ground_force_vy'].min(), opencap_interp.min())
            max_val = max(gold_common['R_ground_force_vy'].max(), opencap_interp.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Linea identidad')
            
            ax.set_title(f'Correlacion Pierna Derecha\nr = {metrics["R"]["correlation"]:.3f}')
            ax.set_xlabel('Gold Standard (N)')
            ax.set_ylabel('OpenCap Sync (N)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Guardar figura
        output_path = self.output_folder / f'{self.trial_name}_validation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Grafico guardado: {output_path}")
        plt.close()
    
    def save_report(self, metrics):
        """Guarda reporte de validación en JSON."""
        excluded_fields = ['gold_duration_s', 'opencap_duration_s', 'duration_difference_ms', 
                          'heel_side_used', 'alignment_method']
        
        report = {
            'trial_name': self.trial_name,
            'gold_standard_forces': str(self.gold_mot_path),
            'gold_standard_markers': str(self.gold_trc_path),
            'opencap_forces': str(self.opencap_mot_path),
            'opencap_markers': str(self.opencap_trc_path),
            'alignment_method': metrics.get('alignment_method', 'unknown'),
            'heel_side_used': metrics.get('heel_side_used', 'unknown'),
            'gold_duration_s': float(metrics.get('gold_duration_s', 0)),
            'opencap_duration_s': float(metrics.get('opencap_duration_s', 0)),
            'duration_difference_ms': float(metrics.get('duration_difference_ms', 0)),
            'metrics': {}
        }
        
        for leg, leg_metrics in metrics.items():
            if leg not in excluded_fields and isinstance(leg_metrics, dict):
                report['metrics'][leg] = {
                    'rmse': float(leg_metrics['rmse']),
                    'mae': float(leg_metrics['mae']),
                    'correlation': float(leg_metrics['correlation']),
                    'p_value': float(leg_metrics['p_value']),
                    'sync_error_ms': float(leg_metrics.get('sync_error_ms', 0)) if leg_metrics.get('sync_error_ms') else None
                }
        
        output_path = self.output_folder / f'{self.trial_name}_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"   Reporte guardado: {output_path}")
    
    def run_validation(self):
        """Ejecuta el flujo completo de validación."""
        print(f"\n{'='*60}")
        print(f"VALIDACION DE SINCRONIZACION")
        print(f"{'='*60}")
        print(f"Trial: {self.trial_name}")
        
        try:
            # Cargar datos
            gold_markers_df, gold_markers_dict, gold_marker_names = self.load_gold_markers()
            opencap_markers_df, opencap_markers_dict, opencap_marker_names = self.load_opencap_markers()
            
            # Detectar límites del movimiento
            alignment_info = self.align_signals_by_markers(
                gold_markers_df, opencap_markers_df,
                gold_markers_dict, opencap_markers_dict,
                gold_marker_names, opencap_marker_names
            )
            
            if alignment_info is None:
                print("ERROR: No se pudo alinear por marcadores")
                return None
            
            # Cargar fuerzas
            gold_df = self.load_gold_standard_forces()
            opencap_df = self.load_opencap_synced_forces()
            
            # Recortar datos
            print(f"\nRecortando señales...")
            
            # Obtener índices de inicio de ambos sistemas
            gold_start_idx = alignment_info['gold_start_idx']
            opencap_start_idx = alignment_info['opencap_start_idx']
            
            # Recortar desde el inicio detectado hasta el final
            gold_markers_df_trimmed = gold_markers_df.iloc[gold_start_idx:].copy()
            gold_df_trimmed = gold_df[(gold_df['time'] >= gold_markers_df_trimmed['time'].iloc[0]) & 
                                      (gold_df['time'] <= gold_markers_df_trimmed['time'].iloc[-1])].copy()
            
            opencap_markers_df_trimmed = opencap_markers_df.iloc[opencap_start_idx:].copy()
            opencap_df_trimmed = opencap_df[(opencap_df['time'] >= opencap_markers_df_trimmed['time'].iloc[0]) & 
                                            (opencap_df['time'] <= opencap_markers_df_trimmed['time'].iloc[-1])].copy()
            
            print(f"   Gold: {len(gold_df_trimmed)} frames de fuerza, {len(gold_markers_df_trimmed)} frames de marcadores")
            print(f"   OpenCap: {len(opencap_df_trimmed)} frames de fuerza, {len(opencap_markers_df_trimmed)} frames de marcadores")
            
            # Normalizar tiempo a porcentaje del ciclo (0-100%)
            print(f"\nNormalizando tiempo a porcentaje...")
            gold_df_trimmed, gold_duration = self._normalize_time_percent(gold_df_trimmed)
            opencap_df_trimmed, opencap_duration = self._normalize_time_percent(opencap_df_trimmed)
            gold_markers_df_trimmed, _ = self._normalize_time_percent(gold_markers_df_trimmed)
            opencap_markers_df_trimmed, _ = self._normalize_time_percent(opencap_markers_df_trimmed)
            
            print(f"   Gold: {gold_duration:.3f}s")
            print(f"   OpenCap: {opencap_duration:.3f}s")
            print(f"   Diferencia: {abs(gold_duration - opencap_duration)*1000:.2f}ms")
            print(f"   (Nota: Comparación ahora en porcentaje del ciclo, no en tiempo absoluto)")
            
            # Calcular métricas
            metrics, _, _ = self.calculate_metrics(gold_df_trimmed, opencap_df_trimmed)
            
            # Añadir información de duración
            metrics['gold_duration_s'] = gold_duration
            metrics['opencap_duration_s'] = opencap_duration
            metrics['duration_difference_ms'] = (gold_duration - opencap_duration) * 1000
            metrics['heel_side_used'] = alignment_info['heel_side']
            metrics['alignment_method'] = 'movement_boundaries_from_heel_strikes'
            
            # Generar gráficos
            self.plot_comparison(gold_df_trimmed, opencap_df_trimmed, metrics,
                               gold_markers_df_trimmed, gold_markers_dict, gold_marker_names,
                               opencap_markers_df_trimmed, opencap_markers_dict, opencap_marker_names,
                               alignment_info)
            
            # Guardar reporte
            self.save_report(metrics)
            
            print(f"\n{'='*60}")
            print(f"VALIDACION COMPLETADA")
            print(f"   Talon usado: {alignment_info['heel_side']}")
            print(f"   Duracion Gold: {gold_duration:.3f}s")
            print(f"   Duracion OpenCap: {opencap_duration:.3f}s")
            if 'R' in metrics:
                print(f"   Pierna Derecha RMSE: {metrics['R']['rmse']:.2f}N (r={metrics['R']['correlation']:.3f})")
            if 'L' in metrics:
                print(f"   Pierna Izquierda RMSE: {metrics['L']['rmse']:.2f}N (r={metrics['L']['correlation']:.3f})")
            print(f"{'='*60}\n")
            
            return metrics
            
        except Exception as e:
            print(f"\nERROR en validacion: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Configuración
    participant_id = "P31T"
    names = ["sentadilla_90_"]
    trial_names = [name + str(i) for name in names for i in range(1, 11)]
    
    # Rutas base
    gold_forces_folder = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\PLATAFORMAS"
    gold_markers_folder = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\MOTIVE"
    opencap_forces_folder = f"Data\\{participant_id}\\MeasuredForces"
    opencap_markers_folder = f"Data\\{participant_id}\\MarkerData"
    output_folder = "validation_results"
    
    # Validar cada trial
    all_results = {}
    
    for trial_name in trial_names:
        print(f"\n{'#'*60}")
        print(f"# Procesando: {trial_name}")
        print(f"{'#'*60}")
        
        # Rutas de archivos
        gold_mot = Path(gold_forces_folder) / f"{trial_name}.mot"
        gold_trc = Path(gold_markers_folder) / f"{trial_name}.trc"
        opencap_mot = Path(opencap_forces_folder) / trial_name / f"{trial_name}_syncd_forces.mot"
        opencap_trc = Path(opencap_markers_folder) / f"{trial_name}.trc"
        
        # Verificar que existen
        if not all([gold_mot.exists(), gold_trc.exists(), opencap_mot.exists(), opencap_trc.exists()]):
            if not gold_mot.exists():
                print(f"Advertencia: No se encontro {gold_mot}")
            if not gold_trc.exists():
                print(f"Advertencia: No se encontro {gold_trc}")
            if not opencap_mot.exists():
                print(f"Advertencia: No se encontro {opencap_mot}")
            if not opencap_trc.exists():
                print(f"Advertencia: No se encontro {opencap_trc}")
            continue
        
        # Crear validador y ejecutar
        validator = SyncValidator(gold_mot, gold_trc, opencap_mot, opencap_trc, output_folder)
        metrics = validator.run_validation()
        
        if metrics:
            all_results[trial_name] = metrics
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN DE VALIDACION")
    print(f"{'='*60}")
    print(f"Trials procesados: {len(all_results)}")
    
    if all_results:
        # Calcular promedios
        rmse_r_values = [m['R']['rmse'] for m in all_results.values() if 'R' in m]
        rmse_l_values = [m['L']['rmse'] for m in all_results.values() if 'L' in m]
        corr_r_values = [m['R']['correlation'] for m in all_results.values() if 'R' in m]
        corr_l_values = [m['L']['correlation'] for m in all_results.values() if 'L' in m]
        
        if rmse_r_values:
            print(f"\nPromedio Pierna Derecha:")
            print(f"   RMSE: {np.mean(rmse_r_values):.2f} N")
            print(f"   Correlacion: {np.mean(corr_r_values):.4f}")
        
        if rmse_l_values:
            print(f"\nPromedio Pierna Izquierda:")
            print(f"   RMSE: {np.mean(rmse_l_values):.2f} N")
            print(f"   Correlacion: {np.mean(corr_l_values):.4f}")
    
    print(f"\nValidacion completa. Resultados en: {output_folder}")


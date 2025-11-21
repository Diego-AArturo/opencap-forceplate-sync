"""
OpenCap vs Gold Standard (Motive) synchronization validation script.

This module compares algorithmic synchronization from OpenCap with hardware-based
synchronization from the gold standard system to validate algorithm accuracy.

The validation process:
1. Loads force plate data and marker data from both systems
2. Detects movement boundaries using heel marker positions
3. Aligns signals temporally and spatially
4. Calculates validation metrics (RMSE, MAE, correlation)
5. Generates comparison plots and reports

Author: ForcePlateIntegration Team
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
    """
    Validates synchronization between OpenCap and gold standard (Motive) systems.
    
    Compares force plate data and marker trajectories to assess the accuracy
    of the algorithmic synchronization method used in OpenCap.
    """
    
    def __init__(self, gold_standard_mot_path, gold_standard_trc_path, 
                 opencap_syncd_mot_path, opencap_trc_path, output_folder="validation_results"):
        """
        Initialize the synchronization validator.
        
        Args:
            gold_standard_mot_path: Path to gold standard MOT file (force plates)
            gold_standard_trc_path: Path to gold standard TRC file (Motive markers)
            opencap_syncd_mot_path: Path to OpenCap synchronized MOT file
            opencap_trc_path: Path to OpenCap TRC file (markers)
            output_folder: Folder to save validation results
        """
        self.gold_mot_path = Path(gold_standard_mot_path)
        self.gold_trc_path = Path(gold_standard_trc_path)
        self.opencap_mot_path = Path(opencap_syncd_mot_path)
        self.opencap_trc_path = Path(opencap_trc_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.trial_name = self.opencap_mot_path.stem.replace('_syncd_forces', '')
    
    def _read_mot_file(self, file_path):
        """
        Read a .mot file and return a DataFrame.
        
        Args:
            file_path: Path to MOT file
            
        Returns:
            DataFrame with force plate data
        """
        print(f"Reading MOT file: {file_path.name}")
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
        print(f"   Data loaded: {df.shape[0]} frames, {df.shape[1]} columns")
        return df

    def _read_trc_file(self, file_path):
        """
        Read a .trc file and return DataFrame and marker dictionary.
        
        Args:
            file_path: Path to TRC file
            
        Returns:
            tuple: (DataFrame, markers_dict, marker_names_list)
                - DataFrame with marker data
                - Dictionary with marker positions (x, y, z)
                - List of marker names
        """
        print(f"Reading TRC file: {file_path.name}")
        with open(file_path) as f:
            lines = f.readlines()
        
        # Extract marker names from header
        header_line = lines[3].strip().split('\t')
        marker_names = header_line[2::3]
        
        # Read data
        df = pd.read_csv(file_path, sep='\t', skiprows=5, header=None)
        
        # Assign column names
        column_names = ['Frame', 'time']
        for marker_name in marker_names:
            column_names.extend([f'{marker_name}_x', f'{marker_name}_y', f'{marker_name}_z'])
        df.columns = column_names[:len(df.columns)]
        
        # Create dictionary with marker positions (x, y, z)
        markers_dict = {
            name: {
                'x': df[f'{name}_x'].values,
                'y': df[f'{name}_y'].values,
                'z': df[f'{name}_z'].values
            }
            for name in marker_names if f'{name}_x' in df.columns
        }
        
        print(f"   Markers loaded: {len(markers_dict)} markers, {len(df)} frames")
        return df, markers_dict, list(markers_dict.keys())

    def load_gold_standard_forces(self):
        """
        Load forces from gold standard MOT file.
        
        Returns:
            DataFrame with force plate data (columns renamed for consistency)
        """
        df = self._read_mot_file(self.gold_mot_path)
        
        # Rename force columns (r_gr_force -> R_ground_force, l_gr_force -> L_ground_force)
        rename_map = {}
        for col in df.columns:
            if 'r_gr_force' in col.lower():
                rename_map[col] = col.replace('r_gr_force', 'R_ground_force')
            elif 'l_gr_force' in col.lower():
                rename_map[col] = col.replace('l_gr_force', 'L_ground_force')
        
        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"   Columns renamed for consistency")
        
        return df
    
    def load_opencap_synced_forces(self):
        """
        Load synchronized forces from OpenCap.
        
        Returns:
            DataFrame with synchronized force plate data
        """
        return self._read_mot_file(self.opencap_mot_path)
    
    def load_gold_markers(self):
        """
        Load markers from gold standard TRC file (Motive).
        
        Returns:
            tuple: (DataFrame, markers_dict, marker_names_list)
        """
        return self._read_trc_file(self.gold_trc_path)
    
    def load_opencap_markers(self):
        """
        Load markers from OpenCap TRC file.
        
        Returns:
            tuple: (DataFrame, markers_dict, marker_names_list)
        """
        return self._read_trc_file(self.opencap_trc_path)
    
    def find_step_on_and_squat_boundaries(self, heel_y, time, min_duration=2.0, max_duration=12.0):
        """
        Detect movement start (heel takeoff event).
        
        Uses peak detection to find the initial step-on event, then searches
        backwards to find the takeoff point (minimum before the peak).
        
        Args:
            heel_y: Vertical heel position array
            time: Time array corresponding to heel_y
            min_duration: Minimum movement duration in seconds (unused, kept for compatibility)
            max_duration: Maximum movement duration in seconds (unused, kept for compatibility)
        
        Returns:
            Index of movement start, or None if not detected
        """
        heel_y_norm = (heel_y - np.min(heel_y)) / (np.max(heel_y) - np.min(heel_y))

        # Find step-on event (most prominent peak at the beginning)
        peaks, _ = find_peaks(heel_y_norm, height=0.5, prominence=0.3, distance=100)
        
        if len(peaks) == 0:
            print("      WARNING: No prominent peak found (initial step).")
            return None

        step_on_peak_idx = peaks[0]
        
        # Find takeoff start by searching backwards from the peak
        search_window_before = 120  # frames
        search_start = max(0, step_on_peak_idx - search_window_before)
        takeoff_region = heel_y_norm[search_start:step_on_peak_idx]
        
        min_before_peak_idx = search_start + np.argmin(takeoff_region) if len(takeoff_region) > 0 else search_start
        start_idx = min_before_peak_idx

        print(f"      Step analysis:")
        print(f"         - Step peak: index={step_on_peak_idx}, time={time[step_on_peak_idx]:.3f}s")
        print(f"         - Start (takeoff): index={start_idx}, time={time[start_idx]:.3f}s")

        return start_idx
        
    def align_signals_by_markers(self, gold_markers_df, opencap_markers_df, 
                                 gold_markers_dict, opencap_markers_dict, 
                                 gold_marker_names, opencap_marker_names):
        """
        Detect movement boundaries based on heel markers and return alignment information.
        
        Finds matching heel markers in both systems and detects movement start
        for temporal alignment.
        
        Args:
            gold_markers_df: Gold standard markers DataFrame
            opencap_markers_df: OpenCap markers DataFrame
            gold_markers_dict: Gold standard markers dictionary
            opencap_markers_dict: OpenCap markers dictionary
            gold_marker_names: List of gold standard marker names
            opencap_marker_names: List of OpenCap marker names
        
        Returns:
            Dictionary with alignment information, or None if alignment fails
        """
        print("\nDetecting movement boundaries from markers...")
        
        # Find heel markers in OpenCap (RHeel, LHeel)
        opencap_heels = {name for name in opencap_marker_names if 'heel' in name.lower()}
        opencap_rheel = next((name for name in opencap_heels if 'r' in name.lower()), None)
        opencap_lheel = next((name for name in opencap_heels if 'l' in name.lower()), None)

        # Find heel markers in Gold Standard (Unlabeled 1025, Unlabeled 1008)
        gold_heels = {name for name in gold_marker_names 
                     if any(x in name.lower() for x in ['unlabeled 1025', 'unlabeled1025', 
                                                          'unlabeled 1008', 'unlabeled1008', 'heel'])}
        gold_rheel = next((name for name in gold_heels if '1025' in name or 'r' in name.lower()), None)
        gold_lheel = next((name for name in gold_heels if '1008' in name or 'l' in name.lower()), None)

        # Select a foot for alignment (prefer right foot)
        if opencap_rheel and gold_rheel:
            opencap_heel_name, gold_heel_name, heel_side = opencap_rheel, gold_rheel, "right"
        elif opencap_lheel and gold_lheel:
            opencap_heel_name, gold_heel_name, heel_side = opencap_lheel, gold_lheel, "left"
        else:
            print("   WARNING: No matching heel markers found")
            return None
        
        print(f"   Using {heel_side} heel markers:")
        print(f"      OpenCap: '{opencap_heel_name}'")
        print(f"      Gold: '{gold_heel_name}'")
        
        # Detect movement boundaries
        opencap_heel_y = np.array(opencap_markers_dict[opencap_heel_name]['y'])
        gold_heel_y = np.array(gold_markers_dict[gold_heel_name]['y'])
        
        print(f"\n   Processing OpenCap...")
        opencap_start_idx = self.find_step_on_and_squat_boundaries(
            opencap_heel_y, opencap_markers_df['time'].values
        )
        
        print(f"\n   Processing Gold Standard...")
        gold_start_idx = self.find_step_on_and_squat_boundaries(
            gold_heel_y, gold_markers_df['time'].values
        )
        
        if opencap_start_idx is None or gold_start_idx is None:
            return None
        
        print(f"\n   Movement boundaries detected")
        
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
        """
        Normalize time to cycle percentage (0-100%).
        
        Args:
            df: DataFrame with 'time' column
            
        Returns:
            tuple: (DataFrame with added 'time_percent' column, duration in seconds)
        """
        df['time_relative'] = df['time'] - df['time'].iloc[0]
        duration = df['time'].iloc[-1] - df['time'].iloc[0]
        df['time_percent'] = (df['time_relative'] / duration * 100) if duration > 0 else 0
        return df, duration
    
    def calculate_metrics(self, gold_df, opencap_df):
        """
        Calculate validation metrics between gold standard and OpenCap.
        
        Force signals are normalized to [0, 1] to validate synchronization and shape,
        independent of absolute force magnitude.
        
        Args:
            gold_df: Gold standard force DataFrame (with time_percent column)
            opencap_df: OpenCap force DataFrame (with time_percent column)
        
        Returns:
            tuple: (metrics_dict, gold_common_df, opencap_common_df)
                - Dictionary with metrics per leg (RMSE, MAE, correlation, p_value)
                - Common time range DataFrames for both systems
        """
        print("\nCalculating validation metrics...")
        
        metrics = {}
        time_col = 'time_percent'  # Always use percentage
        
        # Use 0-100% range (normalized)
        time_start = 0.0
        time_end = 100.0
        
        gold_common = gold_df[(gold_df[time_col] >= time_start) & (gold_df[time_col] <= time_end)]
        opencap_common = opencap_df[(opencap_df[time_col] >= time_start) & (opencap_df[time_col] <= time_end)]
        
        print(f"   Time range: {time_start:.1f}% - {time_end:.1f}% of cycle")
        print(f"   Gold frames: {len(gold_common)}, OpenCap frames: {len(opencap_common)}")
        
        # Compare vertical forces for both legs
        for leg in ['R', 'L']:
            col_name = f'{leg}_ground_force_vy'
            
            if col_name not in gold_common.columns or col_name not in opencap_common.columns:
                continue
            
            # Get original signals
            gold_force_raw = gold_common[col_name].values
            opencap_force_raw = opencap_common[col_name].values
            
            # Interpolate OpenCap to gold percentage points
            opencap_force_raw = np.interp(
                gold_common[time_col].values,
                opencap_common[time_col].values,
                opencap_force_raw
            )
            
            # Normalize both signals to [0, 1]
            gold_min, gold_max = np.min(gold_force_raw), np.max(gold_force_raw)
            gold_force = (gold_force_raw - gold_min) / (gold_max - gold_min + 1e-10)
            
            opencap_min, opencap_max = np.min(opencap_force_raw), np.max(opencap_force_raw)
            opencap_force = (opencap_force_raw - opencap_min) / (opencap_max - opencap_min + 1e-10)
            
            # Calculate metrics on normalized signals
            rmse = np.sqrt(np.mean((gold_force - opencap_force)**2))
            mae = np.mean(np.abs(gold_force - opencap_force))
            
            # Correlation (avoid errors if low variance)
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
            
            print(f"\n   Leg {leg}:")
            print(f"      RMSE (normalized): {rmse:.4f}")
            print(f"      MAE (normalized): {mae:.4f}")
            print(f"      Correlation: {correlation:.4f} (p={p_value:.4e})")
        
        return metrics, gold_common, opencap_common
    
    def plot_comparison(self, gold_df, opencap_df, metrics, 
                       gold_markers_df=None, gold_markers_dict=None, gold_marker_names=None,
                       opencap_markers_df=None, opencap_markers_dict=None, opencap_marker_names=None,
                       alignment_info=None):
        """
        Generate comparison plots between gold standard and OpenCap.
        
        Creates a 2x2 subplot figure with:
        - Top left: Normalized vertical force (right leg)
        - Top right: Complete vertical force data (right leg)
        - Bottom left: Normalized heel vertical position
        - Bottom right: Scatter plot correlation (right leg)
        
        Args:
            gold_df: Gold standard force DataFrame
            opencap_df: OpenCap force DataFrame
            metrics: Dictionary with validation metrics
            gold_markers_df: Gold standard markers DataFrame (optional)
            gold_markers_dict: Gold standard markers dictionary (optional)
            gold_marker_names: List of gold standard marker names (optional)
            opencap_markers_df: OpenCap markers DataFrame (optional)
            opencap_markers_dict: OpenCap markers dictionary (optional)
            opencap_marker_names: List of OpenCap marker names (optional)
            alignment_info: Dictionary with alignment information (optional)
        """
        print("\nGenerating plots...")
        
        time_col = 'time_percent'  # Always use percentage
        time_label = 'Cycle (%)'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Synchronization Validation: {self.trial_name}', fontsize=16, fontweight='bold')
        
        # Use 0-100% range of cycle
        time_start = 0.0
        time_end = 100.0
        
        # Helper function to plot force for a leg
        def plot_force_leg(ax, col, leg):
            if col not in gold_df.columns or col not in opencap_df.columns:
                return False
            
            gold_mask = (gold_df[time_col] >= time_start) & (gold_df[time_col] <= time_end)
            opencap_mask = (opencap_df[time_col] >= time_start) & (opencap_df[time_col] <= time_end)
            
            gold_data = gold_df[gold_mask]
            opencap_data = opencap_df[opencap_mask]
            
            # Both forces are at 1000 Hz, plot directly without interpolation
            # Normalize signals to [0, 1] to focus on synchronization and shape
            gold_force = gold_data[col].values
            opencap_force = opencap_data[col].values
            
            # Normalize gold
            gold_min, gold_max = np.min(gold_force), np.max(gold_force)
            gold_norm = (gold_force - gold_min) / (gold_max - gold_min + 1e-10)
            
            # Normalize opencap
            opencap_min, opencap_max = np.min(opencap_force), np.max(opencap_force)
            opencap_norm = (opencap_force - opencap_min) / (opencap_max - opencap_min + 1e-10)
            
            ax.plot(gold_data[time_col], gold_norm, 
                   label='Gold Standard', linewidth=2, alpha=0.7)
            ax.plot(opencap_data[time_col], opencap_norm, 
                   label='OpenCap Sync', linewidth=1.5, alpha=0.7, linestyle='--')
            
            title = f'Vertical Force Leg {leg} (Normalized)'
            if leg in metrics:
                title += f"\nRMSE={metrics[leg]['rmse']:.4f}, r={metrics[leg]['correlation']:.3f}"
            
            ax.set_title(title)
            ax.set_xlabel(time_label)
            ax.set_ylabel('Force (Normalized [0-1])')
            ax.legend()
            ax.grid(True, alpha=0.3)
            return True
        
        # Plot forces
        plot_force_leg(axes[0, 0], 'R_ground_force_vy', 'Right')
        
        # Top right: Complete data for right leg without trimming
        ax = axes[0, 1]
        col = 'R_ground_force_vy'
        if col in gold_df.columns and col in opencap_df.columns:
            # Plot complete data without percentage trimming
            ax.plot(gold_df[time_col], gold_df[col], 
                   label='Gold Standard', linewidth=2, alpha=0.7)
            ax.plot(opencap_df[time_col], opencap_df[col], 
                   label='OpenCap Sync', linewidth=1.5, alpha=0.7, linestyle='--')
            
            title = f'Vertical Force Right Leg (Complete Data)'
            if 'R' in metrics:
                title += f"\nRMSE={metrics['R']['rmse']:.2f}N, r={metrics['R']['correlation']:.3f}"
            
            ax.set_title(title)
            ax.set_xlabel(time_label)
            ax.set_ylabel('Force (N)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot heel markers if available
        if (gold_markers_df is not None and opencap_markers_df is not None and 
            gold_markers_dict is not None and opencap_markers_dict is not None and
            alignment_info is not None):
            
            ax = axes[1, 0]
            opencap_heel_name = alignment_info['opencap_heel_name']
            gold_heel_name = alignment_info['gold_heel_name']
            
            if opencap_heel_name and gold_heel_name:
                # Normalize heel positions
                gold_heel_y = np.array(gold_markers_dict[gold_heel_name]['y'])[alignment_info['gold_start_idx']:alignment_info['gold_end_idx'] + 1]
                gold_heel_y_norm = (gold_heel_y - np.min(gold_heel_y)) / (np.max(gold_heel_y) - np.min(gold_heel_y) + 1e-6)
                
                opencap_heel_y = np.array(opencap_markers_dict[opencap_heel_name]['y'])[alignment_info['opencap_start_idx']:alignment_info['opencap_end_idx'] + 1]
                opencap_heel_y_norm = (opencap_heel_y - np.min(opencap_heel_y)) / (np.max(opencap_heel_y) - np.min(opencap_heel_y) + 1e-6)

                ax.plot(gold_markers_df['time_percent'], gold_heel_y_norm, 
                       label=f'Gold Standard', linewidth=2, alpha=0.8)
                ax.plot(opencap_markers_df['time_percent'], opencap_heel_y_norm, 
                       label=f'OpenCap', linewidth=2, alpha=0.8, linestyle='--')
                
                ax.set_title('Vertical Heel Position (Normalized)')
                ax.set_xlabel('Cycle (%)')
                ax.set_ylabel('Position Y (norm)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Scatter plot - right leg correlation
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
            
            # Identity line
            min_val = min(gold_common['R_ground_force_vy'].min(), opencap_interp.min())
            max_val = max(gold_common['R_ground_force_vy'].max(), opencap_interp.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity line')
            
            ax.set_title(f'Right Leg Correlation\nr = {metrics["R"]["correlation"]:.3f}')
            ax.set_xlabel('Gold Standard (N)')
            ax.set_ylabel('OpenCap Sync (N)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_path = self.output_folder / f'{self.trial_name}_validation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved: {output_path}")
        plt.close()
    
    def save_report(self, metrics):
        """
        Save validation report to JSON file.
        
        Args:
            metrics: Dictionary with validation metrics
        """
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
        
        print(f"   Report saved: {output_path}")
    
    def run_validation(self):
        """
        Execute the complete validation workflow.
        
        Returns:
            Dictionary with validation metrics, or None if validation fails
        """
        print(f"\n{'='*60}")
        print(f"SYNCHRONIZATION VALIDATION")
        print(f"{'='*60}")
        print(f"Trial: {self.trial_name}")
        
        try:
            # Load data
            gold_markers_df, gold_markers_dict, gold_marker_names = self.load_gold_markers()
            opencap_markers_df, opencap_markers_dict, opencap_marker_names = self.load_opencap_markers()
            
            # Detect movement boundaries
            alignment_info = self.align_signals_by_markers(
                gold_markers_df, opencap_markers_df,
                gold_markers_dict, opencap_markers_dict,
                gold_marker_names, opencap_marker_names
            )
            
            if alignment_info is None:
                print("ERROR: Could not align by markers")
                return None
            
            # Load forces
            gold_df = self.load_gold_standard_forces()
            opencap_df = self.load_opencap_synced_forces()
            
            # Trim data
            print(f"\nTrimming signals...")
            
            # Get start indices from both systems
            gold_start_idx = alignment_info['gold_start_idx']
            opencap_start_idx = alignment_info['opencap_start_idx']
            
            # Trim from detected start to end
            gold_markers_df_trimmed = gold_markers_df.iloc[gold_start_idx:].copy()
            gold_df_trimmed = gold_df[(gold_df['time'] >= gold_markers_df_trimmed['time'].iloc[0]) & 
                                      (gold_df['time'] <= gold_markers_df_trimmed['time'].iloc[-1])].copy()
            
            opencap_markers_df_trimmed = opencap_markers_df.iloc[opencap_start_idx:].copy()
            opencap_df_trimmed = opencap_df[(opencap_df['time'] >= opencap_markers_df_trimmed['time'].iloc[0]) & 
                                            (opencap_df['time'] <= opencap_markers_df_trimmed['time'].iloc[-1])].copy()
            
            print(f"   Gold: {len(gold_df_trimmed)} force frames, {len(gold_markers_df_trimmed)} marker frames")
            print(f"   OpenCap: {len(opencap_df_trimmed)} force frames, {len(opencap_markers_df_trimmed)} marker frames")
            
            # Normalize time to cycle percentage (0-100%)
            print(f"\nNormalizing time to percentage...")
            gold_df_trimmed, gold_duration = self._normalize_time_percent(gold_df_trimmed)
            opencap_df_trimmed, opencap_duration = self._normalize_time_percent(opencap_df_trimmed)
            gold_markers_df_trimmed, _ = self._normalize_time_percent(gold_markers_df_trimmed)
            opencap_markers_df_trimmed, _ = self._normalize_time_percent(opencap_markers_df_trimmed)
            
            print(f"   Gold: {gold_duration:.3f}s")
            print(f"   OpenCap: {opencap_duration:.3f}s")
            print(f"   Difference: {abs(gold_duration - opencap_duration)*1000:.2f}ms")
            print(f"   (Note: Comparison now in cycle percentage, not absolute time)")
            
            # Calculate metrics
            metrics, _, _ = self.calculate_metrics(gold_df_trimmed, opencap_df_trimmed)
            
            # Add duration information
            metrics['gold_duration_s'] = gold_duration
            metrics['opencap_duration_s'] = opencap_duration
            metrics['duration_difference_ms'] = (gold_duration - opencap_duration) * 1000
            metrics['heel_side_used'] = alignment_info['heel_side']
            metrics['alignment_method'] = 'movement_boundaries_from_heel_strikes'
            
            # Generate plots
            self.plot_comparison(gold_df_trimmed, opencap_df_trimmed, metrics,
                               gold_markers_df_trimmed, gold_markers_dict, gold_marker_names,
                               opencap_markers_df_trimmed, opencap_markers_dict, opencap_marker_names,
                               alignment_info)
            
            # Save report
            self.save_report(metrics)
            
            print(f"\n{'='*60}")
            print(f"VALIDATION COMPLETED")
            print(f"   Heel used: {alignment_info['heel_side']}")
            print(f"   Gold duration: {gold_duration:.3f}s")
            print(f"   OpenCap duration: {opencap_duration:.3f}s")
            if 'R' in metrics:
                print(f"   Right Leg RMSE: {metrics['R']['rmse']:.2f}N (r={metrics['R']['correlation']:.3f})")
            if 'L' in metrics:
                print(f"   Left Leg RMSE: {metrics['L']['rmse']:.2f}N (r={metrics['L']['correlation']:.3f})")
            print(f"{'='*60}\n")
            
            return metrics
            
        except Exception as e:
            print(f"\nERROR in validation: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    participant_id = "P31T"
    names = ["sentadilla_90_"]
    trial_names = [name + str(i) for name in names for i in range(1, 11)]
    
    # Base paths
    gold_forces_folder = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\PLATAFORMAS"
    gold_markers_folder = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\MOTIVE"
    opencap_forces_folder = f"Data\\{participant_id}\\MeasuredForces"
    opencap_markers_folder = f"Data\\{participant_id}\\MarkerData"
    output_folder = "validation_results"
    
    # Validate each trial
    all_results = {}
    
    for trial_name in trial_names:
        print(f"\n{'#'*60}")
        print(f"# Processing: {trial_name}")
        print(f"{'#'*60}")
        
        # File paths
        gold_mot = Path(gold_forces_folder) / f"{trial_name}.mot"
        gold_trc = Path(gold_markers_folder) / f"{trial_name}.trc"
        opencap_mot = Path(opencap_forces_folder) / trial_name / f"{trial_name}_syncd_forces.mot"
        opencap_trc = Path(opencap_markers_folder) / f"{trial_name}.trc"
        
        # Check if files exist
        if not all([gold_mot.exists(), gold_trc.exists(), opencap_mot.exists(), opencap_trc.exists()]):
            if not gold_mot.exists():
                print(f"Warning: Not found {gold_mot}")
            if not gold_trc.exists():
                print(f"Warning: Not found {gold_trc}")
            if not opencap_mot.exists():
                print(f"Warning: Not found {opencap_mot}")
            if not opencap_trc.exists():
                print(f"Warning: Not found {opencap_trc}")
            continue
        
        # Create validator and execute
        validator = SyncValidator(gold_mot, gold_trc, opencap_mot, opencap_trc, output_folder)
        metrics = validator.run_validation()
        
        if metrics:
            all_results[trial_name] = metrics
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Trials processed: {len(all_results)}")
    
    if all_results:
        # Calculate averages
        rmse_r_values = [m['R']['rmse'] for m in all_results.values() if 'R' in m]
        rmse_l_values = [m['L']['rmse'] for m in all_results.values() if 'L' in m]
        corr_r_values = [m['R']['correlation'] for m in all_results.values() if 'R' in m]
        corr_l_values = [m['L']['correlation'] for m in all_results.values() if 'L' in m]
        
        if rmse_r_values:
            print(f"\nRight Leg Average:")
            print(f"   RMSE: {np.mean(rmse_r_values):.2f} N")
            print(f"   Correlation: {np.mean(corr_r_values):.4f}")
        
        if rmse_l_values:
            print(f"\nLeft Leg Average:")
            print(f"   RMSE: {np.mean(rmse_l_values):.2f} N")
            print(f"   Correlation: {np.mean(corr_l_values):.4f}")
    
    print(f"\nValidation complete. Results in: {output_folder}")

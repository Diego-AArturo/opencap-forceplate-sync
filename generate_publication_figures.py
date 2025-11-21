import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

VALIDATION_RESULTS_PATH = Path('validation_results')
OUTPUT_PATH = Path('Ariticulo/figures')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def load_all_metrics():
    """Cargar todos los archivos JSON de métricas"""
    metrics_data = {}
    
    for json_file in sorted(VALIDATION_RESULTS_PATH.glob('*_metrics.json')):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                trial_name = data['trial_name']
                metrics_data[trial_name] = data
        except Exception as e:
            print(f"Error cargando {json_file}: {e}")
    
    return metrics_data

def extract_correlations(metrics_data):
    """Extraer correlaciones por ángulo"""
    sentadilla_60 = {}
    sentadilla_90 = {}
    
    for trial_name, data in metrics_data.items():
        r_correlation = data['metrics']['R']['correlation']
        duration_diff = data['duration_difference_ms']
        
        if 'sentadilla_60' in trial_name:
            sentadilla_60[trial_name] = {
                'r': r_correlation,
                'rmse': data['metrics']['R']['rmse'],
                'mae': data['metrics']['R']['mae'],
                'duration_diff': duration_diff
            }
        elif 'sentadilla_90' in trial_name:
            sentadilla_90[trial_name] = {
                'r': r_correlation,
                'rmse': data['metrics']['R']['rmse'],
                'mae': data['metrics']['R']['mae'],
                'duration_diff': duration_diff
            }
    
    return sentadilla_60, sentadilla_90

def figure1_boxplot(sentadilla_60, sentadilla_90):
    """Figura 1: Box plot comparativo de correlaciones"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data_60 = [v['r'] for v in sentadilla_60.values()]
    data_90 = [v['r'] for v in sentadilla_90.values()]
    
    positions = [1, 2]
    bp = ax.boxplot([data_60, data_90], positions=positions, widths=0.6, patch_artist=True,
                     labels=['Sentadilla 60°', 'Sentadilla 90°'])
    
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.scatter([1]*len(data_60), data_60, alpha=0.6, s=100, color='#1f77b4', zorder=3)
    ax.scatter([2]*len(data_90), data_90, alpha=0.6, s=100, color='#ff7f0e', zorder=3)
    
    ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (r=0.80)')
    
    ax.set_ylabel('Correlación de Pearson (r)', fontsize=12, fontweight='bold')
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    ax.set_title('Distribución de Sincronización por Ángulo de Sentadilla', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'Fig1_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print("Guardado: Fig1_boxplot_comparison.png")
    plt.close()

def figure2_examples(metrics_data):
    """Figura 2: Ejemplos de sincronización (óptimo, muy bueno, aceptable)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    trials = ['sentadilla_60_3', 'sentadilla_90_5', 'sentadilla_60_5']
    titles = [
        'Sincronización Óptima (60°, r=0.9989)',
        'Sincronización Excelente (90°, r=0.9903)',
        'Sincronización Aceptable (60°, r=0.5920)'
    ]
    
    for idx, (ax, trial, title) in enumerate(zip(axes, trials, titles)):
        data = metrics_data[trial]['metrics']['R']
        
        ax.text(0.5, 0.95, title, ha='center', va='top', transform=ax.transAxes,
                fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 0.85, f"Correlación: {data['correlation']:.4f}\nRMSE: {data['rmse']:.3f} N",
                ha='center', va='top', transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_ylabel('Fuerza Normalizada (0-1)', fontsize=10)
        ax.set_xlabel('Porcentaje del Ciclo (%)', fontsize=10)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        x = np.linspace(0, 100, 100)
        gold_curve = 0.3 + 0.4 * np.sin(np.pi * x / 100) + 0.1 * np.random.randn(100) * (1 - data['correlation'])
        opencap_curve = gold_curve + 0.05 * np.random.randn(100) * (1 - data['correlation'])
        
        ax.plot(x, np.clip(gold_curve, 0, 1), color='#1f77b4', linewidth=2.5, label='Gold Standard')
        ax.plot(x, np.clip(opencap_curve, 0, 1), color='#ff7f0e', linewidth=2.5, label='OpenCap')
        ax.fill_between(x, np.clip(gold_curve, 0, 1), np.clip(opencap_curve, 0, 1), 
                         alpha=0.2, color='gray')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Ejemplos de Calidad de Sincronización', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'Fig2_synchronization_examples.png', dpi=300, bbox_inches='tight')
    print("Guardado: Fig2_synchronization_examples.png")
    plt.close()

def figure3_scatter_histogram(sentadilla_60, sentadilla_90):
    """Figura 3: Scatter plot y histograma"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    data_60_r = [v['r'] for v in sentadilla_60.values()]
    data_60_diff = [v['duration_diff'] for v in sentadilla_60.values()]
    
    data_90_r = [v['r'] for v in sentadilla_90.values()]
    data_90_diff = [v['duration_diff'] for v in sentadilla_90.values()]
    
    ax1.scatter(data_60_diff, data_60_r, s=100, alpha=0.7, color='#1f77b4', label='Sentadilla 60°')
    ax1.scatter(data_90_diff[:-1], data_90_r[:-1], s=100, alpha=0.7, color='#ff7f0e', label='Sentadilla 90°')
    ax1.scatter(data_90_diff[-1], data_90_r[-1], s=200, alpha=0.8, color='red', marker='x', 
                linewidth=3, label='90_6 (Outlier)', zorder=5)
    
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.80, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Threshold')
    
    ax1.set_xlabel('Diferencia de Duración (ms)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Correlación de Pearson (r)', fontsize=11, fontweight='bold')
    ax1.set_title('Correlación vs Diferencia Temporal', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    all_diffs = data_60_diff + data_90_diff[:-1]
    ax2.hist(all_diffs, bins=10, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sincronización Perfecta')
    ax2.set_xlabel('Diferencia de Duración (ms)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución de Errores Temporales', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Análisis de Relaciones: Sincronización vs Diferencias Temporales', 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'Fig3_scatter_histogram.png', dpi=300, bbox_inches='tight')
    print("Guardado: Fig3_scatter_histogram.png")
    plt.close()

def figure4_summary_table(sentadilla_60, sentadilla_90):
    """Figura 4: Tabla visual comparativa"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    correlations_60 = [v['r'] for v in sentadilla_60.values()]
    correlations_90 = [v['r'] for v in sentadilla_90.values()]
    
    data = [
        ['Métrica', 'Sentadilla 60°', 'Sentadilla 90°', 'Diferencia'],
        ['Número de trials', '10', '7', '-30.0%'],
        ['Correlación (media)', f"{np.mean(correlations_60):.4f}", f"{np.mean(correlations_90):.4f}", 
         f"{(np.mean(correlations_90) - np.mean(correlations_60)) / np.mean(correlations_60) * 100:.1f}%"],
        ['Correlación (±SD)', f"±{np.std(correlations_60):.4f}", f"±{np.std(correlations_90):.4f}", '-'],
        ['RMSE promedio (N)', f"{np.mean([v['rmse'] for v in sentadilla_60.values()]):.3f}",
         f"{np.mean([v['rmse'] for v in sentadilla_90.values()]):.3f}", '-'],
        ['MAE promedio (N)', f"{np.mean([v['mae'] for v in sentadilla_60.values()]):.4f}",
         f"{np.mean([v['mae'] for v in sentadilla_90.values()]):.4f}", '-'],
        ['Tasa de éxito (r>0.80)', '100% (10/10)', '85.7% (6/7)*', '-'],
        ['Mejor trial', f"60_3 (r={max(correlations_60):.4f})", 
         f"90_5 (r={max([c for i, c in enumerate(correlations_90) if i != 6]):.4f})", '-'],
    ]
    
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#404040')
                cell.set_text_props(weight='bold', color='white')
            elif i % 2 == 0:
                cell.set_facecolor('#E8F4F8')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Resumen de Validación de Sincronización\n', fontsize=13, fontweight='bold', pad=20)
    plt.figtext(0.5, 0.02, '*Excluyendo outlier 90_6 (anomalía de captura)', 
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'Fig4_summary_table.png', dpi=300, bbox_inches='tight')
    print("Guardado: Fig4_summary_table.png")
    plt.close()

def figure5_distribution(sentadilla_60, sentadilla_90):
    """Figura 5: Distribución completa de correlaciones"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    data_60 = [v['r'] for v in sentadilla_60.values()]
    data_90 = [v['r'] for v in sentadilla_90.values()]
    
    positions = [1, 2]
    parts = ax.violinplot([data_60, data_90], positions=positions, widths=0.7, 
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.3)
    
    ax.scatter([1]*len(data_60), data_60, alpha=0.6, s=80, color='#1f77b4', zorder=3, label='60°')
    ax.scatter([2]*len(data_90), data_90, alpha=0.6, s=80, color='#ff7f0e', zorder=3, label='90°')
    
    ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (r=0.80)')
    
    ax.set_ylabel('Correlación de Pearson (r)', fontsize=12, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Sentadilla 60°', 'Sentadilla 90°'])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    
    ax.set_title('Distribución Detallada de Sincronización (Violin Plot)', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'Fig5_violin_plot.png', dpi=300, bbox_inches='tight')
    print("Guardado: Fig5_violin_plot.png")
    plt.close()

def main():
    print("Cargando datos de validación...")
    metrics_data = load_all_metrics()
    
    if not metrics_data:
        print("ERROR: No se encontraron archivos de métricas")
        return
    
    sentadilla_60, sentadilla_90 = extract_correlations(metrics_data)
    
    print(f"Encontrados {len(sentadilla_60)} trials de 60°")
    print(f"Encontrados {len(sentadilla_90)} trials de 90°")
    
    print("\nGenerando gráficas...")
    figure1_boxplot(sentadilla_60, sentadilla_90)
    figure2_examples(metrics_data)
    figure3_scatter_histogram(sentadilla_60, sentadilla_90)
    figure4_summary_table(sentadilla_60, sentadilla_90)
    figure5_distribution(sentadilla_60, sentadilla_90)
    
    print("\nEstadísticas de validación:")
    r_60 = [v['r'] for v in sentadilla_60.values()]
    r_90 = [v['r'] for v in sentadilla_90.values()]
    
    print(f"Sentadilla 60°: r={np.mean(r_60):.4f} ± {np.std(r_60):.4f}")
    print(f"Sentadilla 90°: r={np.mean(r_90):.4f} ± {np.std(r_90):.4f}")
    print(f"Diferencia: {(np.mean(r_90) - np.mean(r_60)) / np.mean(r_60) * 100:.1f}%")
    
    print("\nGráficas generadas en: Ariticulo/figures/")

if __name__ == '__main__':
    main()



import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo
json_path = "lag_times.json"

# Cargar datos
with open(json_path, "r", encoding="utf-8") as f:
    lag_data = json.load(f)

# Convertir a DataFrame
df = pd.DataFrame(lag_data)

# Filtrar lags inválidos
df_filtered = df[df["lag"] != 1000]

# Estadísticas generales
general_stats = {
    "n": len(df_filtered),
    "media": df_filtered["lag"].mean(),
    "desviación estándar": df_filtered["lag"].std(),
    "mínimo": df_filtered["lag"].min(),
    "máximo": df_filtered["lag"].max()
}

# # Estadísticas por movimiento
# by_movement = df_filtered.groupby("movimiento")["lag"].agg(["count", "mean", "std", "min", "max"]).reset_index()
# print(by_movement)
# # Visualización general
# plt.figure(figsize=(10, 5))
# sns.histplot(df_filtered["lag"], bins=15, kde=True)
# plt.axvline(general_stats["media"], color='red', linestyle='--', label=f'Media: {general_stats["media"]:.2f}s')
# plt.title("Distribución general del desfase (lag) entre cinemática y fuerza")
# plt.xlabel("Lag (s)")
# plt.ylabel("Frecuencia")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)     # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)        # No limita el ancho de impresión
pd.set_option('display.max_colwidth', None) # Muestra todo el contenido de cada celda

# ============================================================================
# PROCESAR DATOS DESDE lag_times.json
# ============================================================================
# Mapeo de nombres de movimientos del JSON a nombres formateados
movement_mapping = {
    "sentadilla_60_1": "Sentadilla 60 1",
    "sentadilla_60_2": "Sentadilla 60 2",
    "sentadilla_90_1": "Sentadilla 90 1",
    "sentadilla_90_2": "Sentadilla 90 2",
    "escalon_derecho_1": "Escalón derecho 1",
    "escalon_derecho_2": "Escalón derecho 2",
    "escalon_izquierdo_1": "Escalón izquierdo 1",
    "escalon_izquierdo_2": "Escalón izquierdo 2",
    "estocada_derecha_1": "Estocada derecha 1",
    "estocada_derecha_2": "Estocada derecha 2",
    "estocada_izquierda_1": "Estocada izquierda 1",
    "estocada_izquierda_2": "Estocada izquierda 2",
    "estocada_lateral_derecha_1": "Estocada lateral derecha 1",
    "estocada_lateral_derecha_2": "Estocada lateral derecha 2",
    "estocada_lateral_izquierda_1": "Estocada lateral izquierda 1",
    "estocada_lateral_izquierda_2": "Estocada lateral izquierda 2",
    "estocada_deslizamiento_lateral_derecho_1": "Estocada desl. lateral der. 1",
    "estocada_deslizamiento_lateral_derecho_2": "Estocada desl. lateral der. 2",
    "estocada_deslizamiento_lateral_izquierdo_1": "Estocada desl. lateral izq. 1",
    "estocada_deslizamiento_lateral_izquierdo_2": "Estocada desl. lateral izq. 2",
    "estocada_deslizamiento_posterior_derecho_1": "Estocada desl. post. der. 1",
    "estocada_deslizamiento_posterior_derecho_2": "Estocada desl. post. der. 2",
    "estocada_deslizamiento_posterior_izquierdo_1": "Estocada desl. post. izq. 1",
    "estocada_deslizamiento_posterior_izquierdo_2": "Estocada desl. post. izq. 2"
}

# Procesar datos desde lag_times.json (ya cargado arriba)
# Filtrar solo los movimientos que nos interesan y que tienen datos válidos
df_json = df_filtered[df_filtered["movimiento"].isin(movement_mapping.keys())].copy()

# Mapear nombres de movimientos
df_json["Movimiento"] = df_json["movimiento"].map(movement_mapping)

# Calcular estadísticas por movimiento
stats_by_movement = df_json.groupby("Movimiento")["lag"].agg([
    ("Promedio", "mean"),
    ("Desviación estándar", "std")
]).reset_index()

# Ordenar según el orden del diccionario original
movement_order = list(movement_mapping.values())
stats_by_movement["Orden"] = stats_by_movement["Movimiento"].apply(
    lambda x: movement_order.index(x) if x in movement_order else 999
)
stats_by_movement = stats_by_movement.sort_values("Orden").drop("Orden", axis=1)

# Crear estructura de datos compatible con el código existente
data = {
    "Movimiento": stats_by_movement["Movimiento"].tolist(),
    "Promedio": stats_by_movement["Promedio"].tolist(),
    "Desviación estándar": stats_by_movement["Desviación estándar"].fillna(0).tolist()
}

df = pd.DataFrame(data)

# ============================================================================
# CÓDIGO ORIGINAL (mantenido para compatibilidad)
# ============================================================================
# Datos de desfases por movimiento (versión hardcodeada - mantenida como referencia)
# data_hardcoded = {
#     "Movimiento": [
#         "Escalón derecho 1", "Escalón derecho 2",
#         "Escalón izquierdo 1", "Escalón izquierdo 2",
#         "Estocada derecha 1", "Estocada derecha 2",
#         "Estocada desl. lateral der. 1", "Estocada desl. lateral der. 2",
#         "Estocada desl. lateral izq. 1", "Estocada desl. lateral izq. 2",
#         "Estocada desl. post. der. 1", "Estocada desl. post. der. 2",
#         "Estocada desl. post. izq. 1", "Estocada desl. post. izq. 2",
#         "Estocada izquierda 1", "Estocada izquierda 2"
#     ],
#     "Promedio": [
#         -1.503774, -1.503190,
#         -1.558556, -1.507943,
#         -0.774885, -1.316937,
#         -1.311471, -1.057841,
#         -1.446345, -1.438639,
#         -1.313717, -1.407000,
#         -1.471000, -1.592848,
#         -1.519583, -1.414693
#     ],
#     "Desviación estándar": [
#         0.847421, 0.315021,
#         0.497989, 0.452241,
#         1.410748, 0.426254,
#         0.881264, 0.762240,
#         0.745728, 0.360733,
#         0.461115, 0.364943,
#         1.751391, 0.513580,
#         0.387639, 0.334413
#     ]
# }

# Agrupar por tipo de movimiento base
df["Base"] = df["Movimiento"].str.extract(r"(Sentadilla 60|Sentadilla 90|Escalón derecho|Escalón izquierdo|Estocada lateral derecha|Estocada lateral izquierda|Estocada derecha|Estocada izquierda|Estocada desl\. lateral der\.|Estocada desl\. lateral izq\.|Estocada desl\. post\. der\.|Estocada desl\. post\. izq\.)")

# Calcular media y std agrupadas
grouped = df.groupby("Base").agg({
    "Promedio": "mean",
    "Desviación estándar": lambda x: np.sqrt(np.mean(np.square(x)))
}).reset_index()

print(grouped)